//
// Created by q1162 on 2023/3/30.
//
#include "TRTInfer.h"
#include "RetinaFace/common.hpp"

TRTInfer::TRTInfer(std::string modelPath,int w,int h,int o):INPUT_W(w),INPUT_H(h),OUTPUT_SIZE(o),_modelPath(modelPath) {
    std::shared_ptr<float> a(new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W]);
    std::shared_ptr<float> b(new float[BATCH_SIZE * OUTPUT_SIZE]);
    _dataPtr=std::move(a);
    _probPtr=std::move(b);

    char *trtModelStream{nullptr};
    size_t size{0};
    loadModel(&trtModelStream,size);
    //build context
    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    runtime->destroy();
    //ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    _context = engine->createExecutionContext();
    assert(_context != nullptr);

}

void TRTInfer::doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream

    CHECK(cudaStreamCreate(&_stream));
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, _stream));
    context.enqueue(batchSize, buffers, _stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, _stream));
    cudaStreamSynchronize(_stream);
    // Release stream and buffers
    cudaStreamDestroy(_stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void TRTInfer::APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    config->destroy();
    builder->destroy();
}

void TRTInfer::loadModel(char** trtModelStream,size_t& size) {
    namespace fs = std::filesystem;
    fs::path modelPath(_modelPath);
    fs::path cachePath=modelPath;
    cachePath.replace_extension("engine");

    cudaSetDevice(DEVICE);
    if(!fs::exists(cachePath)){
        if(!fs::exists(modelPath)) {
            std::cerr << "Load model fail,please check file path!" << std::endl;
            exit(-1);
        }
        std::cout<<"prepare to build model engine..\nIt will takes a while..."<<std::endl;
        IHostMemory *modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p(cachePath.generic_string(), std::ios::binary);
        assert(p);
        p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
        p.close();
        modelStream->destroy();
    }

    std::ifstream file(cachePath.generic_string(), std::ios::binary);

    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        *trtModelStream = new char[size];
        assert(*trtModelStream);
        file.read(*trtModelStream, size);
        file.close();
    }




}

TRTInfer::StructRst TRTInfer::infer(const cv::Mat& img) {
    static float* data=_dataPtr.get();
    static float* prob=_probPtr.get();

    data=preProcess(img,&data);
    doInference(*_context,data,prob,BATCH_SIZE);

    return std::move(postProcess(prob));
}

float* TRTInfer::preProcess(const cv::Mat& img,float** predata) {
    return nullptr;
}

TRTInfer::StructRst TRTInfer::postProcess(const float *prob) {
    return TRTInfer::StructRst{};
}
