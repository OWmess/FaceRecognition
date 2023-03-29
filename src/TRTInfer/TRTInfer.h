//
// Created by q1162 on 2023/3/28.
//
#ifndef FACERECOGNITION_TRTINFER_H
#define FACERECOGNITION_TRTINFER_H
#include "logging.h"
#include "NvInfer.h"
#include "../config.h"
#include "../utils.h"
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;

class TRTInfer{
public:
    TRTInfer()=delete;
    TRTInfer(int w,int h,int o):INPUT_W(w),INPUT_H(h),OUTPUT_SIZE(o) {
        std::shared_ptr<float> a(new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W]);
        std::shared_ptr<float> b(new float[BATCH_SIZE * OUTPUT_SIZE]);
        dataPtr=std::move(a);
        probPtr=std::move(b);

    }
    virtual ~TRTInfer()= default;

    virtual void process()=0;
    virtual ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)=0;

    void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
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
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
    }

    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
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
        builder->destroy();
    }



protected:
    Logger gLogger;
    // stuff we know about the network and the input/output blobs
    int INPUT_H,INPUT_W,OUTPUT_SIZE;
    const char* INPUT_BLOB_NAME = "data";
    const char* OUTPUT_BLOB_NAME = "prob";
    std::shared_ptr<float> dataPtr;
    std::shared_ptr<float> probPtr;
};


#endif //FACERECOGNITION_TRTINFER_H
