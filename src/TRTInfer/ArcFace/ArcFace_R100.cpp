#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include "dirent.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "../../config.h"
#include "ArcFace_R100.h"
#include "../RetinaFace/common.hpp"
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
IScaleLayer* ArcFace::addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + "_gamma"].values;
    float *beta = (float*)weightMap[lname + "_beta"].values;
    float *mean = (float*)weightMap[lname + "_moving_mean"].values;
    float *var = (float*)weightMap[lname + "_moving_var"].values;
    int len = weightMap[lname + "_moving_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* ArcFace::addPRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname) {
	float *gamma = (float*)weightMap[lname + "_gamma"].values;
	int len = weightMap[lname + "_gamma"].count;

	float *scval_1 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	float *scval_2 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		scval_1[i] = -1.0;
		scval_2[i] = -gamma[i];
	}
	Weights scale_1{ DataType::kFLOAT, scval_1, len };
	Weights scale_2{ DataType::kFLOAT, scval_2, len };

	float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		shval[i] = 0.0;
	}
	Weights shift{ DataType::kFLOAT, shval, len };

	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		pval[i] = 1.0;
	}
	Weights power{ DataType::kFLOAT, pval, len };

	auto relu1 = network->addActivation(input, ActivationType::kRELU);
	assert(relu1);
	IScaleLayer* scale1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale_1, power);
	assert(scale1);
	auto relu2 = network->addActivation(*scale1->getOutput(0), ActivationType::kRELU);
	assert(relu2);
	IScaleLayer* scale2 = network->addScale(*relu2->getOutput(0), ScaleMode::kCHANNEL, shift, scale_2, power);
	assert(scale2);
	IElementWiseLayer* ew1 = network->addElementWise(*relu1->getOutput(0), *scale2->getOutput(0), ElementWiseOperation::kSUM);
	assert(ew1);
	return ew1;
}

ILayer* ArcFace::resUnit(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int num_filters, int s, bool dim_match, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    auto bn1 = addBatchNorm2d(network, weightMap, input, lname + "_bn1", 2e-5);
    IConvolutionLayer* conv1 = network->addConvolutionNd(*bn1->getOutput(0), num_filters, DimsHW{3, 3}, weightMap[lname + "_conv1_weight"], emptywts);
    assert(conv1);
    conv1->setPaddingNd(DimsHW{1, 1});
    auto bn2 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_bn2", 2e-5);
    auto act1 = addPRelu(network, weightMap, *bn2->getOutput(0), lname + "_relu1");
    IConvolutionLayer* conv2 = network->addConvolutionNd(*act1->getOutput(0), num_filters, DimsHW{3, 3}, weightMap[lname + "_conv2_weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{s, s});
    conv2->setPaddingNd(DimsHW{1, 1});
    auto bn3 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "_bn3", 2e-5);

    IElementWiseLayer* ew1;
    if (dim_match) {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        IConvolutionLayer* conv1sc = network->addConvolutionNd(input, num_filters, DimsHW{1, 1}, weightMap[lname + "_conv1sc_weight"], emptywts);
        assert(conv1sc);
        conv1sc->setStrideNd(DimsHW{s, s});
        auto bn1sc = addBatchNorm2d(network, weightMap, *conv1sc->getOutput(0), lname + "_sc", 2e-5);
        ew1 = network->addElementWise(*bn1sc->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    assert(ew1);
    return ew1;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* ArcFace::createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(_modelPath);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv0 = network->addConvolutionNd(*data, 64, DimsHW{3, 3}, weightMap["conv0_weight"], emptywts);
    assert(conv0);
    conv0->setPaddingNd(DimsHW{1, 1});
    auto bn0 = addBatchNorm2d(network, weightMap, *conv0->getOutput(0), "bn0", 2e-5);
    auto relu0 = addPRelu(network, weightMap, *bn0->getOutput(0), "relu0");

    auto s1u1 = resUnit(network, weightMap, *relu0->getOutput(0), 64, 2, false, "stage1_unit1");
    auto s1u2 = resUnit(network, weightMap, *s1u1->getOutput(0), 64, 1, true, "stage1_unit2");
    auto s1u3 = resUnit(network, weightMap, *s1u2->getOutput(0), 64, 1, true, "stage1_unit3");

    auto s2u1 = resUnit(network, weightMap, *s1u3->getOutput(0), 128, 2, false, "stage2_unit1");
    auto s2u2 = resUnit(network, weightMap, *s2u1->getOutput(0), 128, 1, true, "stage2_unit2");
    auto s2u3 = resUnit(network, weightMap, *s2u2->getOutput(0), 128, 1, true, "stage2_unit3");
    auto s2u4 = resUnit(network, weightMap, *s2u3->getOutput(0), 128, 1, true, "stage2_unit4");


    auto s2u5 = resUnit(network, weightMap, *s2u4->getOutput(0), 128, 1, true, "stage2_unit5");
    auto s2u6 = resUnit(network, weightMap, *s2u5->getOutput(0), 128, 1, true, "stage2_unit6");
    auto s2u7 = resUnit(network, weightMap, *s2u6->getOutput(0), 128, 1, true, "stage2_unit7");
    auto s2u8 = resUnit(network, weightMap, *s2u7->getOutput(0), 128, 1, true, "stage2_unit8");

    auto s2u9 = resUnit(network, weightMap, *s2u8->getOutput(0), 128, 1, true, "stage2_unit9");
    auto s2u10 = resUnit(network, weightMap, *s2u9->getOutput(0), 128, 1, true, "stage2_unit10");
    auto s2u11 = resUnit(network, weightMap, *s2u10->getOutput(0), 128, 1, true, "stage2_unit11");
    auto s2u12 = resUnit(network, weightMap, *s2u11->getOutput(0), 128, 1, true, "stage2_unit12");
    auto s2u13 = resUnit(network, weightMap, *s2u12->getOutput(0), 128, 1, true, "stage2_unit13");

    auto s3u1 = resUnit(network, weightMap, *s2u13->getOutput(0), 256, 2, false, "stage3_unit1");
    auto s3u2 = resUnit(network, weightMap, *s3u1->getOutput(0), 256, 1, true, "stage3_unit2");
    auto s3u3 = resUnit(network, weightMap, *s3u2->getOutput(0), 256, 1, true, "stage3_unit3");
    auto s3u4 = resUnit(network, weightMap, *s3u3->getOutput(0), 256, 1, true, "stage3_unit4");
    auto s3u5 = resUnit(network, weightMap, *s3u4->getOutput(0), 256, 1, true, "stage3_unit5");
    auto s3u6 = resUnit(network, weightMap, *s3u5->getOutput(0), 256, 1, true, "stage3_unit6");
    auto s3u7 = resUnit(network, weightMap, *s3u6->getOutput(0), 256, 1, true, "stage3_unit7");
    auto s3u8 = resUnit(network, weightMap, *s3u7->getOutput(0), 256, 1, true, "stage3_unit8");
    auto s3u9 = resUnit(network, weightMap, *s3u8->getOutput(0), 256, 1, true, "stage3_unit9");
    auto s3u10 = resUnit(network, weightMap, *s3u9->getOutput(0), 256, 1, true, "stage3_unit10");
    auto s3u11 = resUnit(network, weightMap, *s3u10->getOutput(0), 256, 1, true, "stage3_unit11");
    auto s3u12 = resUnit(network, weightMap, *s3u11->getOutput(0), 256, 1, true, "stage3_unit12");
    auto s3u13 = resUnit(network, weightMap, *s3u12->getOutput(0), 256, 1, true, "stage3_unit13");
    auto s3u14 = resUnit(network, weightMap, *s3u13->getOutput(0), 256, 1, true, "stage3_unit14");

    auto s3u15 = resUnit(network, weightMap, *s3u14->getOutput(0), 256, 1, true, "stage3_unit15");
    auto s3u16 = resUnit(network, weightMap, *s3u15->getOutput(0), 256, 1, true, "stage3_unit16");
    auto s3u17 = resUnit(network, weightMap, *s3u16->getOutput(0), 256, 1, true, "stage3_unit17");
    auto s3u18 = resUnit(network, weightMap, *s3u17->getOutput(0), 256, 1, true, "stage3_unit18");
    auto s3u19 = resUnit(network, weightMap, *s3u18->getOutput(0), 256, 1, true, "stage3_unit19");
    auto s3u20 = resUnit(network, weightMap, *s3u19->getOutput(0), 256, 1, true, "stage3_unit20");
    auto s3u21 = resUnit(network, weightMap, *s3u20->getOutput(0), 256, 1, true, "stage3_unit21");
    auto s3u22 = resUnit(network, weightMap, *s3u21->getOutput(0), 256, 1, true, "stage3_unit22");
    auto s3u23 = resUnit(network, weightMap, *s3u22->getOutput(0), 256, 1, true, "stage3_unit23");
    auto s3u24 = resUnit(network, weightMap, *s3u23->getOutput(0), 256, 1, true, "stage3_unit24");
    auto s3u25 = resUnit(network, weightMap, *s3u24->getOutput(0), 256, 1, true, "stage3_unit25");
    auto s3u26 = resUnit(network, weightMap, *s3u25->getOutput(0), 256, 1, true, "stage3_unit26");
    auto s3u27 = resUnit(network, weightMap, *s3u26->getOutput(0), 256, 1, true, "stage3_unit27");
    auto s3u28 = resUnit(network, weightMap, *s3u27->getOutput(0), 256, 1, true, "stage3_unit28");
    auto s3u29 = resUnit(network, weightMap, *s3u28->getOutput(0), 256, 1, true, "stage3_unit29");
    auto s3u30 = resUnit(network, weightMap, *s3u29->getOutput(0), 256, 1, true, "stage3_unit30");

    auto s4u1 = resUnit(network, weightMap, *s3u30->getOutput(0), 512, 2, false, "stage4_unit1");
    auto s4u2 = resUnit(network, weightMap, *s4u1->getOutput(0), 512, 1, true, "stage4_unit2");
    auto s4u3 = resUnit(network, weightMap, *s4u2->getOutput(0), 512, 1, true, "stage4_unit3");

    auto bn1 = addBatchNorm2d(network, weightMap, *s4u3->getOutput(0), "bn1", 2e-5);
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*bn1->getOutput(0), 512, weightMap["pre_fc1_weight"], weightMap["pre_fc1_bias"]);
    assert(fc1);
    auto bn2 = addBatchNorm2d(network, weightMap, *fc1->getOutput(0), "fc1", 2e-5);

    bn2->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*bn2->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    size_t freeCuMem, totalCuMem;
    cuMemGetInfo(&freeCuMem, &totalCuMem);
    std::cout << "[INFO]: total gpu mem: " << (totalCuMem >> 20) << "MB, free gpu mem: " << (freeCuMem >> 20) << "MB"
              << std::endl;
    std::cout << "[INFO]: max workspace size will use all of free gpu mem" << std::endl;
    config->setMaxWorkspaceSize(256<<20);
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network anymore
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void ArcFace::process() {

    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    loadModel(&trtModelStream,size);

    // prepare input data ---------------------------
    static float* data=_dataPtr.get();
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float* prob=_probPtr.get();
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    cv::Mat img = cv::imread("../pic0.jpg");
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
    }

    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(data, prob);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    cv::Mat out(512, 1, CV_32FC1, prob);
    cv::Mat out_norm;
    cv::normalize(out, out_norm);

    img = cv::imread("../pic1.jpg");
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
    }

    // Run inference
    start = std::chrono::system_clock::now();
    doInference(data, prob);
    end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    cv::Mat out1(1, 512, CV_32FC1, prob);
    cv::Mat out_norm1;
    cv::normalize(out1, out_norm1);

    cv::Mat res = out_norm1 * out_norm;

    std::cout << "similarity score: " << *(float*)res.data << std::endl;

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    //Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << p_out_norm[i] << ", ";
    //    if (i % 10 == 0) std::cout << i / 10 << std::endl;
    //}
    //std::cout << std::endl;

}

void ArcFace::preProcess(const cv::Mat &img, float **predata) {
    float* &data=*predata;
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
    }
}

TRTInfer::StructRst ArcFace::postProcess(float **prob,int rows,int cols) {
    assert(rows!=0&&cols!=0);
    StructRst result;
    cv::Mat out(rows, cols, CV_32FC1, *prob);
    cv::Mat out_norm;
    cv::normalize(out, result.embedding);
    return result;
}
