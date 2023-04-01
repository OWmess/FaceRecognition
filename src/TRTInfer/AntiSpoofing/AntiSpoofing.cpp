//
// Created by q1162 on 2023/3/31.
//

#include "AntiSpoofing.h"
#include <NvOnnxParser.h>
#include <cuda.h>
ICudaEngine* AntiSpoofing::createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt){

    auto network=builder->createNetworkV2(0U);
    assert(network!= nullptr);
    auto parser=nvonnxparser::createParser(*network,gLogger);
    assert(parser!= nullptr);
    auto parsingSuccess=parser->parseFromFile(_modelPath.c_str(),static_cast<int>(gLogger.getReportableSeverity()));
    assert(parsingSuccess==true);

#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif

    size_t free, total;
    cuMemGetInfo(&free, &total);
    std::cout << "[INFO]: total gpu mem: " << (total >> 20) << "MB, free gpu mem: " << (free >> 20) << "MB"
              << std::endl;
    std::cout << "[INFO]: max workspace size will use all of free gpu mem" << std::endl;
    config->setMaxWorkspaceSize(free);
    std::cout << "Building engine, please wait for a while..." << std::endl;
    auto engine=builder->buildEngineWithConfig(*network,*config);
    std::cout << "Build engine successfully!" << std::endl;
    network->destroy();

    return engine;
}

void AntiSpoofing::process() {

}

void AntiSpoofing::preProcess(const cv::Mat &img, float **predata) {

}

TRTInfer::StructRst AntiSpoofing::postProcess(float **prob, int rows, int cols) {
    return TRTInfer::StructRst();
}
