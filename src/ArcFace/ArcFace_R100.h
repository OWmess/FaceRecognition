//
// Created by q1162 on 2023/3/28.
//

#ifndef FACERECOGNITION_ARCFACE_R100_H
#define FACERECOGNITION_ARCFACE_R100_H
#include "logging.h"
#include "../config.h"
#include "NvInfer.h"
#include <map>
using namespace nvinfer1;
class ArcFace{

public:
    ArcFace()=default;
    ~ArcFace()= default;

    void process();
private:
    std::map<std::string,Weights> loadWeights(const std::string file);

    IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);

    ILayer* addPRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname);

    ILayer* resUnit(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int num_filters, int s, bool dim_match, std::string lname);

    ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);

    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);

    void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
private:
    // stuff we know about the network and the input/output blobs
    static constexpr int INPUT_H = 112;
    static constexpr int INPUT_W = 112;
    static constexpr int OUTPUT_SIZE = 512;
    const char* INPUT_BLOB_NAME = "data";
    const char* OUTPUT_BLOB_NAME = "prob";
    Logger gLogger;
};



#endif //FACERECOGNITION_ARCFACE_R100_H
