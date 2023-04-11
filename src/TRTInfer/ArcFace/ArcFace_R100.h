//
// Created by q1162 on 2023/3/28.
//

#ifndef FACERECOGNITION_ARCFACE_R100_H
#define FACERECOGNITION_ARCFACE_R100_H
#include "logging.h"
#include "../../config.h"
#include "NvInfer.h"
#include <map>
#include "../TRTInfer.hpp"

class __declspec(dllexport) ArcFace final: public TRTInfer{
public:
    ArcFace()=delete;
    ArcFace(std::string modelPath,int w,int h,int o):TRTInfer(modelPath,w,h,o){
        prepareModel();
    }
    ~ArcFace()= default;

    void process() override;

    void preProcess(const cv::Mat& img,float** predata) override;

    StructRst postProcess(float **prob,int rows,int cols) override;

    nvinfer1::ICudaEngine* createEngine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt) override;
private:

    nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, std::string lname, float eps);

    nvinfer1::ILayer* addPRelu(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, std::string lname);

    nvinfer1::ILayer* resUnit(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int num_filters, int s, bool dim_match, std::string lname);




private:
//    // stuff we know about the network and the input/output blobs
//    static constexpr int INPUT_H = 112;
//    static constexpr int INPUT_W = 112;
//    static constexpr int OUTPUT_SIZE = 512;
//    const char* INPUT_BLOB_NAME = "data";
//    const char* OUTPUT_BLOB_NAME = "prob";
//    Logger gLogger;
};



#endif //FACERECOGNITION_ARCFACE_R100_H
