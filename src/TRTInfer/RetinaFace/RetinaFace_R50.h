//
// Created by q1162 on 2023/3/28.
//

#ifndef FACERECOGNITION_RETINAFACE_R50_H
#define FACERECOGNITION_RETINAFACE_R50_H

#include "../../config.h"
#include "logging.h"
#include "NvInfer.h"
#include "../TRTInfer.hpp"
#include "common.hpp"
#include <opencv2/opencv.hpp>
//#include "dirent.h"

#include "decode.h"

class __declspec(dllexport) RetinaFace final: public TRTInfer{
public:
    RetinaFace()=delete;

    RetinaFace(std::string modelPath,int inputW,int inputH,int outputSize): TRTInfer(modelPath,inputW,inputH,outputSize){
        prepareModel();
    }

    ~RetinaFace()=default;

    void process() override;

    void preProcess(const cv::Mat& img,float** predata) override;

    StructRst postProcess(float **prob,int rows,int cols) override;

    virtual nvinfer1::ICudaEngine* createEngine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt) override;
private:
    nvinfer1::IActivationLayer* bottleneck(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int inch, int outch, int stride, std::string lname);

    nvinfer1::ILayer* conv_bn_relu(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int outch, int kernelsize, int stride, int padding, bool userelu, std::string lname);

    nvinfer1::IActivationLayer* ssh(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, std::string lname);

};
#endif //FACERECOGNITION_RETINAFACE_R50_H
