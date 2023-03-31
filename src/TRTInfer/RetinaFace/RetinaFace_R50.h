//
// Created by q1162 on 2023/3/28.
//

#ifndef FACERECOGNITION_RETINAFACE_R50_H
#define FACERECOGNITION_RETINAFACE_R50_H

#include "../../config.h"
#include "logging.h"
#include "common.hpp"
#include <opencv2/opencv.hpp>
#include "dirent.h"
#include "NvInfer.h"
#include "decode.h"
#include "../TRTInfer.h"
class RetinaFace final: public TRTInfer{
public:
    RetinaFace()=delete;

    RetinaFace(std::string modelPath,int inputW,int inputH,int outputSize): TRTInfer(modelPath,inputW,inputH,outputSize){

    }

    ~RetinaFace()=default;

    void process() override;

    void preProcess(const cv::Mat& img,float** predata) override;

    StructRst postProcess(float **prob,int rows,int cols) override;
private:
    IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname);

    ILayer* conv_bn_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int kernelsize, int stride, int padding, bool userelu, std::string lname);

    IActivationLayer* ssh(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname);

    ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) override;

};
#endif //FACERECOGNITION_RETINAFACE_R50_H
