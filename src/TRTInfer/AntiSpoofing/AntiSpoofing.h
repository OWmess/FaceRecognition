//
// Created by q1162 on 2023/3/31.
//

#ifndef FACERECOGNITION_ANTISPOOFING_H
#define FACERECOGNITION_ANTISPOOFING_H
#include <utility>

#include "../TRTInfer.hpp"
using namespace nvinfer1;
class AntiSpoofing final: public TRTInfer{
public:
    AntiSpoofing()=delete;
    AntiSpoofing(std::string modelPath,int w,int h,int o):TRTInfer(modelPath,w,h,o){

    }
    ~AntiSpoofing()=default;

    void process() override;

    void preProcess(const cv::Mat& img,float** predata) override;

    StructRst postProcess(float **prob,int rows,int cols) override;

private:
    ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) override;




};


#endif //FACERECOGNITION_ANTISPOOFING_H
