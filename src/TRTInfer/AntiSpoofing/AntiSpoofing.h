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
    AntiSpoofing(std::string modelPath,int w,int h,int o);
    ~AntiSpoofing()=default;

    void process() override;

    void preProcess(const cv::Mat& img,float** predata) override;

    StructRst postProcess(float **prob,int rows,int cols) override;

    virtual StructRst infer(const cv::Mat &img,decodeplugin::Detection bbox,float thresh) override;

    virtual StructRst infer(const cv::Mat &img,int rows=0,int cols=0) override;
private:
    ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) override;

    void doInference(float* input, float* output) override;

    inline cv::Mat getRoi(const cv::Mat& src, float bbox[], float scale);

};


#endif //FACERECOGNITION_ANTISPOOFING_H
