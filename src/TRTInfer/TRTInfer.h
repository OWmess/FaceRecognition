//
// Created by q1162 on 2023/3/28.
//
#ifndef FACERECOGNITION_TRTINFER_H
#define FACERECOGNITION_TRTINFER_H
#include <utility>
#include "logging.h"
#include "NvInfer.h"
#include "../config.h"
#include "../utils.h"
#include "RetinaFace/decode.h"
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
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
    struct StructRst{
        std::vector<decodeplugin::Detection> detector;
        cv::Mat embedding;
    };
    TRTInfer()=delete;

    TRTInfer(std::string modelPath,int w,int h,int o);

    virtual ~TRTInfer()= default;

    virtual void process()=0;

    virtual ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)=0;

    virtual float* preProcess(const cv::Mat& img,float** predata);

    virtual StructRst postProcess(const float* prob);


    void doInference(IExecutionContext& context, float* input, float* output, int batchSize);

    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);

    void setModelPath(std::string str){
        _modelPath=std::move(str);
    }

    void loadModel(char** trtModelStream,size_t& size);

    StructRst infer(const cv::Mat& img);
protected:
    Logger gLogger;
    // stuff we know about the network and the input/output blobs
    int INPUT_H,INPUT_W,OUTPUT_SIZE;
    const char* INPUT_BLOB_NAME = "data";
    const char* OUTPUT_BLOB_NAME = "prob";
    std::shared_ptr<float> _dataPtr;
    std::shared_ptr<float> _probPtr;
    std::string _modelPath;
    IExecutionContext *_context;
    cudaStream_t _stream;
};


#endif //FACERECOGNITION_TRTINFER_H
