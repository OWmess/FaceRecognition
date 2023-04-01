//
// Created by q1162 on 2023/3/28.
//
#ifndef FACERECOGNITION_TRTINFER_HPP
#define FACERECOGNITION_TRTINFER_HPP

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

class TRTInfer {
public:
    struct StructRst {
        std::vector<decodeplugin::Detection> detector;
        cv::Mat embedding;
    };

    TRTInfer() = delete;

    TRTInfer(std::string modelPath, int w, int h, int o);

    virtual ~TRTInfer();

    virtual void process() = 0;

    void prepareModel();
    
    virtual ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt)=0;

    virtual void preProcess(const cv::Mat &img, float **predata) = 0;

    virtual StructRst postProcess(float **prob,int rows=0,int cols=0) = 0;

    void doInference(float *input, float *output);

    void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream);

    void loadModel(char **trtModelStream, size_t &size);

    StructRst infer(const cv::Mat &img,int rows=0,int cols=0);

protected:
    Logger gLogger;
    // stuff we know about the network and the input/output blobs
    int INPUT_H, INPUT_W, OUTPUT_SIZE;
    const char *INPUT_BLOB_NAME = "data";
    const char *OUTPUT_BLOB_NAME = "prob";
    std::shared_ptr<float> _dataPtr;
    std::shared_ptr<float> _probPtr;
    std::string _modelPath;
    IExecutionContext *_context;
    cudaStream_t _stream;
    int _inputIndex, _outputIndex;
    const int _batchSize = BATCH_SIZE;
    void *_buffers[2];
};


#endif //FACERECOGNITION_TRTINFER_HPP
