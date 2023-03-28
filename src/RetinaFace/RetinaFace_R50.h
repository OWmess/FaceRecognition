//
// Created by q1162 on 2023/3/28.
//

#ifndef FACERECOGNITION_RETINAFACE_R50_H
#define FACERECOGNITION_RETINAFACE_R50_H

#include "../config.h"
#include "logging.h"
#include "common.hpp"
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "decode.h"
class RetinaFace{
public:
    RetinaFace()=default;
    ~RetinaFace()=default;

    void process();
private:
    IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname);

    ILayer* conv_bn_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int kernelsize, int stride, int padding, bool userelu, std::string lname);

    IActivationLayer* ssh(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname);

    ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);

    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);

    void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
private:
    // stuff we know about the network and the input/output blobs
    static constexpr int INPUT_H = decodeplugin::INPUT_H;  // H, W must be able to  be divided by 32.
    static constexpr int INPUT_W = decodeplugin::INPUT_W;
    static constexpr int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2  * 15 + 1;
    const char* INPUT_BLOB_NAME = "data";
    const char* OUTPUT_BLOB_NAME = "prob";

    Logger gLogger;
};
#endif //FACERECOGNITION_RETINAFACE_R50_H
