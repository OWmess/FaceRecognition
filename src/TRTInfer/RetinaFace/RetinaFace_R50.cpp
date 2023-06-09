
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cuda.h>
#include "cuda_runtime_api.h"
#include <NvInfer.h>
#include "calibrator.h"
#include "opencv2/opencv.hpp"
#include "RetinaFace_R50.h"
using namespace nvinfer1;
IActivationLayer* RetinaFace::bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (stride != 1 || inch != outch * 4) {
        IConvolutionLayer* conv4 = network->addConvolutionNd(input, outch * 4, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv4);
        conv4->setStrideNd(DimsHW{stride, stride});

        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

ILayer* RetinaFace::conv_bn_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int kernelsize, int stride, int padding, bool userelu, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{kernelsize, kernelsize}, getWeights(weightMap, lname + ".0.weight"), emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{padding, padding});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);

    if (!userelu) return bn1;

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    return relu1;
}

IActivationLayer* RetinaFace::ssh(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname) {
    auto conv3x3 = conv_bn_relu(network, weightMap, input, 256 / 2, 3, 1, 1, false, lname + ".conv3X3");
    auto conv5x5_1 = conv_bn_relu(network, weightMap, input, 256 / 4, 3, 1, 1, true, lname + ".conv5X5_1");
    auto conv5x5 = conv_bn_relu(network, weightMap, *conv5x5_1->getOutput(0), 256 / 4, 3, 1, 1, false, lname + ".conv5X5_2");
    auto conv7x7 = conv_bn_relu(network, weightMap, *conv5x5_1->getOutput(0), 256 / 4, 3, 1, 1, true, lname + ".conv7X7_2");
    conv7x7 = conv_bn_relu(network, weightMap, *conv7x7->getOutput(0), 256 / 4, 3, 1, 1, false, lname + ".conv7x7_3");
    ITensor* inputTensors[] = {conv3x3->getOutput(0), conv5x5->getOutput(0), conv7x7->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 3);
    IActivationLayer* relu1 = network->addActivation(*cat->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    return relu1;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* RetinaFace::createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(_modelPath);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // ------------- backbone resnet50 ---------------
    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["body.conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "body.bn1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer* x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "body.layer1.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "body.layer1.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "body.layer1.2.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2, "body.layer2.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.3.");
    IActivationLayer* layer2 = x;

    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2, "body.layer3.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.3.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.4.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.5.");
    IActivationLayer* layer3 = x;

    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2, "body.layer4.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "body.layer4.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "body.layer4.2.");
    IActivationLayer* layer4 = x;

    // ------------- FPN ---------------
    auto output1 = conv_bn_relu(network, weightMap, *layer2->getOutput(0), 256, 1, 1, 0, true, "fpn.output1");
    auto output2 = conv_bn_relu(network, weightMap, *layer3->getOutput(0), 256, 1, 1, 0, true, "fpn.output2");
    auto output3 = conv_bn_relu(network, weightMap, *layer4->getOutput(0), 256, 1, 1, 0, true, "fpn.output3");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* up3 = network->addDeconvolutionNd(*output3->getOutput(0), 256, DimsHW{2, 2}, deconvwts, emptywts);
    assert(up3);
    up3->setStrideNd(DimsHW{2, 2});
    up3->setNbGroups(256);
    weightMap["up3"] = deconvwts;

    output2 = network->addElementWise(*output2->getOutput(0), *up3->getOutput(0), ElementWiseOperation::kSUM);
    output2 = conv_bn_relu(network, weightMap, *output2->getOutput(0), 256, 3, 1, 1, true, "fpn.merge2");

    IDeconvolutionLayer* up2 = network->addDeconvolutionNd(*output2->getOutput(0), 256, DimsHW{2, 2}, deconvwts, emptywts);
    assert(up2);
    up2->setStrideNd(DimsHW{2, 2});
    up2->setNbGroups(256);
    output1 = network->addElementWise(*output1->getOutput(0), *up2->getOutput(0), ElementWiseOperation::kSUM);
    output1 = conv_bn_relu(network, weightMap, *output1->getOutput(0), 256, 3, 1, 1, true, "fpn.merge1");

    // ------------- SSH ---------------
    auto ssh1 = ssh(network, weightMap, *output1->getOutput(0), "ssh1");
    auto ssh2 = ssh(network, weightMap, *output2->getOutput(0), "ssh2");
    auto ssh3 = ssh(network, weightMap, *output3->getOutput(0), "ssh3");

    // ------------- Head ---------------
    auto bbox_head1 = network->addConvolutionNd(*ssh1->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.0.conv1x1.weight"], weightMap["BboxHead.0.conv1x1.bias"]);
    auto bbox_head2 = network->addConvolutionNd(*ssh2->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.1.conv1x1.weight"], weightMap["BboxHead.1.conv1x1.bias"]);
    auto bbox_head3 = network->addConvolutionNd(*ssh3->getOutput(0), 2 * 4, DimsHW{1, 1}, weightMap["BboxHead.2.conv1x1.weight"], weightMap["BboxHead.2.conv1x1.bias"]);

    auto cls_head1 = network->addConvolutionNd(*ssh1->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.0.conv1x1.weight"], weightMap["ClassHead.0.conv1x1.bias"]);
    auto cls_head2 = network->addConvolutionNd(*ssh2->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.1.conv1x1.weight"], weightMap["ClassHead.1.conv1x1.bias"]);
    auto cls_head3 = network->addConvolutionNd(*ssh3->getOutput(0), 2 * 2, DimsHW{1, 1}, weightMap["ClassHead.2.conv1x1.weight"], weightMap["ClassHead.2.conv1x1.bias"]);

    auto lmk_head1 = network->addConvolutionNd(*ssh1->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.0.conv1x1.weight"], weightMap["LandmarkHead.0.conv1x1.bias"]);
    auto lmk_head2 = network->addConvolutionNd(*ssh2->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.1.conv1x1.weight"], weightMap["LandmarkHead.1.conv1x1.bias"]);
    auto lmk_head3 = network->addConvolutionNd(*ssh3->getOutput(0), 2 * 10, DimsHW{1, 1}, weightMap["LandmarkHead.2.conv1x1.weight"], weightMap["LandmarkHead.2.conv1x1.bias"]);

    // ------------- Decode bbox, conf, landmark ---------------
    ITensor* inputTensors1[] = {bbox_head1->getOutput(0), cls_head1->getOutput(0), lmk_head1->getOutput(0)};
    auto cat1 = network->addConcatenation(inputTensors1, 3);
    ITensor* inputTensors2[] = {bbox_head2->getOutput(0), cls_head2->getOutput(0), lmk_head2->getOutput(0)};
    auto cat2 = network->addConcatenation(inputTensors2, 3);
    ITensor* inputTensors3[] = {bbox_head3->getOutput(0), cls_head3->getOutput(0), lmk_head3->getOutput(0)};
    auto cat3 = network->addConcatenation(inputTensors3, 3);

    auto creator = getPluginRegistry()->getPluginCreator("Decode_TRT", "1");
    PluginFieldCollection pfc;
    IPluginV2 *pluginObj = creator->createPlugin("decode", &pfc);
    ITensor* inputTensors[] = {cat1->getOutput(0), cat2->getOutput(0), cat3->getOutput(0)};
    auto decodelayer = network->addPluginV2(inputTensors, 3, *pluginObj);
    assert(decodelayer);

    decodelayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*decodelayer->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    size_t freeCuMem, totalCuMem;
    cuMemGetInfo(&freeCuMem, &totalCuMem);
    std::cout << "[INFO]: total gpu mem: " << (totalCuMem >> 20) << "MB, free gpu mem: " << (freeCuMem >> 20) << "MB"
              << std::endl;
    std::cout << "[INFO]: max workspace size will use all of free gpu mem" << std::endl;
    config->setMaxWorkspaceSize(256<<20);

//    config->setMaxWorkspaceSize(1 << 20);
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./widerface_calib/", "r50_int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network anymore
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
        mem.second.values = NULL;
    }

    return engine;
}

void RetinaFace::process() {

    char *trtModelStream{nullptr};
    size_t size{0};
    loadModel(&trtModelStream,size);
    // prepare input data ---------------------------
    static float* data=_dataPtr.get();
    static float* prob=_probPtr.get();
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    runtime->destroy();
    //ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    _context = engine->createExecutionContext();
    assert(_context != nullptr);
    //从摄像头读取图像

    cv::VideoCapture capture(0);
    cv::Mat img=cv::imread("worlds-largest-selfie.jpg");
    while(true) {
        int key=cv::waitKey(20);
        if(key==27)
            break;

        capture>>img;
        //旋转图像
        transpose(img, img);
        flip(img, img, 0);
        cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);
        //cv::imwrite("preprocessed.jpg", pr_img);

        // For multi-batch, I feed the same image multiple times.
        // If you want to process different images in a batch, you need adapt it.
        for (int b = 0; b < BATCH_SIZE; b++) {
            float *p_data = &data[b * 3 * INPUT_H * INPUT_W];
            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
                p_data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
                p_data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
            }
        }



        // Run inference
//        static float prob[BATCH_SIZE * OUTPUT_SIZE];
//        for (int cc = 0; cc < 1000; cc++) {
//            auto start = std::chrono::system_clock::now();
//            doInference(*context, data, prob, BATCH_SIZE);
//            auto end = std::chrono::system_clock::now();
//            std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us"
//                      << std::endl;
//        }
        doInference(data, prob);
        cv::Mat showimg;
        for (int b = 0; b < BATCH_SIZE; b++) {
            std::vector<decodeplugin::Detection> res;
            nms(res, &prob[b * OUTPUT_SIZE], IOU_THRESH);
            std::cout << "number of detections -> " << prob[b * OUTPUT_SIZE] << std::endl;
            std::cout << " -> " << prob[b * OUTPUT_SIZE + 10] << std::endl;
            std::cout << "after nms -> " << res.size() << std::endl;
            cv::Mat tmp = img.clone();
            for (size_t j = 0; j < res.size(); j++) {
                if (res[j].class_confidence < CONF_THRESH) continue;
                cv::Rect r = get_rect_adapt_landmark(tmp, INPUT_W, INPUT_H, res[j].bbox, res[j].landmark);
                cv::rectangle(tmp, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
                for (int k = 0; k < 10; k += 2) {
                    cv::circle(tmp, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1,
                               cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
                }
                showimg=tmp.clone();
            }
//        cv::imwrite(std::to_string(b) + "_result.jpg", tmp);
            if(!showimg.size().empty())
                cv::imshow("frame",showimg);
        }
    }

    // Destroy the engine
    _context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << i / 10 << std::endl;
    //}
    //std::cout << std::endl;

}

void RetinaFace::preProcess(const cv::Mat &img, float **predata) {
    float* &data=*predata;
    cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);
    //cv::imwrite("preprocessed.jpg", pr_img);
    // For multi-batch, I feed the same image multiple times.
    // If you want to process different images in a batch, you need adapt it.
    for (int b = 0; b < BATCH_SIZE; b++) {
        float *p_data = &data[b * 3 * INPUT_H * INPUT_W];
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
            p_data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
            p_data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
        }
    }
}

TRTInfer::StructRst RetinaFace::postProcess(float **postprob,int rows,int cols) {
    StructRst result;
    float* &prob=*postprob;
    for (int b = 0; b < BATCH_SIZE; b++) {
        std::vector<decodeplugin::Detection>& res=result.detector;
        nms(res, &prob[b * OUTPUT_SIZE], IOU_THRESH);
//        std::cout << "number of detections -> " << prob[b * OUTPUT_SIZE] << std::endl;
//        std::cout << " -> " << prob[b * OUTPUT_SIZE + 10] << std::endl;
//        std::cout << "after nms -> " << res.size() << std::endl;

        for(auto i=res.begin();i!=res.end();){

            if(i->class_confidence<CONF_THRESH)
                i=res.erase(i);
            else
                i++;

        }

    }
    return result;
}
