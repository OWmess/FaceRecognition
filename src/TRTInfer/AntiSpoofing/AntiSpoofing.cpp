//
// Created by q1162 on 2023/3/31.
//

#include "AntiSpoofing.h"
#include <NvOnnxParser.h>
#include <cuda.h>

using namespace nvinfer1;
inline void softmax(std::vector<float>& t){
    float total=0;
    for(int i=0;i<3;i++)
        total+=exp(t[i]);
    for(int i=0;i<3;i++)
        t[i]=exp(t[i])/total;
}

AntiSpoofing::AntiSpoofing(std::string modelPath, int w, int h, int o) : TRTInfer(modelPath, w, h, o) {
    INPUT_BLOB_NAME="input";
    OUTPUT_BLOB_NAME="output";
    prepareModel();

}
ICudaEngine* AntiSpoofing::createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt){
    auto explicitBatch=1U<<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network=builder->createNetworkV2(explicitBatch);
    assert(network!= nullptr);
    auto parser=nvonnxparser::createParser(*network,gLogger);
    assert(parser!= nullptr);
    auto parsingSuccess=parser->parseFromFile(_modelPath.c_str(),static_cast<int>(gLogger.getReportableSeverity()));
    assert(parsingSuccess==true);

#ifdef USE_FP16
//    config->setFlag(BuilderFlag::kFP16);
#endif
    auto profile=builder->createOptimizationProfile();
    profile->setDimensions(INPUT_BLOB_NAME,OptProfileSelector::kMIN,Dims4{1,3,80,80});
    profile->setDimensions(INPUT_BLOB_NAME,OptProfileSelector::kOPT,Dims4{1,3,80,80});
    profile->setDimensions(INPUT_BLOB_NAME,OptProfileSelector::kMAX,Dims4{1,3,80,80});
    config->setCalibrationProfile(profile);

    builder->setMaxBatchSize(_batchSize);
    size_t free, total;
    cuMemGetInfo(&free, &total);
    std::cout << "[INFO]: total gpu mem: " << (total >> 20) << "MB, free gpu mem: " << (free >> 20) << "MB"
              << std::endl;
    std::cout << "[INFO]: max workspace size will use all of free gpu mem" << std::endl;
    config->setMaxWorkspaceSize(free);
    std::cout << "Building engine, please wait for a while..." << std::endl;
    auto engine=builder->buildEngineWithConfig(*network,*config);
    std::cout << "Build engine successfully!" << std::endl;


    network->destroy();

    return engine;
}

TRTInfer::StructRst AntiSpoofing::infer(const cv::Mat &img,decodeplugin::Detection bbox,float thresh) {
    static float* prob=_probPtr.get();
    static float* data=_dataPtr.get();
    cv::Mat x= getRoi(img,bbox.bbox,ANTI_SPOOFING_SCALE);
//    x=cv::imread(GET_PRJ_DIR()+"/models/real.jpg");
    cv::cvtColor(x,x,cv::COLOR_BGR2RGB);
    cv::Mat b(80,80,CV_32FC3,data);
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        b.at<cv::Vec3f>(i)[0] = ((float)x.at<cv::Vec3b>(i)[2]) ;
        b.at<cv::Vec3f>(i)[1] = ((float)x.at<cv::Vec3b>(i)[0]) ;
        b.at<cv::Vec3f>(i)[2] = ((float)x.at<cv::Vec3b>(i)[1]);
    }

    doInference(data,prob);
    std::vector<float>vec{prob[0],prob[1],prob[2]};

    softmax(vec);
    int cnt=0;
    float max=0;
    int maxIndex=0;
    for(const auto& i:vec){
        if(i>max) {
            max=i;
            maxIndex = cnt;
        }
        cnt++;
    }
//    std::cout<<"maxIndex:   "<<maxIndex<<std::endl;
    StructRst rst;
    //TODO  取反了，待修改！！
    rst.antiSpoof={vec[1],!(maxIndex!=1 || vec[1]<thresh ? true :false)};
    return rst;
}

void AntiSpoofing::doInference(float* input, float* output) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(_buffers[_inputIndex], input, _batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, _stream));
    _context->enqueueV2( _buffers, _stream, nullptr);
    CHECK(cudaMemcpyAsync(output, _buffers[_outputIndex], _batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, _stream));
    cudaStreamSynchronize(_stream);

}

cv::Mat AntiSpoofing::getRoi(const cv::Mat &src, float bbox[], float scale) {
    const float x=bbox[0];
    const float y=bbox[1];
    const float box_w=abs(bbox[0]-bbox[2]);
    const float box_h=abs(bbox[1]-bbox[3]);
    scale= std::min({(src.size().height-1.f)/box_h,(src.size().width-1.f)/box_w,scale});

    const float new_width=box_w*scale;
    const float new_height=box_h*scale;

    cv::Point2f center={box_w/2+x,box_h/2+y};

    cv::Point2f lt={center.x-new_width/2,center.y-new_height/2};
    cv::Point2f rb={center.x+new_width/2,center.y+new_height/2};
    if(lt.x<0){
        rb.x-=lt.x;
        lt.x=0;
    }
    if(lt.y<0){
        rb.y-=lt.y;
        lt.y=0;
    }
    if(rb.x>src.size().width-1){
        lt.x-=rb.x-src.size().width+1;
        rb.x=src.size().width-1;
    }
    if(rb.y>src.size().height-1){
        lt.y-=rb.y-src.size().height+1;
        rb.y=src.size().height-1;
    }
    cv::Rect rect={lt,rb};
    makeRectSafe(rect,src.size().width,src.size().height);
    cv::Mat roi=src(rect);
    cv::resize(roi,roi,{INPUT_W,INPUT_H});
    return roi;
}

TRTInfer::StructRst AntiSpoofing::infer(const cv::Mat &img, int rows, int cols) {
    static float* prob=_probPtr.get();
    static float* data=_dataPtr.get();
    cv::Mat x=img.clone();
    if(x.cols!=80||x.rows!=80){
        cv::resize(x,x,{80,80});
    }
    cv::cvtColor(x,x,cv::COLOR_BGR2RGB);
    cv::Mat b(80,80,CV_32FC3);
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        b.at<cv::Vec3f>(i)[0] = ((float)x.at<cv::Vec3b>(i)[2]) ;
        b.at<cv::Vec3f>(i)[1] = ((float)x.at<cv::Vec3b>(i)[0]) ;
        b.at<cv::Vec3f>(i)[2] = ((float)x.at<cv::Vec3b>(i)[1]);
    }
    cv::Mat y(80,80,CV_32FC3,data);
    b.convertTo(y,CV_32FC3);
    //-2.83916 -2.18445 5.02528
    //-3.68966 -0.699292 4.3899
    CHECK(cudaMemcpyAsync(_buffers[_inputIndex], y.data, _batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, _stream));
    _context->enqueueV2( _buffers, _stream, nullptr);
    CHECK(cudaMemcpyAsync(prob, _buffers[_outputIndex], _batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, _stream));
    cudaStreamSynchronize(_stream);
    std::vector<float>vec{prob[0],prob[1],prob[2]};
    //tensor([[ 0.5834, -3.5567,  2.9750]])
    std::cout<<"prob"<<std::endl;
    for(int i=0;i<3;i++){
        std::cout<<prob[i]<<" ";
    }
    std::cout<<std::endl;
    softmax(vec);
    //[[0.08370467 0.00133263 0.9149627 ]]
    int cnt=0;
    float max=0;
    int maxIndex=0;
    for(const auto& i:vec){
        if(i>max) {
            max=i;
            maxIndex = cnt;
        }
        std::cout<<cnt++<<" "<<i<<"  ";
    }
    return TRTInfer::StructRst{};
}


void AntiSpoofing::process() {


}

void AntiSpoofing::preProcess(const cv::Mat &img, float **predata) {

}

TRTInfer::StructRst AntiSpoofing::postProcess(float **prob, int rows, int cols) {
    return TRTInfer::StructRst();
}