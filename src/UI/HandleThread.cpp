//
// Created by q1162 on 2023/4/11.
//
#include "HandleThread.hpp"
#include "../config.h"
#include "../FileManager.hpp"
HandleThread::HandleThread(QObject *parent) : QThread(parent), _newImage(false),_saveMode(false) {

}

HandleThread::~HandleThread() {

}

void HandleThread::updateFrame(const cv::Mat &img) {
    this->_frame = img.clone();
    _newImage = true;
}

void HandleThread::run() {

    cv::Mat showImg;
    std::vector<cv::Mat> norm;
    while (!isInterruptionRequested()) {
        if (_newImage) {
            showImg = _frame.clone();
            if(_saveMode){
                auto norm=appendProcess(_frame,showImg,SAVE_FORMAT);
                if(norm.cols==0&&norm.rows==0){
                    emit detectorEmpty();
                    _saveMode=false;
                    continue;
                }
                auto &fm=FileManager::getInstance();
                fm.getFaceData().insert(std::make_pair(_appendId.toStdString(),norm));
                _saveMode=false;

            }else{

                process(_frame,showImg,INFER_FORMAT);
            }

            emit handleReady(showImg);
            _newImage = false;
        }
    }
}

std::vector<cv::Mat> HandleThread::process(const cv::Mat& inputMat,cv::Mat& outputMat,int rows, int cols) {
    const cv::Mat& img=inputMat;
    auto detectResult = _retinaFacePtr->infer(img);
    std::vector<cv::Mat> resultNorm;
    for (auto &i: detectResult.detector) {

        cv::Rect r = get_rect_adapt_landmark(outputMat, MODELCONFIG::RETINAFACE::INPUT_W,
                                             MODELCONFIG::RETINAFACE::INPUT_H,
                                             i.bbox, i.landmark);
        makeRectSafe(r, MODELCONFIG::RETINAFACE::INPUT_W, MODELCONFIG::RETINAFACE::INPUT_H);
        cv::rectangle(outputMat, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
        for (int k = 0; k < 10; k += 2) {
            cv::circle(outputMat, cv::Point(i.landmark[k], i.landmark[k + 1]), 1,
                       cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
        }
        auto antiRst = _antiSpoofingPtr->infer(img, i, ANTI_SPOOFING_THRESH).antiSpoof;
        std::string antiStr = std::string(antiRst.isFake ? "fake " : "real ") + "  score:" +
                              std::to_string(antiRst.antiSpoofConf);
        cv::putText(outputMat, antiStr, r.tl(), 1, 1, cv::Scalar(0x27, 0xC1, 0x36));
        std::cout << antiRst.isFake << "  conf: " << antiRst.antiSpoofConf << std::endl;
#if ENABLE_ANTI
        if(antiRst.isFake) {
                    continue;
                }
#endif
        cv::Mat resizeImg;
        cv::resize(img(r), resizeImg, {MODELCONFIG::ARCFACE::INPUT_W, MODELCONFIG::ARCFACE::INPUT_H});

        auto result=_arcFacePtr->infer(resizeImg, rows,cols);
        resultNorm.emplace_back(std::move(result.embedding));

    }

    return resultNorm;
}

void HandleThread::updateData(bool mode, const QString& id, const QString& name) {
    if(mode==true){//append face
        _saveMode=true;
        _appendId=id;
    }else{//delete face ,already delete at UI thread

    }
}

cv::Mat HandleThread::appendProcess(const cv::Mat& inputMat,cv::Mat& outputMat,int rows, int cols) {
    const cv::Mat& img=inputMat;
    auto detectResult = _retinaFacePtr->infer(img);
    if(detectResult.detector.empty()){
        return cv::Mat(0,0,CV_16UC1);
    }
    int maxArea=0;
    std::vector<decodeplugin::Detection>::iterator iter;
    for (auto i= detectResult.detector.begin();i!=detectResult.detector.end();i++) {
        int area=abs(i->bbox[0]-i->bbox[2])*(i->bbox[1]*i->bbox[3]);
        if(area>maxArea){
            maxArea=area;
            iter=i;
        }
    }
    decodeplugin::Detection &rst1=*iter;
    cv::Rect r = get_rect_adapt_landmark(outputMat, MODELCONFIG::RETINAFACE::INPUT_W,
                                         MODELCONFIG::RETINAFACE::INPUT_H,
                                         rst1.bbox, rst1.landmark);
    makeRectSafe(r, MODELCONFIG::RETINAFACE::INPUT_W, MODELCONFIG::RETINAFACE::INPUT_H);
    cv::rectangle(outputMat, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
    //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
    for (int k = 0; k < 10; k += 2) {
        cv::circle(outputMat, cv::Point(rst1.landmark[k], rst1.landmark[k + 1]), 1,
                   cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
    }
    cv::Mat resizeImg;
    cv::resize(img(r), resizeImg, {MODELCONFIG::ARCFACE::INPUT_W, MODELCONFIG::ARCFACE::INPUT_H});

    auto result = _arcFacePtr->infer(resizeImg, rows, cols);
    return result.embedding;
}