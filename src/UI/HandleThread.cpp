//
// Created by q1162 on 2023/4/11.
//
#pragma execution_character_set("utf-8")
#include "HandleThread.hpp"
#include "../config.h"
#include "../FileManager.hpp"
#include <codecvt>
#include <qdebug.h>

HandleThread::HandleThread(QObject *parent) : QThread(parent), _newImage(false),_saveMode(false) {
    _arcSoftThread.start();
}

HandleThread::~HandleThread() {
    _arcSoftThread.requestInterruption();
    _arcSoftThread.wait();

}

void HandleThread::updateFrame(const cv::Mat &img) {
    this->_frame = img.clone();
    _newImage = true;
}

void HandleThread::run() {

    cv::Mat showImg;
    auto &fm=FileManager::getInstance();
    while (!isInterruptionRequested()) {
        if (_newImage) {
            std::vector<NameFormat> nameVec;
            showImg = _frame.clone();
            if(_saveMode){
                auto norm=appendProcess(_frame,showImg,SAVE_FORMAT);
                if(norm.cols==0&&norm.rows==0){
                    emit detectorEmpty();
                    _saveMode=false;
                    continue;
                }
                fm.appendFace({_appendId.toStdString(),std::string(_appendName.toStdString())},norm);
                msleep(10);
                auto savedir=fm.getSaveDir()+_appendId.toStdString()+".jpg";
                _saveMode=false;
            }else{
                auto normvec=process(_frame,showImg,INFER_FORMAT);
                for(const auto& norm:normvec) {
                    QMutexLocker locker(&fm._mutex);
//                    if (fm.writeReady())
                        std::for_each(fm.getFaceData().begin(), fm.getFaceData().end(), [&](const auto &pair) {
                            cv::Mat res = norm.embebding * pair.second;
                            float score = *(float *) res.data;
                            if (score > CONTRAST_THRESH) {
                                QString name = QString::fromStdString(pair.first.name);
                                nameVec.push_back({name, norm.pt});
                                std::cout << pair.first.name << " conf: " << score << std::endl;
                            }
                            return;
                        });
                }
            }

            emit handleReady(showImg,nameVec);
            _newImage = false;
        }
        msleep(10);
    }
}

std::vector<EmbeddingFormat> HandleThread::process(const cv::Mat& inputMat,cv::Mat& outputMat,int rows, int cols) {
    _arcSoftThread.setImage(inputMat);
    const cv::Mat& img=inputMat;
    auto detectResult = _retinaFacePtr->infer(img);
    std::vector<EmbeddingFormat> resultNorm;
    for (auto &i: detectResult.detector) {

        cv::Rect r = get_rect_adapt_landmark(outputMat, MODELCONFIG::RETINAFACE::INPUT_W,
                                             MODELCONFIG::RETINAFACE::INPUT_H,
                                             i.bbox, i.landmark);
        makeRectSafe(r, MODELCONFIG::RETINAFACE::INPUT_W, MODELCONFIG::RETINAFACE::INPUT_H);
        if(r.height<100||r.width<100)
            continue;
        cv::rectangle(outputMat, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
        for (int k = 0; k < 10; k += 2) {
            cv::circle(outputMat, cv::Point(i.landmark[k], i.landmark[k + 1]), 1,
                       cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
        }
//        bool liveness=_arcSoftThread.getLiveness();
//        std::string antiStr = std::string(liveness ? "real " : "fake ");
        auto antiRst = _antiSpoofingPtr->infer(img, i, ANTI_SPOOFING_THRESH).antiSpoof;
        std::string antiStr = std::string(antiRst.isFake ? "fake " : "real ");


//        std::cout <<"Anti info: "<< antiRst.isFake << "  conf: " << antiRst.antiSpoofConf << std::endl;
#if ENABLE_ANTI
        if(antiRst.isFake) {
                    continue;
                }
#endif
        cv::Mat resizeImg;
        cv::resize(img(r), resizeImg, {MODELCONFIG::ARCFACE::INPUT_W, MODELCONFIG::ARCFACE::INPUT_H});

        auto result=_arcFacePtr->infer(resizeImg, rows,cols);
//        std::string nameStr;
//        auto &fm=FileManager::getInstance();
//        std::for_each(fm.getFaceData().begin(), fm.getFaceData().end(), [&](const auto &pair) {
//            cv::Mat res=result.embedding*pair.second;
//            float score=*(float*)res.data;
//            if(score>CONTRAST_THRESH){
//
//                nameStr="  "+pair.first.name;
//
//                            std::cout<<nameStr<<" conf: "<<score<<std::endl;
//            }
//            return;
//        });
//        std::string str=antiStr+nameStr;
//
//        std::cout<<str<<std::endl;
////        std::cout<<std::string("utf8: ")<<utf8String<<std::endl;
//
        cv::putText(outputMat, antiStr, r.tl(), 1, 1, cv::Scalar(0x27, 0xC1, 0x36));

        resultNorm.push_back({result.embedding,r.tl()});

    }

    return resultNorm;
}

void HandleThread::updateData(bool mode, const QString& id, const QString& name) {
    if(mode==true){//append face
        _saveMode=true;
        _appendId=id;
        _appendName=name;
    }else{//delete face ,already delete at UI thread

    }
}

cv::Mat HandleThread::appendProcess(const cv::Mat& inputMat,cv::Mat& outputMat,int rows, int cols) {
    _arcSoftThread.setImage(inputMat);
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
    auto antiRst = _antiSpoofingPtr->infer(img, rst1, ANTI_SPOOFING_THRESH).antiSpoof;
    std::string antiStr = std::string(antiRst.isFake ? "fake " : "real ");
//    bool liveness=_arcSoftThread.getLiveness();
//    std::string antiStr = std::string(liveness ? "real " : "fake ");
    cv::putText(outputMat, antiStr, r.tl(), 1, 1, cv::Scalar(0x27, 0xC1, 0x36));
    cv::Mat resizeImg;
    cv::resize(img(r), resizeImg, {MODELCONFIG::ARCFACE::INPUT_W, MODELCONFIG::ARCFACE::INPUT_H});

    auto result = _arcFacePtr->infer(resizeImg, rows, cols);
    return result.embedding;
}