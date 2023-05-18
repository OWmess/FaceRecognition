//
// Created by q1162 on 2023/5/19.
//

#ifndef FACERECOGNITION_ARCSOFTTHREAD_H
#define FACERECOGNITION_ARCSOFTTHREAD_H
#include <QThread>
#include <QMutex>
#include "./ArcSoft.h"
class ArcSoftThread:public QThread{
public:
    void run() override{
        while (!isInterruptionRequested()) {
            if(!_img.empty()) {
                cv::Mat src;
                {
                    QMutexLocker locker(&_mutex);
                    src = _img.clone();
                }
                bool liveness = _arcsoft.interLiveness(src);
                {
                    QMutexLocker locker(&_mutex);
                    this->_liveness = liveness;
                }
            }
            msleep(10);
        }
    }
    void setImage(const cv::Mat src){
        QMutexLocker locker(&_mutex);
        this->_img=src;
    }

    [[nodiscard]]bool getLiveness(){
        QMutexLocker locker(&_mutex);
        return _liveness;
    }
private:
    QMutex _mutex;
    bool _liveness;
    cv::Mat _img;
    ArcSoft _arcsoft;
};


#endif //FACERECOGNITION_ARCSOFTTHREAD_H
