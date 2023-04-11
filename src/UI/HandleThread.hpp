//
// Created by q1162 on 2023/4/11.
//

#ifndef FACERECOGNITION_HANDLETHREAD_HPP
#define FACERECOGNITION_HANDLETHREAD_HPP

#include <QThread>
#include <QPixmap>
#include "../TRTInfer/RetinaFace/RetinaFace_R50.h"
#include "../TRTInfer/ArcFace/ArcFace_R100.h"
#include "../TRTInfer/AntiSpoofing/AntiSpoofing.h"
#include<memory>
class HandleThread : public QThread {
    Q_OBJECT

public:
    HandleThread(QObject *parent = nullptr);

    ~HandleThread();
signals:

    void handleReady(cv::Mat image);
public slots:

    void updateFrame(const cv::Mat &img);



protected:
    void run() override;
private:
    cv::Mat _frame;
    bool newImage;
    std::vector<cv::Mat> norm;
};

#endif //FACERECOGNITION_HANDLETHREAD_HPP
