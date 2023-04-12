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

    void handleReady(const cv::Mat &image,QString);

    void detectorEmpty();

public slots:

    void updateFrame(const cv::Mat &img);

    void updateData(bool mode,const QString& id,const QString& name);

protected:
    void run() override;
private:
    std::vector<cv::Mat> process(const cv::Mat& inputMat,cv::Mat& outputMat,int rows,int cols);
    cv::Mat appendProcess(const cv::Mat& inputMat,cv::Mat& outputMat,int rows,int cols);
    cv::Mat _frame;
    bool _newImage;
    bool _saveMode;
    QString _appendId;
    QString _appendName;
    std::vector<cv::Mat> norm;
    std::unique_ptr<TRTInfer> _antiSpoofingPtr = std::make_unique<AntiSpoofing>(
            GET_PRJ_DIR() + "/models/AntiSpoofing.onnx", MODELCONFIG::ANTISPOOLING::INPUT_W,
            MODELCONFIG::ANTISPOOLING::INPUT_H, MODELCONFIG::ANTISPOOLING::OUTPUTSIZE);
    std::unique_ptr<TRTInfer> _retinaFacePtr{
            new RetinaFace(GET_PRJ_DIR() + "/models/retinaface.wts", MODELCONFIG::RETINAFACE::INPUT_W,
                           MODELCONFIG::RETINAFACE::INPUT_H, MODELCONFIG::RETINAFACE::OUTPUT_SIZE)};
    std::unique_ptr<TRTInfer> _arcFacePtr = std::make_unique<ArcFace>(GET_PRJ_DIR() + "/models/arcface-r100.wts",
                                                                      MODELCONFIG::ARCFACE::INPUT_W,
                                                                      MODELCONFIG::ARCFACE::INPUT_H,
                                                                      MODELCONFIG::ARCFACE::OUTPUT_SIZE);
};

#endif //FACERECOGNITION_HANDLETHREAD_HPP
