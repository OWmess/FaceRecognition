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

    [[nodiscard]] std::shared_ptr<TRTInfer> getAntiSpoofingPtr() const{
        return _antiSpoofingPtr;
    }

    [[nodiscard]] std::shared_ptr<TRTInfer> getRetinaFace() const{
        return _retinaFacePtr;
    }

    [[nodiscard]] std::shared_ptr<TRTInfer> getArcFace() const{
        return _arcFacePtr;
    }


signals:

    void handleReady(cv::Mat image);

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
    std::vector<cv::Mat> norm;
    std::shared_ptr<TRTInfer> _antiSpoofingPtr = std::make_shared<AntiSpoofing>(
            GET_PRJ_DIR() + "/models/AntiSpoofing.onnx", MODELCONFIG::ANTISPOOLING::INPUT_W,
            MODELCONFIG::ANTISPOOLING::INPUT_H, MODELCONFIG::ANTISPOOLING::OUTPUTSIZE);
    std::shared_ptr<TRTInfer> _retinaFacePtr{
            new RetinaFace(GET_PRJ_DIR() + "/models/retinaface.wts", MODELCONFIG::RETINAFACE::INPUT_W,
                           MODELCONFIG::RETINAFACE::INPUT_H, MODELCONFIG::RETINAFACE::OUTPUT_SIZE)};
    std::shared_ptr<TRTInfer> _arcFacePtr = std::make_shared<ArcFace>(GET_PRJ_DIR() + "/models/arcface-r100.wts",
                                                                      MODELCONFIG::ARCFACE::INPUT_W,
                                                                      MODELCONFIG::ARCFACE::INPUT_H,
                                                                      MODELCONFIG::ARCFACE::OUTPUT_SIZE);
};

#endif //FACERECOGNITION_HANDLETHREAD_HPP
