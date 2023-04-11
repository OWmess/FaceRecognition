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

    std::vector<cv::Mat> process(const cv::Mat& inputMat,cv::Mat& outputMat,int rows,int cols);
signals:

    void handleReady(cv::Mat image);
public slots:

    void updateFrame(const cv::Mat &img);



protected:
    void run() override;
private:
    cv::Mat _frame;
    bool _newImage;
    std::vector<cv::Mat> norm;
    std::shared_ptr<TRTInfer> _antiSpoofingPtr = std::make_unique<AntiSpoofing>(
            GET_PRJ_DIR() + "/models/AntiSpoofing.onnx", MODELCONFIG::ANTISPOOLING::INPUT_W,
            MODELCONFIG::ANTISPOOLING::INPUT_H, MODELCONFIG::ANTISPOOLING::OUTPUTSIZE);
    std::shared_ptr<TRTInfer> _retinaFacePtr{
            new RetinaFace(GET_PRJ_DIR() + "/models/retinaface.wts", MODELCONFIG::RETINAFACE::INPUT_W,
                           MODELCONFIG::RETINAFACE::INPUT_H, MODELCONFIG::RETINAFACE::OUTPUT_SIZE)};
    std::shared_ptr<TRTInfer> _arcFacePtr = std::make_unique<ArcFace>(GET_PRJ_DIR() + "/models/arcface-r100.wts",
                                                                      MODELCONFIG::ARCFACE::INPUT_W,
                                                                      MODELCONFIG::ARCFACE::INPUT_H,
                                                                      MODELCONFIG::ARCFACE::OUTPUT_SIZE);
};

#endif //FACERECOGNITION_HANDLETHREAD_HPP
