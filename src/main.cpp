//
// Created by q1162 on 2023/3/28.
//

#include "UI/mainwindow.h"
#include <windows.h>
#include <QApplication>


int main(int argc, char *argv[])
{
#if defined(Q_OS_WIN)
    // 隐藏控制台窗口
    HWND hWnd = GetConsoleWindow();
    if (hWnd != NULL)
    {
        ShowWindow(hWnd, SW_HIDE);
    }
#endif
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}












//#include "TRTInfer/RetinaFace/RetinaFace_R50.h"
//#include "TRTInfer/ArcFace/ArcFace_R100.h"
//#include "TRTInfer/AntiSpoofing/AntiSpoofing.h"
//#include "config.h"
//#include "utils.h"
//#include <string>
//#include <thread>
//#include "handleThread.hpp"
//int main(int argc, char **argv) {
//    std::unique_ptr<TRTInfer> AntiSpoofingPtr=std::make_unique<AntiSpoofing>(GET_PRJ_DIR() + "/models/AntiSpoofing.onnx",MODELCONFIG::ANTISPOOLING::INPUT_W,MODELCONFIG::ANTISPOOLING::INPUT_H,MODELCONFIG::ANTISPOOLING::OUTPUTSIZE);
//    std::unique_ptr<TRTInfer> retinaFacePtr{
//            new RetinaFace(GET_PRJ_DIR() + "/models/retinaface.wts", MODELCONFIG::RETINAFACE::INPUT_W,
//                           MODELCONFIG::RETINAFACE::INPUT_H, MODELCONFIG::RETINAFACE::OUTPUT_SIZE)};
//    std::unique_ptr<TRTInfer> ArcFacePtr = std::make_unique<ArcFace>(GET_PRJ_DIR() + "/models/arcface-r100.wts",
//                                                                     MODELCONFIG::ARCFACE::INPUT_W,
//                                                                     MODELCONFIG::ARCFACE::INPUT_H,
//                                                                     MODELCONFIG::ARCFACE::OUTPUT_SIZE);
//
//    cv::VideoCapture capture(0);
//    cv::Mat img=cv::imread(GET_PRJ_DIR() + "/models/test.jpg");
//    cv::Mat showImg;
//    std::vector<cv::Mat> norm;
//    bool save;
//    while(1){
//        int key=cv::waitKey(1);
//        save= (key=='s')? true:false;
//        capture >> img;
//        //旋转图像
////        cv::transpose(img, img);
////        cv::flip(img, img, 0);
//
//        showImg=img.clone();
//        auto detectResult = retinaFacePtr->infer(img);
//
//        for (auto &i: detectResult.detector) {
//
//            cv::Rect r = get_rect_adapt_landmark(showImg, MODELCONFIG::RETINAFACE::INPUT_W,
//                                                 MODELCONFIG::RETINAFACE::INPUT_H,
//                                                 i.bbox, i.landmark);
//            makeRectSafe(r,MODELCONFIG::RETINAFACE::INPUT_W,MODELCONFIG::RETINAFACE::INPUT_H);
//            cv::rectangle(showImg, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
//            //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
//            for (int k = 0; k < 10; k += 2) {
//                cv::circle(showImg, cv::Point(i.landmark[k], i.landmark[k + 1]), 1,
//                           cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
//            }
//            auto antiRst=AntiSpoofingPtr->infer(img,i,ANTI_SPOOFING_THRESH).antiSpoof;
//            std::string antiStr=std::string(antiRst.isFake?"fake ":"real ")+"  score:"+std::to_string(antiRst.antiSpoofConf);
//            cv::putText(showImg,antiStr,r.tl(),1,1,cv::Scalar(0x27, 0xC1, 0x36));
//            std::cout<<antiRst.isFake<<"  conf: "<<antiRst.antiSpoofConf<<std::endl;
//            cv::Mat resizeImg;
//            cv::resize(img(r),resizeImg,{MODELCONFIG::ARCFACE::INPUT_W,MODELCONFIG::ARCFACE::INPUT_H});
//            TRTInfer::StructRst resultNorm;
//            std::string embeddingStr;
//            if(save){
//                resultNorm=ArcFacePtr->infer(resizeImg,SAVE_FORMAT);
//                norm.push_back(resultNorm.embedding);
//                Sleep(500);
//                std::cout<<"saving face embedding..."<<std::endl;
//            }else{
//                resultNorm=ArcFacePtr->infer(resizeImg,INFER_FORMAT);
//                if(!norm.empty()){
//                    int cnt=0;
//                    int maxIndex=0;
//                    float max=0;
//                    for(const auto& x:norm){
//
//                        cv::Mat score=resultNorm.embedding*x;
//                        float t=*(float*)score.data;
//                        if(t>max){
//                            max=t;
//                            maxIndex=cnt;
//                        }
//                        std::cout<<cnt<<"  similarity score: "<<*(float*)score.data<<std::endl;
//                        cnt++;
//                    }
//                    embeddingStr=std::to_string(maxIndex)+ "  score:"+std::to_string(max);
//                    cv::putText(showImg,embeddingStr,r.br(),1,1,cv::Scalar(0x27, 0xC1, 0x36));
//                }
//            }
//
//        }
//
//
//
//        cv::imshow("showimg",showImg);
//
//        std::cout<<"FPS: "<<calFrameFps()<<std::endl;
//    }
//
//}