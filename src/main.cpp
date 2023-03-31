//
// Created by q1162 on 2023/3/28.
//
#include "TRTInfer/TRTInfer.h"
#include "TRTInfer/RetinaFace/RetinaFace_R50.h"
#include "TRTInfer/ArcFace/ArcFace_R100.h"
#include "config.h"
#include "utils.h"
#include <direct.h>
#include <string>
int main(int argc, char **argv) {

    std::unique_ptr<TRTInfer> retinaFacePtr{
            new RetinaFace(GET_PRJ_DIR() + "/models/retinaface.wts", MODELCONFIG::RETINAFACE::INPUT_W,
                           MODELCONFIG::RETINAFACE::INPUT_H, MODELCONFIG::RETINAFACE::OUTPUT_SIZE)};
    std::unique_ptr<TRTInfer> ArcFacePtr = std::make_unique<ArcFace>(GET_PRJ_DIR() + "/models/arcface-r100.wts",
                                                                     MODELCONFIG::ARCFACE::INPUT_W,
                                                                     MODELCONFIG::ARCFACE::INPUT_H,
                                                                     MODELCONFIG::ARCFACE::OUTPUT_SIZE);
    cv::VideoCapture capture(0);
    cv::Mat img;
    cv::Mat detectImg;
    cv::Mat showImg;
    std::vector<cv::Mat> norm;
    while(1) {
        int key=cv::waitKey(20);
        capture >> img;
        transpose(img, img);
        flip(img, img, 0);
        showImg=img.clone();
        auto detectResult = retinaFacePtr->infer(img);

        for (auto &i: detectResult.detector) {

            cv::Rect r = get_rect_adapt_landmark(showImg, MODELCONFIG::RETINAFACE::INPUT_W,
                                                 MODELCONFIG::RETINAFACE::INPUT_H,
                                                 i.bbox, i.landmark);
            makeRectSafe(r,MODELCONFIG::RETINAFACE::INPUT_W,MODELCONFIG::RETINAFACE::INPUT_H);
            cv::rectangle(showImg, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
            for (int k = 0; k < 10; k += 2) {
                cv::circle(showImg, cv::Point(i.landmark[k], i.landmark[k + 1]), 1,
                           cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
            }
            detectImg = img(r);
        }
        cv::imshow("detect_roi", detectImg);
        cv::imshow("frame", showImg);
        cv::resize(detectImg,detectImg,{MODELCONFIG::ARCFACE::INPUT_W,MODELCONFIG::ARCFACE::INPUT_H});
        TRTInfer::StructRst resultNorm;

        if(key=='s'){
            static int num=0;
            cv::imwrite(GET_PRJ_DIR()+"/pic"+std::to_string(num++)+".jpg",detectImg);
            Sleep(1000);
        }

        if(key=='p'){
            ArcFacePtr->process();
        }

        if(key=='t'){
            resultNorm=ArcFacePtr->infer(detectImg,SAVE_FORMAT);
            norm.push_back(resultNorm.embedding);
            Sleep(500);
            std::cout<<"saving face embedding..."<<std::endl;
        }else{
            resultNorm=ArcFacePtr->infer(detectImg,INFER_FORMAT);
            if(!norm.empty()){

                int cnt=0;
                for(const auto& i:norm){

                    cv::Mat score=resultNorm.embedding*i;

                    std::cout<<cnt<<"  similarity score: "<<*(float*)score.data<<std::endl;
                    cnt++;
                }
            }
        }


        cv::imshow("Norm_roi", detectImg);
    }


//    std::unique_ptr<TRTInfer> ArcFacePtr = std::make_unique<ArcFace>(GET_PRJ_DIR() + "/models/arcface-r100.wts",
//                                                                     MODELCONFIG::ARCFACE::INPUT_W,
//                                                                     MODELCONFIG::ARCFACE::INPUT_H,
//                                                                     MODELCONFIG::ARCFACE::OUTPUT_SIZE);
//    cv::Mat img;
//    img = cv::imread(GET_PRJ_DIR() + "/models/joey0.ppm");
//
//    auto out_norm = ArcFacePtr->infer(img, 512, 1).embedding;
//    img = cv::imread(GET_PRJ_DIR() + "/models/joey1.ppm");
//    auto out_norm1 = ArcFacePtr->infer(img, 1, 512).embedding;
//
//    cv::Mat res = out_norm1 * out_norm;
//
//    std::cout << "similarity score: " << *(float *) res.data << std::endl;
    return 0;
}