//
// Created by q1162 on 2023/3/28.
//
#include "TRTInfer/TRTInfer.h"
#include "TRTInfer/RetinaFace/RetinaFace_R50.h"
#include "TRTInfer/ArcFace/ArcFace_R100.h"
#include "config.h"
#include <direct.h>
int main(int argc, char** argv){
//    constexpr int w=640,h=480,outputSize=(h / 8 * w / 8 + h / 16 * w / 16 +h / 32 * w / 32) * 2  * 15 + 1;
//
//    std::unique_ptr<TRTInfer> retinaFacePtr{new RetinaFace(GET_PRJ_DIR()+"/models/retinaface.wts",w,h,outputSize)};
//
//    cv::VideoCapture capture(0);
//
//    while(1){
//        cv::Mat img;
//        capture>>img;
//        transpose(img, img);
//        flip(img, img, 0);
//        auto start = std::chrono::system_clock::now();
//        auto result=retinaFacePtr->infer(img);
//        auto end = std::chrono::system_clock::now();
//        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//        for(auto& i:result.detector) {
//
//            cv::Rect r = get_rect_adapt_landmark(img, w, h, i.bbox, i.landmark);
//            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
//            //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
//            for (int k = 0; k < 10; k += 2) {
//                cv::circle(img, cv::Point(i.landmark[k], i.landmark[k + 1]), 1,
//                           cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
//            }
//        }
//        cv::imshow("frame",img);
//        cv::waitKey(5);
//
//    }
    constexpr int INPUT_H = 112;
    constexpr int INPUT_W = 112;
    constexpr int OUTPUT_SIZE = 512;
    std::unique_ptr<TRTInfer> ArcFacePtr=std::make_unique<ArcFace>(GET_PRJ_DIR()+"/models/arcface-r100.wts",INPUT_W,INPUT_H,OUTPUT_SIZE);
    cv::Mat img = cv::imread(GET_PRJ_DIR()+"/models/joey0.ppm");
    auto out_norm= ArcFacePtr->infer(img,512,1).embedding;
    img=cv::imread(GET_PRJ_DIR()+"/models/joey1.ppm");
    auto out_norm1= ArcFacePtr->infer(img,1,512).embedding;

    cv::Mat res = out_norm1 * out_norm;

    std::cout << "similarity score: " << *(float*)res.data << std::endl;
    return 0;
}