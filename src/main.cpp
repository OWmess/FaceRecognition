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
//    std::unique_ptr<TRTInfer> retinaFacePtr{new RetinaFace(w,h,outputSize)};
//    retinaFacePtr->setModelPath(GET_PRJ_DIR()+"/models/retinaface.wts");
//    retinaFacePtr->process();
    constexpr int INPUT_H = 112;
    constexpr int INPUT_W = 112;
    constexpr int OUTPUT_SIZE = 512;
    std::unique_ptr<TRTInfer> ArcFacePtr=std::make_unique<ArcFace>(INPUT_W,INPUT_H,OUTPUT_SIZE);
    ArcFacePtr->setModelPath(GET_PRJ_DIR()+"/models/arcface-r100.wts");
    ArcFacePtr->process();

}