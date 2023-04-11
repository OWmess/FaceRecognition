//
// Created by q1162 on 2023/3/29.
//

#ifndef FACERECOGNITION_UTILS_H
#define FACERECOGNITION_UTILS_H
#include "config.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
inline std::string GET_PRJ_DIR(){
    std::filesystem::path currentPath = std::filesystem::current_path();
    while (!currentPath.empty())
    {
        if (std::filesystem::exists(currentPath / PRJ_NAME))
        {
            return currentPath.generic_string()+"/"+PRJ_NAME;
        }
        currentPath = currentPath.parent_path();
    }
    return "GET_PRJ_DIR FAIL!";
}

inline bool makeRectSafe(cv::Rect& rect,int w,int h){

    if (rect.x < 0)                                //rect.x不能小于0（起点不能小于0）
        rect.x = 0;
    if (rect.x + rect.width > w)          //rect.x+rect.width不能大于src.size.width（x+width得到ROI的宽度终点，不能大于源图片宽度）
        rect.width = w - rect.x;
    if (rect.y < 0)                                //rect.y不能小于0（起点不能小于0
        rect.y = 0;
    if (rect.y + rect.height > h)        //rect.y+rect.height不能大于src.size.height（y+height得到ROI的高度终点，不能大于源图片高度）
        rect.height = h - rect.y;
    return !(rect.width <= 0 || rect.height <= 0);
}

inline float calFrameFps() {
    //frameFps_ = 0;
    static float frameFps=0;
    static int frameCount = 0;
    static double lastTime = (double) cv::getTickCount() / cv::getTickFrequency() * 1000; // ms

    ++frameCount;

    if (frameCount >= 20) // 取固定帧数为100帧
    {
        double curTime = (double) cv::getTickCount() / cv::getTickFrequency() * 1000.f;
        frameFps = frameCount / (curTime - lastTime) * 1000.f;
        lastTime = curTime;
        frameCount = 0;
    }
    return frameFps;
}

#endif //FACERECOGNITION_UTILS_H
