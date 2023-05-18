//
// Created by q1162 on 2023/5/19.
//

#ifndef FACERECOGNITION_ARCSOFT_H
#define FACERECOGNITION_ARCSOFT_H

#include "amcomdef.h"
#include <opencv2/opencv.hpp>

class __declspec(dllexport) ArcSoft {
public:
    ArcSoft();
    ~ArcSoft()=default;
    bool interLiveness(const cv::Mat src);
private:
    MHandle _handle = NULL;
};


#endif //FACERECOGNITION_ARCSOFT_H
