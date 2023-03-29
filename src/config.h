//
// Created by q1162 on 2023/3/28.
//

#ifndef FACERECOGNITION_CONFIG_H
#define FACERECOGNITION_CONFIG_H

#define PRJ_NAME "FaceRecognition"
#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1  // currently, only support BATCH=1
#define CONF_THRESH 0.75
#define IOU_THRESH 0.4


#endif //FACERECOGNITION_CONFIG_H
