//
// Created by q1162 on 2023/5/19.
//

#include "ArcSoft.h"
#include "../../config.h"
#include "amcomdef.h"
#include "arcsoft_face_sdk.h"
#include "asvloffscreen.h"
#include "merror.h"
ArcSoft::ArcSoft() {
    MRESULT res = MOK;
    ASF_ActiveFileInfo activeFileInfo = { 0 };
    res = ASFGetActiveFileInfo(&activeFileInfo);
    if (res != MOK)
    {
        printf("ASFGetActiveFileInfo fail: %d\n", res);
    }

    //激活接口,首次激活需联网
    res = ASFOnlineActivation(const_cast<char*>(APPID), const_cast<char*>(SDKKEY));
    if (MOK != res && MERR_ASF_ALREADY_ACTIVATED != res)
        printf("ASFActivation fail: %d\n", res);
    else
        printf("ASFActivation sucess: %d\n", res);

    //初始化接口
    MInt32 mask = ASF_FACE_DETECT | ASF_FACERECOGNITION |ASF_LIVENESS;
    res = ASFInitEngine(ASF_DETECT_MODE_IMAGE, ASF_OP_0_ONLY, 30, 5, mask, &_handle);
    if (res != MOK)
        printf("ASFInitEngine fail: %d\n", res);
    else
        printf("ASFInitEngine sucess: %d\n", res);

    ASF_LivenessThreshold threshold = { 0 };
    threshold.thresholdmodel_BGR = ANTI_SPOOFING_THRESH;
    threshold.thresholdmodel_IR = 0.7;
    res = ASFSetLivenessParam(_handle, &threshold);
    if (res != MOK)
        printf("ASFSetLivenessParam fail: %d\n", res);
    else
        printf("RGB Threshold: %f  IR Threshold: %f\n", threshold.thresholdmodel_BGR, threshold.thresholdmodel_IR);
}

bool ArcSoft::interLiveness(const cv::Mat src) {
    MRESULT res = MOK;
    cv::Mat cutImg=src({0, 0, src.cols - src.cols % 4, src.rows});

    ASVLOFFSCREEN offscreen = {0};
    offscreen.u32PixelArrayFormat = ASVL_PAF_RGB24_B8G8R8;
    offscreen.i32Width = cutImg.cols;
    offscreen.i32Height = cutImg.rows;
    offscreen.pi32Pitch[0] = cutImg.step;
    offscreen.ppu8Plane[0] = cutImg.data;
    ASF_SingleFaceInfo SingleDetectedFaces = {0 };
    ASF_MultiFaceInfo detectedFaces = { 0 };
    ASF_FaceFeature feature = { 0 };
    res = ASFDetectFacesEx(_handle, &offscreen, &detectedFaces);
//    std::cout<<"detectedFaces.faceNum: "<<detectedFaces.faceNum<<std::endl;
    if(res == MOK&& detectedFaces.faceNum > 1){
        SingleDetectedFaces.faceRect.left = detectedFaces.faceRect[0].left;
        SingleDetectedFaces.faceRect.top = detectedFaces.faceRect[0].top;
        SingleDetectedFaces.faceRect.right = detectedFaces.faceRect[0].right;
        SingleDetectedFaces.faceRect.bottom = detectedFaces.faceRect[0].bottom;
        SingleDetectedFaces.faceOrient = detectedFaces.faceOrient[0];
        res = ASFFaceFeatureExtractEx(_handle, &offscreen, &SingleDetectedFaces, &feature);
        if (MOK != res)
            printf("ASFFaceFeatureExtractEx 2 fail: %d\n", res);
    }else if(res==MOK && detectedFaces.faceNum==0){
        return false;
    }else if(!(res==MOK && detectedFaces.faceNum==1)){
        std::cerr<<"ASFDetectFacesEx fail :"<<res<<std::endl;
    }

    MInt32 processMask =ASF_LIVENESS;
    res = ASFProcessEx(_handle, &offscreen, &detectedFaces, processMask);
    if (res != MOK)
        printf("ASFSProcessEx fail: %d\n", res);
//    else
//        printf("ASFProcessEx sucess: %d\n", res);

    ASF_LivenessInfo rgbLivenessInfo = { 0 };
    res = ASFGetLivenessScore(_handle, &rgbLivenessInfo);
//    std::cout<<"liveness:"<<rgbLivenessInfo.isLive[0]<<std::endl;
    if (res != MOK)
        printf("ASFGetLivenessScore fail: %d\n", res);
    else
        return rgbLivenessInfo.isLive[0];
    return false;
}
