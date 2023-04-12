//
// Created by q1162 on 2023/3/28.
//

#ifndef FACERECOGNITION_CONFIG_H
#define FACERECOGNITION_CONFIG_H

#define PRJ_NAME "FaceRecognition"
#define USE_FP16
#define DEVICE 0
#define BATCH_SIZE 1
#define CONF_THRESH 0.75
#define IOU_THRESH 0.4
#define SAVE_FORMAT 512,1
#define INFER_FORMAT 1,512
#define ANTI_SPOOFING_SCALE 4.0
#define ANTI_SPOOFING_THRESH 0.75
#define ENABLE_ANTI 0
#define ROTATE_CAMERA 0
#define CONTRAST_THRESH 0.4
namespace MODELCONFIG{
    class ARCFACE{
    public:
        static constexpr int INPUT_W=112;
        static constexpr int INPUT_H=112;
        static constexpr int OUTPUT_SIZE=512;
    };
    class RETINAFACE{
    public:
        static constexpr int INPUT_W=640;
        static constexpr int INPUT_H=480;
        static constexpr int OUTPUT_SIZE=(INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 +INPUT_H / 32 * INPUT_W / 32) * 2  * 15 + 1;
    };
    class ANTISPOOLING{
    public:
        static constexpr int INPUT_W=80;
        static constexpr int INPUT_H=80;
        static constexpr int OUTPUTSIZE=3;
    };

}

#endif //FACERECOGNITION_CONFIG_H
