//
// Created by q1162 on 2023/4/12.
//

#ifndef FACERECOGNITION_FILEMANAGER_HPP
#define FACERECOGNITION_FILEMANAGER_HPP
#include <iostream>
#include <filesystem>
#include "utils.h"
using FaceMap=std::map<std::string,cv::Mat>;
class FileManager {
public:
    static FileManager& getInstance() {
        static FileManager instance;
        return instance;
    }

    inline bool findExist(const std::string& filename) {
        for (const auto& file : std::filesystem::directory_iterator(_saveDir)) {
            if (file.is_regular_file() && file.path().filename() == filename+".jpg") {
                return true;
            }
        }
        return false;
    }

    FaceMap& getFaceData(){
        return _faceData;
    }
    FileManager(const FileManager&) = delete;
    FileManager& operator=(const FileManager&) = delete;
private:
    FileManager() {
    }
    ~FileManager() = default;


    const std::string _projDir=GET_PRJ_DIR();
    const std::string _saveDir=_projDir+"/savedata/";
    FaceMap _faceData;
};





#endif //FACERECOGNITION_FILEMANAGER_HPP
