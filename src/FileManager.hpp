//
// Created by q1162 on 2023/4/12.
//

#ifndef FACERECOGNITION_FILEMANAGER_HPP
#define FACERECOGNITION_FILEMANAGER_HPP
#include <iostream>
#include <filesystem>
#include "utils.h"
class FaceInfo{
public:
    std::string id;
    std::string name;
    bool operator<(const FaceInfo& other) const{
        return this->id<other.id;
    }
};
using FaceMap=std::map<FaceInfo,cv::Mat>;
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

    std::string getSaveDir(){
        return _saveDir;
    }

    void eraseFace(const std::string& id){
        auto iter=std::find_if(_faceData.begin(),_faceData.end(),[&](const auto& pair){
            return id==pair.first.id;
        });
        _faceData.erase(iter);
        auto deletePath=_saveDir+id+".jpg";
        std::filesystem::remove(deletePath);
    }

    void appendFace(const FaceInfo& info, const cv::Mat& norm){
        _faceData.insert(std::make_pair(info, norm));

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
