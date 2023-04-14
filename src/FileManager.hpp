//
// Created by q1162 on 2023/4/12.
//

#ifndef FACERECOGNITION_FILEMANAGER_HPP
#define FACERECOGNITION_FILEMANAGER_HPP
#include <iostream>
#include <filesystem>
#include "utils.h"
#include <fstream>
class FaceInfo{
public:
    std::string id;
    std::string name;

    // Custom copy constructor to ensure proper handling of std::string members
    FaceInfo(const FaceInfo& other) : id(other.id), name(other.name) {}

    // Custom destructor to release memory owned by std::string members
    ~FaceInfo() {}

    // Default constructor
    FaceInfo() {}

    // Constructor with parameters
    FaceInfo(const std::string& id_, const std::string& name_)
            : id(id_), name(name_) {}

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

    const FaceMap& getFaceData() {
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
        saveFaceInfo();
    }

    void appendFace(const FaceInfo& info, const cv::Mat& norm){
        _faceData.insert(std::make_pair(info, norm));
        saveFaceInfo();
    }

    FileManager(const FileManager&) = delete;
    FileManager& operator=(const FileManager&) = delete;
private:
    FileManager() {
        loadFaceInfo();
    }
    ~FileManager() = default;

    bool saveFaceInfo(){
        cv::FileStorage fs(_saveDir+"FaceInfo.yaml",cv::FileStorage::WRITE| cv::FileStorage::FORMAT_YAML);
        if (!fs.isOpened())
        {
            std::cerr << "Failed to save face data to yaml "<< std::endl;
            return false;
        }
        fs << "FaceMap" << "{";
        int cnt=1;
        for(const auto &it:_faceData){
            fs<<"Map_"+std::to_string(cnt++)<<"{";
            fs<<"id"<<it.first.id;
            fs<<"name"<<it.first.name;
            fs<<"mat"<<it.second;
            fs<<"}";
        }
        fs<<"}";
        fs.release();
        return true;

    }

    bool loadFaceInfo(){
        cv::FileStorage fs(_saveDir+"FaceInfo.yaml",cv::FileStorage::READ| cv::FileStorage::FORMAT_YAML);
        _faceData.clear();
        if (!fs.isOpened())
        {
            std::cerr << "Failed to save face data to yaml "<< std::endl;
            return false;
        }

        cv::FileNode node = fs["FaceMap"];
        if (node.empty())
        {
            std::cerr << "No data found in file "<<std::endl;
            return false;
        }
        int cnt=1;
        for(auto it:node){
            FaceInfo tmpInfo;
            cv::Mat tmpMat;
            std::string index="Map_"+std::to_string(cnt++);
            node[index]["id"]>>tmpInfo.id;
            node[index]["name"]>>tmpInfo.name;
            node[index]["mat"]>>tmpMat;

            _faceData.insert(std::pair(tmpInfo,tmpMat));
        }

        fs.release();
        return true;
    }

    const std::string _projDir=GET_PRJ_DIR();
    const std::string _saveDir=_projDir+"/savedata/";
    FaceMap _faceData;
};





#endif //FACERECOGNITION_FILEMANAGER_HPP
