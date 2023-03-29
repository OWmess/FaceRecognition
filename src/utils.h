//
// Created by q1162 on 2023/3/29.
//

#ifndef FACERECOGNITION_UTILS_H
#define FACERECOGNITION_UTILS_H
#include "config.h"
#include <filesystem>

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
#endif //FACERECOGNITION_UTILS_H
