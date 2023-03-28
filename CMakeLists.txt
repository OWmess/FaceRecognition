cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES 75)
project(FaceRecognition CUDA)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)
#添加opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#添加tensorrt
include_directories(C:\\Program\ Files\\NVIDIA\\TensorRT-8.5.3.1\\include)
link_directories(C:\\Program\ Files\\NVIDIA\\TensorRT-8.5.3.1\\lib)


find_package(CUDA REQUIRED)
#用cuda编译
cuda_add_library(decodeplugin SHARED ${PROJECT_SOURCE_DIR}/src/RetinaFace/decode.cu)
target_link_libraries(decodeplugin nvinfer cudart)
cuda_add_library(preluplugins SHARED ${PROJECT_SOURCE_DIR}/src/ArcFace/prelu.cu)
target_link_libraries(preluplugins nvinfer cudart)


#
file(GLOB_RECURSE SOURCE_FILES "${PROJECT_SOURCE_DIR}/Src/*.cpp")
add_executable(${PROJECT_NAME} src/main.cpp ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} nvinfer cudart preluplugins decodeplugin ${OpenCV_LIBS})
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)