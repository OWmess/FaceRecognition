cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES 75)
project(FaceRecognition VERSION 0.1 LANGUAGES CXX CUDA)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_GENERATOR "Visual Studio 17 2022")
set(CMAKE_PREFIX_PATH C:/Qt/6.5.0/msvc2019_64)
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)


#添加opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#添加tensorrt
include_directories(C:\\Program\ Files\\NVIDIA\\TensorRT-8.5.3.1\\include)
link_directories(C:\\Program\ Files\\NVIDIA\\TensorRT-8.5.3.1\\lib)


find_package(CUDA REQUIRED)
#用cuda编译
cuda_add_library(decodeplugin SHARED src/TRTInfer/RetinaFace/decode.cu)
target_link_libraries(decodeplugin nvinfer cudart)
cuda_add_library(preluplugins SHARED src/TRTInfer/ArcFace/prelu.cu)
target_link_libraries(preluplugins nvinfer cudart)

add_library(trtinfer SHARED src/TRTInfer/TRTInfer.cpp)
target_link_libraries(trtinfer nvonnxparser nvinfer cudart cuda decodeplugin ${OpenCV_LIBS})

add_library(antispoofing SHARED src/TRTInfer/AntiSpoofing/AntiSpoofing.cpp)
target_link_libraries(antispoofing nvonnxparser nvinfer cudart cuda decodeplugin preluplugins trtinfer ${OpenCV_LIBS})

add_library(arcface SHARED src/TRTInfer/ArcFace/ArcFace_R100.cpp)
target_link_libraries(arcface nvinfer cudart cuda decodeplugin preluplugins trtinfer ${OpenCV_LIBS})

add_library(retinaface SHARED src/TRTInfer/RetinaFace/RetinaFace_R50.cpp)
target_link_libraries(retinaface nvinfer cudart cuda decodeplugin preluplugins trtinfer ${OpenCV_LIBS})

#

set(PROJECT_FILE
        ${PROJECT_SOURCE_DIR}/src/main.cpp
        ${PROJECT_SOURCE_DIR}/src/UI/HandleThread.cpp
        ${PROJECT_SOURCE_DIR}/src/UI/mainwindow.cpp
        ${PROJECT_SOURCE_DIR}/src/UI/mainwindow.h
        ${PROJECT_SOURCE_DIR}/src/UI/mainwindow.ui
        src/UI/facedialog.h src/UI/facedialog.cpp src/UI/facedialog.ui
        src/FileManager.hpp src/UI/ErrorMsg.hpp)
#file(GLOB_RECURSE SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/UI/.cpp")
#file(GLOB_RECURSE UI_FILES "${PROJECT_SOURCE_DIR}/src/UI/*.ui")
#if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(${PROJECT_NAME}
            MANUAL_FINALIZATION
#            ${SOURCE_FILES}
#            ${UI_FILES}
#            ${PROJECT_SOURCE_DIR}/src/main.cpp
            ${PROJECT_FILE}
            )
    # Define target properties for Android with Qt 6 as:
    #    set_property(TARGET FaceRecognitionUI APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
    #                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
    # For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
#else()
#    if(ANDROID)
#        add_library(${PROJECT_NAME} SHARED
#                ${PROJECT_SOURCES}
#                )
#        # Define properties for Android with Qt 5 after find_package() calls as:
#        #    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
#    else()
#        add_executable(${PROJECT_NAME}
#                ${PROJECT_SOURCES}
#                )
#    endif()
#endif()




#add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cpp ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}
        PRIVATE antispoofing arcface retinaface trtinfer
        Qt${QT_VERSION_MAJOR}::Widgets ${OpenCV_LIBS})
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
#        WIN32_EXECUTABLE TRUE
        )

install(TARGETS ${PROJECT_NAME}
        BUNDLE DESTINATION .
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(${PROJECT_NAME})
endif()
