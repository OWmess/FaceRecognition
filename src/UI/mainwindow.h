#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <QPixmap>
#include <QThread>
#include <QFileDialog>
#include "HandleThread.hpp"
#include "facedialog.h"
#include "ErrorMsg.hpp"
#include "imagedialog.h"
#include <string>
#include <filesystem>
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class CameraThread : public QThread {
Q_OBJECT

public:
    CameraThread(QObject *parent = nullptr);

    ~CameraThread();

protected:
    void run() override;

signals:

    void imageReady(cv::Mat image);

private:
    cv::Mat _frame;
    cv::VideoCapture _camera;
};

class DisplayThread : public QThread {
Q_OBJECT

public:
    DisplayThread(QObject *parent = nullptr);

    ~DisplayThread();

public slots:

    void updateImage(const cv::Mat &image, std::vector<NameFormat> nameVec);

signals:

    void sendNameStr(QString);

    void sendImage(QPixmap image);

protected:
    void run() override;

private:
    cv::Mat image;
    bool newImage;
    std::vector<NameFormat> _nameVec;
};

class MainWindow : public QMainWindow {
Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);

    ~MainWindow();

signals:

    void detectImgSendStr(QString s);

public slots:

    void faceAppendSlot() {
        faceDialog->setMode(true);
        faceDialog->exec();
        this->update();
    }

    void faceDeleteSlot() {
        faceDialog->setMode(false);
        faceDialog->exec();
        this->update();
    }

    void appendImgSlot() {
        QString filesPath = QFileDialog::getOpenFileName(this, tr("选择图片"), QDir::homePath(),
                                                         tr("Image Files(*.png *.jpg *.bmp);;All Files (*.*)"));
        std::string str_path = filesPath.toLocal8Bit().constData();
        if (!filesPath.isEmpty()) {
            cv::Mat img = cv::imread(str_path);
            if (img.empty()) {
                errorMsg("无法打开文件,请确保路径中不包含中文!", this);
                return;
            }
            cv::resize(img, img, {640, 480});

            cv::Mat outImg = img.clone();
            auto norm = handleThread.appendProcess(img, outImg, SAVE_FORMAT);
            if (norm.cols == 0 && norm.rows == 0) {
                errorMsg("未识别到人脸!", this);
                return;
            }
            cv::Mat tmp;
            cv::cvtColor(outImg, tmp, cv::COLOR_BGR2RGB);
            QImage qimage(tmp.data, tmp.cols, tmp.rows, tmp.step, QImage::Format_RGB888);
            QPixmap qpixmap = QPixmap::fromImage(qimage);
            imageDialog->setMode(false);
            imageDialog->setPixmap(qpixmap);
            imageDialog->setNorm(norm);
            imageDialog->exec();
            // 处理所选文件的路径
        }

    }

    void detectImgSlot() {
        QString filesPath = QFileDialog::getOpenFileName(this, tr("选择图片"), QDir::homePath(),
                                                         tr("Image Files(*.png *.jpg *.bmp);;All Files (*.*)"));
        std::string str_path = filesPath.toLocal8Bit().constData();
        if (!filesPath.isEmpty()) {
            cv::Mat img = cv::imread(str_path);
            if (img.empty()) {//TODO 暂时不能使用带中文的路径
                errorMsg("无法打开文件,请确保路径中不包含中文!", this);
                return;
            }
            cv::resize(img, img, {640, 480});
            cv::Mat outImg = img.clone();
            auto norm = handleThread.appendProcess(img, outImg, INFER_FORMAT);
            if (norm.cols == 0 && norm.rows == 0) {
                errorMsg("未识别到人脸!", this);
                return;
            }
            cv::Mat tmp;
            cv::cvtColor(outImg, tmp, cv::COLOR_BGR2RGB);
            QImage qimage(tmp.data, tmp.cols, tmp.rows, tmp.step, QImage::Format_RGB888);
            QPixmap qpixmap = QPixmap::fromImage(qimage);
            imageDialog->setMode(true);
            imageDialog->setPixmap(qpixmap);
            imageDialog->setNorm(norm);
            auto &fm = FileManager::getInstance();
            std::string nameStr;
            std::for_each(fm.getFaceData().begin(), fm.getFaceData().end(), [&](const auto &pair) {
                cv::Mat res = norm * pair.second;
                float score = *(float *) res.data;
                if (score > CONTRAST_THRESH) {
                    nameStr += " " + pair.first.name;
//                    std::cout << pair.first.name << " conf: " << score << std::endl;
                }
                return;
            });
            emit detectImgSendStr(QString::fromStdString(nameStr));
            imageDialog->exec();
        }
    }

    void detectorEmptySlot() {
        qDebug() << "未检测到人脸" << Qt::endl;
        errorMsg("未检测到人脸", this);
    }

    void saveFaceBaseSlot() {

        QString filesPath = QFileDialog::getSaveFileName(this, tr("选择路径"), QDir::homePath(),
                                                         tr("Faces Files(*.facedata)"));
        if(filesPath.isEmpty())
            return;
        auto &fm=FileManager::getInstance();
        if(!fm.saveFaceInfo(filesPath))
            errorMsg("无法打开文件,请确保路径中不包含中文!", this);

    }

    void loadFaceBaseSlot(){
        QString filesPath = QFileDialog::getOpenFileName(this, tr("选择文件"), QDir::homePath(),
                                                         tr("Faces Files(*.facedata)"));
        if(filesPath.isEmpty())
            return;
        auto &fm=FileManager::getInstance();
        if(!fm.loadFaceInfo(filesPath))
            errorMsg("无法打开文件,请确保路径中不包含中文!", this);
    }
private:
    std::unique_ptr<FaceDialog> faceDialog = std::make_unique<FaceDialog>(false, this);
    std::unique_ptr<ImageDialog> imageDialog = std::make_unique<ImageDialog>(this);
    Ui::MainWindow *ui;
    CameraThread cameraThread;
    DisplayThread displayThread;
    HandleThread handleThread;

};

#endif // MAINWINDOW_H
