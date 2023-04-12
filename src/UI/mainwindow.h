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
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class CameraThread : public QThread
{
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

class DisplayThread : public QThread
{
    Q_OBJECT

public:
    DisplayThread(QObject *parent = nullptr);
    ~DisplayThread();

public slots:
    void updateImage(const cv::Mat& image,QString str);

signals:
    void sendNameStr(QString);
    void sendImage(QPixmap image);

protected:
    void run() override;

private:
    cv::Mat image;
    bool newImage;
    QString nameStr;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

public slots:
    void faceAppendSlot() {
        dialog->setMode(true);
        dialog->exec();
        this->update();
    }

    void faceDeleteSlot() {
        dialog->setMode(false);
        dialog->exec();
        this->update();
    }

    void detectSlot(){
        QStringList filesPath = QFileDialog::getOpenFileNames(this, tr("选择图片"), QDir::homePath(), tr("Image Files(*.png *.jpg *.bmp);;All Files (*.*)"));
        if (!filesPath.isEmpty()) {
            for(const auto& i:filesPath.toVector()){
                cv::Mat img=cv::imread(i.toStdString());

            }


            // 处理所选文件的路径
        }

    }

    void detectorEmptySlot(){
        qDebug()<<"未检测到人脸"<<Qt::endl;
        errorMsg("未检测到人脸",this);
    }

private:
    std::unique_ptr<FaceDialog> dialog = std::make_unique<FaceDialog>(false,this);
    Ui::MainWindow *ui;
    CameraThread cameraThread;
    DisplayThread displayThread;
    HandleThread handleThread;

};

#endif // MAINWINDOW_H
