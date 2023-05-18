#include "mainwindow.h"

#include <utility>
#include <QPainter>
#include "ui_mainwindow.h"
#include "../config.h"
CameraThread::CameraThread(QObject *parent)
    : QThread(parent)
{
    try {
        _camera.open(0);
    }catch(const std::exception& e){
        std:: string str=e.what();
        errorMsg(QString::fromStdString(str));
    }
}

CameraThread::~CameraThread()
{
    _camera.release();
}

void CameraThread::run()
{
    while (!isInterruptionRequested()) {

        _camera.read(_frame);
        cv::resize(_frame,_frame,{640,480});
#if ROTATE_CAMERA
        cv::transpose(_frame, _frame);
        cv::flip(_frame, _frame, 0);
#endif
        emit imageReady(_frame);
    }
}

DisplayThread::DisplayThread(QObject *parent)
    : QThread(parent), newImage(false)
{
}

DisplayThread::~DisplayThread()
{
}

void DisplayThread::updateImage(const cv::Mat& image,std::vector<NameFormat> nameVec)
{
    this->image = image.clone();
    this->_nameVec=nameVec;
    newImage = true;
//    qDebug()<<str<<Qt::endl;
}

void DisplayThread::run()
{
    while (!isInterruptionRequested()) {
        if (newImage) {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            QImage qimage(image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
            QPainter painter(&qimage);
            // 设置字体属性
            QFont font("Arial", 10, QFont::Bold);
            painter.setFont(font);

            // 设置文本颜色
            QPen pen(Qt::yellow);
            painter.setPen(pen);
            QString nameStr;
            for(const auto &name:_nameVec){
                painter.drawText(QRect(name.pt.x, name.pt.y, image.cols, image.rows), name.name);
                nameStr.append(" "+name.name);
            }



            emit sendImage(QPixmap::fromImage(qimage));
            emit sendNameStr(nameStr);
            newImage = false;
            auto fps=calFrameFps();
            printf("fps:%.1f\n",fps);
        }
        msleep(10);
    }
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    cameraThread.start();
    displayThread.start();
    handleThread.start();
    connect(&cameraThread, SIGNAL(imageReady(cv::Mat)), &handleThread, SLOT(updateFrame(cv::Mat)));
    connect(&handleThread, SIGNAL(handleReady(const cv::Mat&,std::vector<NameFormat>)), &displayThread, SLOT(updateImage(const cv::Mat&,std::vector<NameFormat>)));
    connect(&displayThread, SIGNAL(sendImage(QPixmap)), ui->label, SLOT(setPixmap(QPixmap)));
    connect(&displayThread, SIGNAL(sendNameStr(QString)), ui->nameLabel, SLOT(setText(QString)));
    connect(ui->appendAction, SIGNAL(triggered()),this,SLOT(faceAppendSlot()));
    connect(ui->deleteAction, SIGNAL(triggered()),this,SLOT(faceDeleteSlot()));
    connect(ui->appendImgAction, SIGNAL(triggered()),this,SLOT(appendImgSlot()));
    connect(ui->detectImgAction, SIGNAL(triggered()),this,SLOT(detectImgSlot()));
    connect(faceDialog.get(), SIGNAL(updateData(bool, QString, QString)), &handleThread, SLOT(updateData(bool, QString, QString)));
    connect(&handleThread, SIGNAL(detectorEmpty()), this, SLOT(detectorEmptySlot()));
    connect(this,SIGNAL(detectImgSendStr(QString)),imageDialog.get(),SLOT(updateNameText(QString)));
    connect(ui->saveFaceBaseAction, SIGNAL(triggered()),this,SLOT(saveFaceBaseSlot()));
    connect(ui->loadFaceBaseAction, SIGNAL(triggered()),this,SLOT(loadFaceBaseSlot()));
}

MainWindow::~MainWindow()
{
    handleThread.requestInterruption();
    handleThread.wait();
    cameraThread.requestInterruption();
    cameraThread.wait();
    displayThread.requestInterruption();
    displayThread.wait();
    delete ui;
}
