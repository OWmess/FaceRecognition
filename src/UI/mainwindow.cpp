#include "mainwindow.h"
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

void DisplayThread::updateImage(const cv::Mat& image,QString str)
{
    this->image = image.clone();
    this->nameStr=str;
    newImage = true;
    qDebug()<<str<<Qt::endl;
}

void DisplayThread::run()
{
    while (!isInterruptionRequested()) {
        if (newImage) {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            QImage qimage(image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);

            emit sendImage(QPixmap::fromImage(qimage));
            emit sendNameStr(nameStr);
            newImage = false;
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
    connect(&handleThread, SIGNAL(handleReady(const cv::Mat&,QString)),&displayThread, SLOT(updateImage(const cv::Mat&,QString)));
    connect(&displayThread, SIGNAL(sendImage(QPixmap)), ui->label, SLOT(setPixmap(QPixmap)));
    connect(&displayThread, SIGNAL(sendNameStr(QString)), ui->nameLabel, SLOT(setText(QString)));
    connect(ui->appendAction, SIGNAL(triggered()),this,SLOT(faceAppendSlot()));
    connect(ui->deleteAction, SIGNAL(triggered()),this,SLOT(faceDeleteSlot()));
    connect(ui->appendImgAction, SIGNAL(triggered()),this,SLOT(appendImgSlot()));
    connect(ui->detectImgAction, SIGNAL(triggered()),this,SLOT(detectImgSlot()));
    connect(faceDialog.get(), SIGNAL(updateData(bool, QString, QString)), &handleThread, SLOT(updateData(bool, QString, QString)));
    connect(&handleThread, SIGNAL(detectorEmpty()),this, SLOT(detectorEmptySlot()));
    connect(this,SIGNAL(detectImgSendStr(QString)),imageDialog.get(),SLOT(updateNameText(QString)));


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
