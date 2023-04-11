#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <QPixmap>
#include <QThread>
#include "HandleThread.hpp"

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
    void updateImage(cv::Mat image);

signals:
    void sendImage(QPixmap image);

protected:
    void run() override;

private:
    cv::Mat image;
    bool newImage;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    CameraThread cameraThread;
    DisplayThread displayThread;
    HandleThread handleThread;
};

#endif // MAINWINDOW_H
