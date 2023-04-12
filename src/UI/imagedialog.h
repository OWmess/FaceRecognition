#ifndef IMAGEDIALOG_H
#define IMAGEDIALOG_H

#include <QDialog>
#include "../FileManager.hpp"
namespace Ui {
class ImageDialog;
}

class ImageDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ImageDialog(QWidget *parent = nullptr);
    ~ImageDialog();
    void setPixmap(const QPixmap &t){
        emit sendPixmap(t);
    }
    void setNorm(const cv::Mat& n){
        norm=n;
    }

    bool getWriteFlag(std::string& str){
        str=_id;
        bool t=write;
        write=false;
        return t;
    }
public slots:

    signals:
    void sendPixmap(const QPixmap&);

    void sendFaceInfo(const FaceInfo&);


private slots:
    void accpetSlot();
private:
    Ui::ImageDialog *ui;
    QPixmap pixmap;
    cv::Mat norm;
    bool write=false;
    std::string _id;
};

#endif // IMAGEDIALOG_H
