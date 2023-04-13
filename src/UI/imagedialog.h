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
    void setPixmap(const QPixmap &t);
    void setNorm(const cv::Mat& n){
        norm=n;
    }

    bool getWriteFlag(std::string& str){
        str=_id;
        bool t=write;
        write=false;
        return t;
    }

    void setMode(bool m){
        mode=m;
    }
public slots:
    void updateNameText(QString text);

    signals:
    void sendPixmap(const QPixmap&);

private slots:
    void accpetSlot();
private:
    Ui::ImageDialog *ui;
    QPixmap pixmap;
    cv::Mat norm;
    bool write=false;
    std::string _id;
    //false是添加，true是检测
    bool mode;
};

#endif // IMAGEDIALOG_H
