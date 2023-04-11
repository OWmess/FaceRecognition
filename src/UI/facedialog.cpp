#include "facedialog.h"
#include "ui_facedialog.h"

FaceDialog::FaceDialog(bool m,QWidget *parent) :
    QDialog(parent),
    ui(new Ui::FaceDialog),mode(m)
{
    ui->setupUi(this);
    if(mode==true){
        this->setWindowTitle("添加人脸");
    }else{
        this->setWindowTitle("删除人脸");
        ui->textEdit_2->setReadOnly(true);
    }




}

FaceDialog::~FaceDialog()
{
    delete ui;
}
