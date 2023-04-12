#include "facedialog.h"
#include "ui_facedialog.h"
#include "../FileManager.hpp"
#include "ErrorMsg.hpp"
#include <QIntValidator>
FaceDialog::FaceDialog(bool m,QWidget *parent) :
    QDialog(parent),
    ui(new Ui::FaceDialog),mode(m)
{
    ui->setupUi(this);
    if(mode==true){
        this->setWindowTitle("添加人脸");
    }else{
        this->setWindowTitle("删除人脸");
        ui->nameEdit->setReadOnly(true);
    }
    connect(ui->buttonBox,&QDialogButtonBox::accepted,this,&FaceDialog::accpetSlot);
    ui->idEdit->setValidator(new QIntValidator(ui->idEdit));
}

FaceDialog::~FaceDialog()
{
    delete ui;
}

void FaceDialog::accpetSlot() {
    auto id=ui->idEdit->text();
    auto name=ui->nameEdit->text();
    auto &fm=FileManager::getInstance();
    if(mode==true){
        if(fm.getFaceData().count({id.toStdString(),name.toStdString()}))
            errorMsg(id+","+name+" 已存在",this);
        else
            emit updateData(mode,id,name);
    }else{
        if(!fm.getFaceData().count({id.toStdString(),name.toStdString()}))
            errorMsg(id+","+name+" 不存在",this);
        else{
            fm.eraseFace(id.toStdString());
        }
    }
}

void FaceDialog::setMode(bool m) {
    mode=m;
    if(mode==true){
        this->setWindowTitle("添加人脸");
        ui->nameEdit->setReadOnly(false);
    }else{
        this->setWindowTitle("删除人脸");
        ui->nameEdit->setReadOnly(true);
    }
    ui->nameEdit->clear();
    ui->idEdit->clear();
}
