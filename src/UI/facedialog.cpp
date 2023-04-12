#include "facedialog.h"
#include "ui_facedialog.h"
#include "../FileManager.hpp"
#include "ErrorMsg.hpp"
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

}

FaceDialog::~FaceDialog()
{
    delete ui;
}

void FaceDialog::accpetSlot() {
    auto id=ui->idEdit->text();
    auto name=ui->nameEdit->text();
    auto &fm=FileManager::getInstance();
    if(fm.findExist(id.toStdString())&&mode==true){
        errorMsg(id+","+name+" 已存在",this);
    }else{
        emit updateData(mode,id,name);
    }
}
