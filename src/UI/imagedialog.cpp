#include "imagedialog.h"
#include "ui_imagedialog.h"
#include <QIntValidator>

#include "ErrorMsg.hpp"
ImageDialog::ImageDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ImageDialog)
{
    ui->setupUi(this);
    connect(ui->buttonBox,&QDialogButtonBox::accepted,this,&ImageDialog::accpetSlot);
    connect(this, SIGNAL(sendPixmap(QPixmap)),ui->frameLabel, SLOT(setPixmap(QPixmap)));
    ui->idEdit->setValidator(new QIntValidator(ui->idEdit));
}

ImageDialog::~ImageDialog()
{
    delete ui;
}

void ImageDialog::accpetSlot() {
    auto id=ui->idEdit->text();
    auto name=ui->nameEdit->text();
    auto &fm=FileManager::getInstance();
    if(fm.getFaceData().count({id.toStdString(),name.toStdString()}))
        errorMsg(id+","+name+" 已存在",this);
    else{
        fm.appendFace({id.toStdString(),name.toStdString()},norm);
        _id=id.toStdString();
        write=true;
    }

}
