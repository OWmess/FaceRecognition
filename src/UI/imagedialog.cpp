#include "imagedialog.h"
#include "ui_imagedialog.h"
#include <QIntValidator>

#include "ErrorMsg.hpp"
ImageDialog::ImageDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ImageDialog),mode(false)
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
    auto id = ui->idEdit->text();
    auto name = ui->nameEdit->text();
    auto &fm = FileManager::getInstance();
    std::string nameStr;
    if(mode==true) {

        std::for_each(fm.getFaceData().begin(), fm.getFaceData().end(), [&](const auto &pair) {
            cv::Mat res = norm * pair.second;
            float score = *(float *) res.data;
            if (score > CONTRAST_THRESH) {
                nameStr += " " + pair.first.name;
//                std::cout << pair.first.name << " conf: " << score << std::endl;
            }
            return;
        });
        ui->nameEdit->setText(QString::fromStdString(nameStr));
    }else{
        if (fm.getFaceData().count({id.toStdString(), name.toStdString()}))
            errorMsg(id + "," + name + " 已存在", this);
        else {
            fm.appendFace({id.toStdString(), name.toStdString()}, norm);
            _id = id.toStdString();
            write = true;
        }
    }
}

void ImageDialog::setPixmap(const QPixmap &t) {
    ui->idEdit->clear();
    ui->nameEdit->clear();
    ui->idEdit->setReadOnly(mode);
    ui->idEdit->setReadOnly(mode);

    emit sendPixmap(t);
}

void ImageDialog::updateNameText(QString text) {
    ui->nameEdit->setText(text);
}
