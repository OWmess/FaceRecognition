#ifndef FACEDIALOG_H
#define FACEDIALOG_H

#include <QDialog>

namespace Ui {
class FaceDialog;
}

class FaceDialog : public QDialog
{
    Q_OBJECT

public:
    explicit FaceDialog(bool mode,QWidget *parent = nullptr);
    ~FaceDialog();

    void setMode(bool mode);
signals:
    void updateData(bool mode,QString id,QString name);

private slots:
    void accpetSlot();

private:
    Ui::FaceDialog *ui;
    //true为添加人脸
    bool mode;
};

#endif // FACEDIALOG_H
