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

private:
    Ui::FaceDialog *ui;
    bool mode;
};

#endif // FACEDIALOG_H
