//
// Created by q1162 on 2023/4/12.
//

#ifndef FACERECOGNITION_ERRORMSG_HPP
#define FACERECOGNITION_ERRORMSG_HPP
#include <QString>
#include <QErrorMessage>
#include <QWidget>
inline void errorMsg(QString str,QWidget* parent){
    QErrorMessage msg(parent);
    msg.showMessage(str);
    msg.exec();
}
#endif //FACERECOGNITION_ERRORMSG_HPP
