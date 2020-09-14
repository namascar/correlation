/********************************************************************************
** Form generated from reading UI file 'mainapp.ui'
**
** Created by: Qt User Interface Compiler version 5.6.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINAPP_H
#define UI_MAINAPP_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainApp {
public:
  QAction *action_Open;
  QAction *action_Print;
  QAction *action_Exit;
  QAction *actionZoom_in_25;
  QAction *actionZoom_Out_25;
  QAction *action_Normal_Size;
  QAction *actionFit_to_Window;
  QAction *action_Select_Area;
  QAction *actionParame_ters;
  QAction *action_Correlate;
  QAction *action_About;
  QAction *action_Save_Analysis;
  QAction *actiong;
  QWidget *centralwidget;
  QScrollArea *scrollArea_all_images;
  QWidget *scrollAreaWidgetContents;
  QLabel *label_correlation_model;
  QComboBox *comboBox_correlation_model;
  QComboBox *comboBox_interpolation_model;
  QLabel *label_interpolation_model;
  QTabWidget *QTabWidget_domain;
  QWidget *Rectangular;
  QLabel *label_domain_2;
  QSpinBox *spinBox_rect_horizontal;
  QLabel *label_domain_3;
  QSpinBox *spinBox_rect_vertical;
  QLabel *label_domain_4;
  QDoubleSpinBox *doubleSpinBox_rect_y0;
  QLabel *label_domain_8;
  QDoubleSpinBox *doubleSpinBox_rect_x0;
  QLabel *label_domain_9;
  QDoubleSpinBox *doubleSpinBox_rect_xend;
  QDoubleSpinBox *doubleSpinBox_rect_yend;
  QLabel *label_domain_10;
  QLabel *label_domain_11;
  QWidget *Annular;
  QSpinBox *spinBox_ann_radial;
  QLabel *label_domain_6;
  QLabel *label_domain_7;
  QSpinBox *spinBox_ann_angular;
  QLabel *label_domain_5;
  QDoubleSpinBox *doubleSpinBox_ann_center_x;
  QDoubleSpinBox *doubleSpinBox_ann_center_y;
  QLabel *label_domain_22;
  QDoubleSpinBox *doubleSpinBox_ann_ri;
  QLabel *label_domain_23;
  QDoubleSpinBox *doubleSpinBox_ann_ro;
  QLabel *label_domain_24;
  QLabel *label_domain_25;
  QLabel *label_domain_27;
  QLabel *label_domain_29;
  QWidget *Blob;
  QLabel *label_domain_30;
  QLabel *label_domain_28;
  QLabel *label_domain_26;
  QLabel *label_domain_31;
  QDoubleSpinBox *doubleSpinBox_blob_center_x;
  QLabel *label_domain_32;
  QLabel *label_domain_33;
  QDoubleSpinBox *doubleSpinBox_blob_center_y;
  QDoubleSpinBox *doubleSpinBox_blob_x_scale;
  QDoubleSpinBox *doubleSpinBox_blob_y_scale;
  QLabel *label_domain;
  QLabel *label_Initial_Guess;
  QComboBox *comboBox_Initial_Guess;
  QComboBox *comboBox_update;
  QLabel *label_Initial_Guess_2;
  QLabel *label_Initial_Guess_3;
  QLabel *label_Initial_Guess_4;
  QComboBox *comboBox_color;
  QSpinBox *spinBox_max_iters;
  QLabel *label_Initial_Guess_5;
  QDoubleSpinBox *doubleSpinBox_precision;
  QSpinBox *spinBox_py_start;
  QLabel *label_Initial_Guess_6;
  QLabel *label_stop_pyram;
  QSpinBox *spinBox_py_stop;
  QSpinBox *spinBox_py_step;
  QLabel *label_Initial_Guess_8;
  QPushButton *pushButton_show_sets;
  QPushButton *pushButton_correlate;
  QLabel *label_undeformed_name;
  QLabel *label_deformed_name;
  QProgressBar *progressBar;
  QScrollArea *scrollArea_und_image;
  QWidget *scrollAreaWidgetContents_2;
  QScrollArea *scrollArea_def_image;
  QWidget *scrollAreaWidgetContents_3;
  QScrollArea *scrollArea_und_set;
  QWidget *scrollAreaWidgetContents_4;
  QScrollArea *scrollArea_def_set;
  QWidget *scrollAreaWidgetContents_5;
  QScrollArea *scrollArea_und_info;
  QWidget *scrollAreaWidgetContents_6;
  QScrollArea *scrollArea_def_info;
  QWidget *scrollAreaWidgetContents_7;
  QLabel *label_results_display;
  QLabel *label_best_rotation_value;
  QLabel *label_best_rotation;
  QMenuBar *menubar;
  QMenu *menu_File;
  QMenu *menu_View;
  QMenu *menu_Tools;
  QMenu *menu_Help;
  QStatusBar *statusbar;

  void setupUi(QMainWindow *MainApp) {
    if (MainApp->objectName().isEmpty())
      MainApp->setObjectName(QStringLiteral("MainApp"));
    MainApp->resize(1551, 734);
    action_Open = new QAction(MainApp);
    action_Open->setObjectName(QStringLiteral("action_Open"));
    action_Print = new QAction(MainApp);
    action_Print->setObjectName(QStringLiteral("action_Print"));
    action_Exit = new QAction(MainApp);
    action_Exit->setObjectName(QStringLiteral("action_Exit"));
    actionZoom_in_25 = new QAction(MainApp);
    actionZoom_in_25->setObjectName(QStringLiteral("actionZoom_in_25"));
    actionZoom_Out_25 = new QAction(MainApp);
    actionZoom_Out_25->setObjectName(QStringLiteral("actionZoom_Out_25"));
    action_Normal_Size = new QAction(MainApp);
    action_Normal_Size->setObjectName(QStringLiteral("action_Normal_Size"));
    actionFit_to_Window = new QAction(MainApp);
    actionFit_to_Window->setObjectName(QStringLiteral("actionFit_to_Window"));
    action_Select_Area = new QAction(MainApp);
    action_Select_Area->setObjectName(QStringLiteral("action_Select_Area"));
    actionParame_ters = new QAction(MainApp);
    actionParame_ters->setObjectName(QStringLiteral("actionParame_ters"));
    action_Correlate = new QAction(MainApp);
    action_Correlate->setObjectName(QStringLiteral("action_Correlate"));
    action_About = new QAction(MainApp);
    action_About->setObjectName(QStringLiteral("action_About"));
    action_Save_Analysis = new QAction(MainApp);
    action_Save_Analysis->setObjectName(QStringLiteral("action_Save_Analysis"));
    actiong = new QAction(MainApp);
    actiong->setObjectName(QStringLiteral("actiong"));
    centralwidget = new QWidget(MainApp);
    centralwidget->setObjectName(QStringLiteral("centralwidget"));
    scrollArea_all_images = new QScrollArea(centralwidget);
    scrollArea_all_images->setObjectName(
        QStringLiteral("scrollArea_all_images"));
    scrollArea_all_images->setGeometry(QRect(10, 30, 161, 651));
    scrollArea_all_images->setWidgetResizable(true);
    scrollAreaWidgetContents = new QWidget();
    scrollAreaWidgetContents->setObjectName(
        QStringLiteral("scrollAreaWidgetContents"));
    scrollAreaWidgetContents->setGeometry(QRect(0, 0, 159, 649));
    scrollArea_all_images->setWidget(scrollAreaWidgetContents);
    label_correlation_model = new QLabel(centralwidget);
    label_correlation_model->setObjectName(
        QStringLiteral("label_correlation_model"));
    label_correlation_model->setGeometry(QRect(1300, 330, 131, 21));
    comboBox_correlation_model = new QComboBox(centralwidget);
    comboBox_correlation_model->setObjectName(
        QStringLiteral("comboBox_correlation_model"));
    comboBox_correlation_model->setGeometry(QRect(1300, 350, 131, 25));
    comboBox_interpolation_model = new QComboBox(centralwidget);
    comboBox_interpolation_model->setObjectName(
        QStringLiteral("comboBox_interpolation_model"));
    comboBox_interpolation_model->setGeometry(QRect(1300, 550, 131, 25));
    label_interpolation_model = new QLabel(centralwidget);
    label_interpolation_model->setObjectName(
        QStringLiteral("label_interpolation_model"));
    label_interpolation_model->setGeometry(QRect(1300, 530, 141, 21));
    QTabWidget_domain = new QTabWidget(centralwidget);
    QTabWidget_domain->setObjectName(QStringLiteral("QTabWidget_domain"));
    QTabWidget_domain->setGeometry(QRect(1300, 40, 241, 281));
    QTabWidget_domain->setStyleSheet(QStringLiteral(""));
    Rectangular = new QWidget();
    Rectangular->setObjectName(QStringLiteral("Rectangular"));
    label_domain_2 = new QLabel(Rectangular);
    label_domain_2->setObjectName(QStringLiteral("label_domain_2"));
    label_domain_2->setGeometry(QRect(20, 30, 81, 21));
    spinBox_rect_horizontal = new QSpinBox(Rectangular);
    spinBox_rect_horizontal->setObjectName(
        QStringLiteral("spinBox_rect_horizontal"));
    spinBox_rect_horizontal->setGeometry(QRect(20, 50, 81, 26));
    spinBox_rect_horizontal->setMinimum(1);
    label_domain_3 = new QLabel(Rectangular);
    label_domain_3->setObjectName(QStringLiteral("label_domain_3"));
    label_domain_3->setGeometry(QRect(30, 10, 91, 21));
    spinBox_rect_vertical = new QSpinBox(Rectangular);
    spinBox_rect_vertical->setObjectName(
        QStringLiteral("spinBox_rect_vertical"));
    spinBox_rect_vertical->setGeometry(QRect(130, 50, 81, 26));
    spinBox_rect_vertical->setMinimum(1);
    label_domain_4 = new QLabel(Rectangular);
    label_domain_4->setObjectName(QStringLiteral("label_domain_4"));
    label_domain_4->setGeometry(QRect(130, 30, 81, 21));
    doubleSpinBox_rect_y0 = new QDoubleSpinBox(Rectangular);
    doubleSpinBox_rect_y0->setObjectName(
        QStringLiteral("doubleSpinBox_rect_y0"));
    doubleSpinBox_rect_y0->setGeometry(QRect(130, 130, 81, 26));
    doubleSpinBox_rect_y0->setMaximum(99999);
    label_domain_8 = new QLabel(Rectangular);
    label_domain_8->setObjectName(QStringLiteral("label_domain_8"));
    label_domain_8->setGeometry(QRect(20, 110, 81, 21));
    doubleSpinBox_rect_x0 = new QDoubleSpinBox(Rectangular);
    doubleSpinBox_rect_x0->setObjectName(
        QStringLiteral("doubleSpinBox_rect_x0"));
    doubleSpinBox_rect_x0->setGeometry(QRect(20, 130, 81, 26));
    doubleSpinBox_rect_x0->setMaximum(99999);
    label_domain_9 = new QLabel(Rectangular);
    label_domain_9->setObjectName(QStringLiteral("label_domain_9"));
    label_domain_9->setGeometry(QRect(130, 110, 81, 21));
    doubleSpinBox_rect_xend = new QDoubleSpinBox(Rectangular);
    doubleSpinBox_rect_xend->setObjectName(
        QStringLiteral("doubleSpinBox_rect_xend"));
    doubleSpinBox_rect_xend->setGeometry(QRect(20, 210, 81, 26));
    doubleSpinBox_rect_xend->setMaximum(99999);
    doubleSpinBox_rect_yend = new QDoubleSpinBox(Rectangular);
    doubleSpinBox_rect_yend->setObjectName(
        QStringLiteral("doubleSpinBox_rect_yend"));
    doubleSpinBox_rect_yend->setGeometry(QRect(130, 210, 81, 26));
    doubleSpinBox_rect_yend->setMaximum(99999);
    label_domain_10 = new QLabel(Rectangular);
    label_domain_10->setObjectName(QStringLiteral("label_domain_10"));
    label_domain_10->setGeometry(QRect(130, 190, 91, 21));
    label_domain_11 = new QLabel(Rectangular);
    label_domain_11->setObjectName(QStringLiteral("label_domain_11"));
    label_domain_11->setGeometry(QRect(20, 190, 91, 21));
    QTabWidget_domain->addTab(Rectangular, QString());
    Annular = new QWidget();
    Annular->setObjectName(QStringLiteral("Annular"));
    spinBox_ann_radial = new QSpinBox(Annular);
    spinBox_ann_radial->setObjectName(QStringLiteral("spinBox_ann_radial"));
    spinBox_ann_radial->setGeometry(QRect(20, 50, 81, 26));
    spinBox_ann_radial->setMinimum(1);
    label_domain_6 = new QLabel(Annular);
    label_domain_6->setObjectName(QStringLiteral("label_domain_6"));
    label_domain_6->setGeometry(QRect(30, 10, 91, 21));
    label_domain_7 = new QLabel(Annular);
    label_domain_7->setObjectName(QStringLiteral("label_domain_7"));
    label_domain_7->setGeometry(QRect(130, 30, 81, 21));
    spinBox_ann_angular = new QSpinBox(Annular);
    spinBox_ann_angular->setObjectName(QStringLiteral("spinBox_ann_angular"));
    spinBox_ann_angular->setGeometry(QRect(130, 50, 81, 26));
    spinBox_ann_angular->setMinimum(1);
    label_domain_5 = new QLabel(Annular);
    label_domain_5->setObjectName(QStringLiteral("label_domain_5"));
    label_domain_5->setGeometry(QRect(20, 30, 81, 21));
    doubleSpinBox_ann_center_x = new QDoubleSpinBox(Annular);
    doubleSpinBox_ann_center_x->setObjectName(
        QStringLiteral("doubleSpinBox_ann_center_x"));
    doubleSpinBox_ann_center_x->setGeometry(QRect(20, 130, 81, 26));
    doubleSpinBox_ann_center_x->setMaximum(99999);
    doubleSpinBox_ann_center_y = new QDoubleSpinBox(Annular);
    doubleSpinBox_ann_center_y->setObjectName(
        QStringLiteral("doubleSpinBox_ann_center_y"));
    doubleSpinBox_ann_center_y->setGeometry(QRect(130, 130, 81, 26));
    doubleSpinBox_ann_center_y->setMaximum(99999);
    label_domain_22 = new QLabel(Annular);
    label_domain_22->setObjectName(QStringLiteral("label_domain_22"));
    label_domain_22->setGeometry(QRect(20, 190, 81, 21));
    doubleSpinBox_ann_ri = new QDoubleSpinBox(Annular);
    doubleSpinBox_ann_ri->setObjectName(QStringLiteral("doubleSpinBox_ann_ri"));
    doubleSpinBox_ann_ri->setGeometry(QRect(130, 210, 81, 26));
    doubleSpinBox_ann_ri->setMaximum(99999);
    label_domain_23 = new QLabel(Annular);
    label_domain_23->setObjectName(QStringLiteral("label_domain_23"));
    label_domain_23->setGeometry(QRect(130, 110, 81, 21));
    doubleSpinBox_ann_ro = new QDoubleSpinBox(Annular);
    doubleSpinBox_ann_ro->setObjectName(QStringLiteral("doubleSpinBox_ann_ro"));
    doubleSpinBox_ann_ro->setGeometry(QRect(20, 210, 81, 26));
    doubleSpinBox_ann_ro->setMaximum(99999);
    label_domain_24 = new QLabel(Annular);
    label_domain_24->setObjectName(QStringLiteral("label_domain_24"));
    label_domain_24->setGeometry(QRect(20, 110, 81, 21));
    label_domain_25 = new QLabel(Annular);
    label_domain_25->setObjectName(QStringLiteral("label_domain_25"));
    label_domain_25->setGeometry(QRect(130, 190, 81, 21));
    label_domain_27 = new QLabel(Annular);
    label_domain_27->setObjectName(QStringLiteral("label_domain_27"));
    label_domain_27->setGeometry(QRect(30, 90, 81, 21));
    label_domain_29 = new QLabel(Annular);
    label_domain_29->setObjectName(QStringLiteral("label_domain_29"));
    label_domain_29->setGeometry(QRect(30, 170, 81, 21));
    QTabWidget_domain->addTab(Annular, QString());
    Blob = new QWidget();
    Blob->setObjectName(QStringLiteral("Blob"));
    label_domain_30 = new QLabel(Blob);
    label_domain_30->setObjectName(QStringLiteral("label_domain_30"));
    label_domain_30->setGeometry(QRect(30, 170, 81, 21));
    label_domain_28 = new QLabel(Blob);
    label_domain_28->setObjectName(QStringLiteral("label_domain_28"));
    label_domain_28->setGeometry(QRect(30, 90, 81, 21));
    label_domain_26 = new QLabel(Blob);
    label_domain_26->setObjectName(QStringLiteral("label_domain_26"));
    label_domain_26->setGeometry(QRect(130, 110, 81, 21));
    label_domain_31 = new QLabel(Blob);
    label_domain_31->setObjectName(QStringLiteral("label_domain_31"));
    label_domain_31->setGeometry(QRect(130, 190, 81, 21));
    doubleSpinBox_blob_center_x = new QDoubleSpinBox(Blob);
    doubleSpinBox_blob_center_x->setObjectName(
        QStringLiteral("doubleSpinBox_blob_center_x"));
    doubleSpinBox_blob_center_x->setGeometry(QRect(20, 130, 81, 26));
    doubleSpinBox_blob_center_x->setMaximum(99999);
    label_domain_32 = new QLabel(Blob);
    label_domain_32->setObjectName(QStringLiteral("label_domain_32"));
    label_domain_32->setGeometry(QRect(20, 190, 81, 21));
    label_domain_33 = new QLabel(Blob);
    label_domain_33->setObjectName(QStringLiteral("label_domain_33"));
    label_domain_33->setGeometry(QRect(20, 110, 81, 21));
    doubleSpinBox_blob_center_y = new QDoubleSpinBox(Blob);
    doubleSpinBox_blob_center_y->setObjectName(
        QStringLiteral("doubleSpinBox_blob_center_y"));
    doubleSpinBox_blob_center_y->setGeometry(QRect(130, 130, 81, 26));
    doubleSpinBox_blob_center_y->setMaximum(99999);
    doubleSpinBox_blob_x_scale = new QDoubleSpinBox(Blob);
    doubleSpinBox_blob_x_scale->setObjectName(
        QStringLiteral("doubleSpinBox_blob_x_scale"));
    doubleSpinBox_blob_x_scale->setGeometry(QRect(18, 210, 81, 26));
    doubleSpinBox_blob_x_scale->setMinimum(0.1);
    doubleSpinBox_blob_x_scale->setValue(1);
    doubleSpinBox_blob_y_scale = new QDoubleSpinBox(Blob);
    doubleSpinBox_blob_y_scale->setObjectName(
        QStringLiteral("doubleSpinBox_blob_y_scale"));
    doubleSpinBox_blob_y_scale->setGeometry(QRect(130, 210, 81, 26));
    doubleSpinBox_blob_y_scale->setMinimum(0.1);
    doubleSpinBox_blob_y_scale->setValue(1);
    QTabWidget_domain->addTab(Blob, QString());
    label_domain = new QLabel(centralwidget);
    label_domain->setObjectName(QStringLiteral("label_domain"));
    label_domain->setGeometry(QRect(1300, 20, 179, 21));
    label_Initial_Guess = new QLabel(centralwidget);
    label_Initial_Guess->setObjectName(QStringLiteral("label_Initial_Guess"));
    label_Initial_Guess->setGeometry(QRect(1300, 380, 91, 21));
    comboBox_Initial_Guess = new QComboBox(centralwidget);
    comboBox_Initial_Guess->setObjectName(
        QStringLiteral("comboBox_Initial_Guess"));
    comboBox_Initial_Guess->setGeometry(QRect(1300, 400, 131, 25));
    comboBox_update = new QComboBox(centralwidget);
    comboBox_update->setObjectName(QStringLiteral("comboBox_update"));
    comboBox_update->setGeometry(QRect(1300, 450, 131, 25));
    label_Initial_Guess_2 = new QLabel(centralwidget);
    label_Initial_Guess_2->setObjectName(
        QStringLiteral("label_Initial_Guess_2"));
    label_Initial_Guess_2->setGeometry(QRect(1300, 430, 91, 21));
    label_Initial_Guess_3 = new QLabel(centralwidget);
    label_Initial_Guess_3->setObjectName(
        QStringLiteral("label_Initial_Guess_3"));
    label_Initial_Guess_3->setGeometry(QRect(1450, 330, 101, 21));
    label_Initial_Guess_4 = new QLabel(centralwidget);
    label_Initial_Guess_4->setObjectName(
        QStringLiteral("label_Initial_Guess_4"));
    label_Initial_Guess_4->setGeometry(QRect(1300, 480, 91, 21));
    comboBox_color = new QComboBox(centralwidget);
    comboBox_color->setObjectName(QStringLiteral("comboBox_color"));
    comboBox_color->setGeometry(QRect(1300, 500, 131, 25));
    spinBox_max_iters = new QSpinBox(centralwidget);
    spinBox_max_iters->setObjectName(QStringLiteral("spinBox_max_iters"));
    spinBox_max_iters->setGeometry(QRect(1450, 350, 81, 26));
    spinBox_max_iters->setMaximum(9999);
    label_Initial_Guess_5 = new QLabel(centralwidget);
    label_Initial_Guess_5->setObjectName(
        QStringLiteral("label_Initial_Guess_5"));
    label_Initial_Guess_5->setGeometry(QRect(1450, 380, 91, 21));
    doubleSpinBox_precision = new QDoubleSpinBox(centralwidget);
    doubleSpinBox_precision->setObjectName(
        QStringLiteral("doubleSpinBox_precision"));
    doubleSpinBox_precision->setGeometry(QRect(1450, 400, 81, 26));
    doubleSpinBox_precision->setDecimals(5);
    spinBox_py_start = new QSpinBox(centralwidget);
    spinBox_py_start->setObjectName(QStringLiteral("spinBox_py_start"));
    spinBox_py_start->setGeometry(QRect(1450, 450, 81, 26));
    label_Initial_Guess_6 = new QLabel(centralwidget);
    label_Initial_Guess_6->setObjectName(
        QStringLiteral("label_Initial_Guess_6"));
    label_Initial_Guess_6->setGeometry(QRect(1450, 430, 91, 21));
    label_stop_pyram = new QLabel(centralwidget);
    label_stop_pyram->setObjectName(QStringLiteral("label_stop_pyram"));
    label_stop_pyram->setGeometry(QRect(1450, 530, 101, 21));
    spinBox_py_stop = new QSpinBox(centralwidget);
    spinBox_py_stop->setObjectName(QStringLiteral("spinBox_py_stop"));
    spinBox_py_stop->setGeometry(QRect(1450, 550, 81, 26));
    spinBox_py_step = new QSpinBox(centralwidget);
    spinBox_py_step->setObjectName(QStringLiteral("spinBox_py_step"));
    spinBox_py_step->setGeometry(QRect(1450, 500, 81, 26));
    label_Initial_Guess_8 = new QLabel(centralwidget);
    label_Initial_Guess_8->setObjectName(
        QStringLiteral("label_Initial_Guess_8"));
    label_Initial_Guess_8->setGeometry(QRect(1450, 480, 101, 21));
    pushButton_show_sets = new QPushButton(centralwidget);
    pushButton_show_sets->setObjectName(QStringLiteral("pushButton_show_sets"));
    pushButton_show_sets->setGeometry(QRect(1420, 620, 111, 61));
    pushButton_correlate = new QPushButton(centralwidget);
    pushButton_correlate->setObjectName(QStringLiteral("pushButton_correlate"));
    pushButton_correlate->setGeometry(QRect(1330, 620, 81, 61));
    label_undeformed_name = new QLabel(centralwidget);
    label_undeformed_name->setObjectName(
        QStringLiteral("label_undeformed_name"));
    label_undeformed_name->setGeometry(QRect(190, 620, 541, 31));
    label_undeformed_name->setAlignment(Qt::AlignLeading | Qt::AlignLeft |
                                        Qt::AlignTop);
    label_deformed_name = new QLabel(centralwidget);
    label_deformed_name->setObjectName(QStringLiteral("label_deformed_name"));
    label_deformed_name->setGeometry(QRect(750, 620, 541, 31));
    label_deformed_name->setAlignment(Qt::AlignLeading | Qt::AlignLeft |
                                      Qt::AlignTop);
    progressBar = new QProgressBar(centralwidget);
    progressBar->setObjectName(QStringLiteral("progressBar"));
    progressBar->setGeometry(QRect(190, 660, 1101, 20));
    progressBar->setValue(24);
    scrollArea_und_image = new QScrollArea(centralwidget);
    scrollArea_und_image->setObjectName(QStringLiteral("scrollArea_und_image"));
    scrollArea_und_image->setGeometry(QRect(190, 30, 541, 581));
    scrollArea_und_image->setWidgetResizable(true);
    scrollAreaWidgetContents_2 = new QWidget();
    scrollAreaWidgetContents_2->setObjectName(
        QStringLiteral("scrollAreaWidgetContents_2"));
    scrollAreaWidgetContents_2->setGeometry(QRect(0, 0, 539, 579));
    scrollArea_und_image->setWidget(scrollAreaWidgetContents_2);
    scrollArea_def_image = new QScrollArea(centralwidget);
    scrollArea_def_image->setObjectName(QStringLiteral("scrollArea_def_image"));
    scrollArea_def_image->setGeometry(QRect(750, 30, 541, 581));
    scrollArea_def_image->setWidgetResizable(true);
    scrollAreaWidgetContents_3 = new QWidget();
    scrollAreaWidgetContents_3->setObjectName(
        QStringLiteral("scrollAreaWidgetContents_3"));
    scrollAreaWidgetContents_3->setGeometry(QRect(0, 0, 539, 579));
    scrollArea_def_image->setWidget(scrollAreaWidgetContents_3);
    scrollArea_und_set = new QScrollArea(centralwidget);
    scrollArea_und_set->setObjectName(QStringLiteral("scrollArea_und_set"));
    scrollArea_und_set->setGeometry(QRect(190, 690, 261, 271));
    scrollArea_und_set->setWidgetResizable(true);
    scrollAreaWidgetContents_4 = new QWidget();
    scrollAreaWidgetContents_4->setObjectName(
        QStringLiteral("scrollAreaWidgetContents_4"));
    scrollAreaWidgetContents_4->setGeometry(QRect(0, 0, 259, 269));
    scrollArea_und_set->setWidget(scrollAreaWidgetContents_4);
    scrollArea_def_set = new QScrollArea(centralwidget);
    scrollArea_def_set->setObjectName(QStringLiteral("scrollArea_def_set"));
    scrollArea_def_set->setGeometry(QRect(750, 690, 261, 271));
    scrollArea_def_set->setWidgetResizable(true);
    scrollAreaWidgetContents_5 = new QWidget();
    scrollAreaWidgetContents_5->setObjectName(
        QStringLiteral("scrollAreaWidgetContents_5"));
    scrollAreaWidgetContents_5->setGeometry(QRect(0, 0, 259, 269));
    scrollArea_def_set->setWidget(scrollAreaWidgetContents_5);
    scrollArea_und_info = new QScrollArea(centralwidget);
    scrollArea_und_info->setObjectName(QStringLiteral("scrollArea_und_info"));
    scrollArea_und_info->setGeometry(QRect(470, 690, 261, 271));
    scrollArea_und_info->setWidgetResizable(true);
    scrollAreaWidgetContents_6 = new QWidget();
    scrollAreaWidgetContents_6->setObjectName(
        QStringLiteral("scrollAreaWidgetContents_6"));
    scrollAreaWidgetContents_6->setGeometry(QRect(0, 0, 259, 269));
    scrollArea_und_info->setWidget(scrollAreaWidgetContents_6);
    scrollArea_def_info = new QScrollArea(centralwidget);
    scrollArea_def_info->setObjectName(QStringLiteral("scrollArea_def_info"));
    scrollArea_def_info->setGeometry(QRect(1030, 690, 261, 271));
    scrollArea_def_info->setWidgetResizable(true);
    scrollAreaWidgetContents_7 = new QWidget();
    scrollAreaWidgetContents_7->setObjectName(
        QStringLiteral("scrollAreaWidgetContents_7"));
    scrollAreaWidgetContents_7->setGeometry(QRect(0, 0, 259, 269));
    scrollArea_def_info->setWidget(scrollAreaWidgetContents_7);
    label_results_display = new QLabel(centralwidget);
    label_results_display->setObjectName(
        QStringLiteral("label_results_display"));
    label_results_display->setGeometry(QRect(750, 10, 541, 17));
    label_best_rotation_value = new QLabel(centralwidget);
    label_best_rotation_value->setObjectName(
        QStringLiteral("label_best_rotation_value"));
    label_best_rotation_value->setGeometry(QRect(460, 10, 271, 20));
    label_best_rotation = new QLabel(centralwidget);
    label_best_rotation->setObjectName(QStringLiteral("label_best_rotation"));
    label_best_rotation->setGeometry(QRect(200, 10, 251, 20));
    MainApp->setCentralWidget(centralwidget);
    menubar = new QMenuBar(MainApp);
    menubar->setObjectName(QStringLiteral("menubar"));
    menubar->setGeometry(QRect(0, 0, 1551, 22));
    menu_File = new QMenu(menubar);
    menu_File->setObjectName(QStringLiteral("menu_File"));
    menu_View = new QMenu(menubar);
    menu_View->setObjectName(QStringLiteral("menu_View"));
    menu_Tools = new QMenu(menubar);
    menu_Tools->setObjectName(QStringLiteral("menu_Tools"));
    menu_Help = new QMenu(menubar);
    menu_Help->setObjectName(QStringLiteral("menu_Help"));
    MainApp->setMenuBar(menubar);
    statusbar = new QStatusBar(MainApp);
    statusbar->setObjectName(QStringLiteral("statusbar"));
    MainApp->setStatusBar(statusbar);

    menubar->addAction(menu_File->menuAction());
    menubar->addAction(menu_View->menuAction());
    menubar->addAction(menu_Tools->menuAction());
    menubar->addAction(menu_Help->menuAction());
    menu_File->addAction(action_Open);
    menu_File->addAction(action_Print);
    menu_File->addAction(action_Save_Analysis);
    menu_File->addSeparator();
    menu_File->addAction(action_Exit);
    menu_View->addAction(actionZoom_in_25);
    menu_View->addAction(actionZoom_Out_25);
    menu_View->addAction(action_Normal_Size);
    menu_View->addAction(actionFit_to_Window);
    menu_Tools->addAction(actionParame_ters);
    menu_Help->addAction(action_About);

    retranslateUi(MainApp);

    QTabWidget_domain->setCurrentIndex(2);

    QMetaObject::connectSlotsByName(MainApp);
  } // setupUi

  void retranslateUi(QMainWindow *MainApp) {
    MainApp->setWindowTitle(
        QApplication::translate("MainApp", "MainWindow", 0));
    action_Open->setText(QApplication::translate("MainApp", "&Open", 0));
    action_Print->setText(QApplication::translate("MainApp", "&Print", 0));
    action_Exit->setText(QApplication::translate("MainApp", "Exit", 0));
    actionZoom_in_25->setText(
        QApplication::translate("MainApp", "Zoom &In (25%)", 0));
    actionZoom_Out_25->setText(
        QApplication::translate("MainApp", "Zoom &Out (25%)", 0));
    action_Normal_Size->setText(
        QApplication::translate("MainApp", "&Normal Size", 0));
    actionFit_to_Window->setText(
        QApplication::translate("MainApp", "&Fit to Window", 0));
    action_Select_Area->setText(
        QApplication::translate("MainApp", "Select &Area", 0));
    actionParame_ters->setText(
        QApplication::translate("MainApp", "Parame&ters", 0));
    action_Correlate->setText(
        QApplication::translate("MainApp", "&Correlate", 0));
    action_About->setText(QApplication::translate("MainApp", "A&bout", 0));
    action_Save_Analysis->setText(
        QApplication::translate("MainApp", "&Save Report", 0));
    actiong->setText(QApplication::translate("MainApp", "g", 0));
    label_correlation_model->setText(
        QApplication::translate("MainApp", "Correlation Model", 0));
    label_interpolation_model->setText(
        QApplication::translate("MainApp", "Interpolation Model", 0));
    label_domain_2->setText(
        QApplication::translate("MainApp", "Horizontal", 0));
    label_domain_3->setText(
        QApplication::translate("MainApp", "Subdibisions", 0));
    label_domain_4->setText(QApplication::translate("MainApp", "Vertical", 0));
    label_domain_8->setText(
        QApplication::translate("MainApp", " X0 (pixel)", 0));
    label_domain_9->setText(
        QApplication::translate("MainApp", "Y0 (pixel)", 0));
    label_domain_10->setText(
        QApplication::translate("MainApp", "Yend (pixel)", 0));
    label_domain_11->setText(
        QApplication::translate("MainApp", "Xend (pixel)", 0));
    QTabWidget_domain->setTabText(
        QTabWidget_domain->indexOf(Rectangular),
        QApplication::translate("MainApp", "Rectangular", 0));
    label_domain_6->setText(
        QApplication::translate("MainApp", "Subdibisions", 0));
    label_domain_7->setText(QApplication::translate("MainApp", "Angular", 0));
    label_domain_5->setText(QApplication::translate("MainApp", "Radial", 0));
    label_domain_22->setText(
        QApplication::translate("MainApp", "Ro (pixel)", 0));
    label_domain_23->setText(
        QApplication::translate("MainApp", "Y0 (pixel)", 0));
    label_domain_24->setText(
        QApplication::translate("MainApp", " X0 (pixel)", 0));
    label_domain_25->setText(
        QApplication::translate("MainApp", " Ri (pixel)", 0));
    label_domain_27->setText(QApplication::translate("MainApp", " Center", 0));
    label_domain_29->setText(QApplication::translate("MainApp", "Radius", 0));
    QTabWidget_domain->setTabText(
        QTabWidget_domain->indexOf(Annular),
        QApplication::translate("MainApp", "Annular", 0));
    label_domain_30->setText(QApplication::translate("MainApp", "Scale", 0));
    label_domain_28->setText(QApplication::translate("MainApp", " Center", 0));
    label_domain_26->setText(
        QApplication::translate("MainApp", "Y0 (pixel)", 0));
    label_domain_31->setText(
        QApplication::translate("MainApp", "y - Scale", 0));
    label_domain_32->setText(
        QApplication::translate("MainApp", "x - Scale", 0));
    label_domain_33->setText(
        QApplication::translate("MainApp", " X0 (pixel)", 0));
    QTabWidget_domain->setTabText(
        QTabWidget_domain->indexOf(Blob),
        QApplication::translate("MainApp", "Blob", 0));
    label_domain->setText(QApplication::translate("MainApp", "Domain Type", 0));
    label_Initial_Guess->setText(
        QApplication::translate("MainApp", "Initial Guess", 0));
    label_Initial_Guess_2->setText(
        QApplication::translate("MainApp", "Update", 0));
    label_Initial_Guess_3->setText(
        QApplication::translate("MainApp", "Max Iters", 0));
    label_Initial_Guess_4->setText(
        QApplication::translate("MainApp", "Color", 0));
    label_Initial_Guess_5->setText(
        QApplication::translate("MainApp", "Precision", 0));
    label_Initial_Guess_6->setText(
        QApplication::translate("MainApp", "Start Pyramid", 0));
    label_stop_pyram->setText(
        QApplication::translate("MainApp", "Stop Pyramid", 0));
    label_Initial_Guess_8->setText(
        QApplication::translate("MainApp", "Pyramid  Step", 0));
    pushButton_show_sets->setText(
        QApplication::translate("MainApp", "Show Sets", 0));
    pushButton_correlate->setText(
        QApplication::translate("MainApp", "Correlate", 0));
    label_undeformed_name->setText(
        QApplication::translate("MainApp", "Undeformed Image", 0));
    label_deformed_name->setText(
        QApplication::translate("MainApp", "Deformed Image", 0));
    label_results_display->setText(QString());
    label_best_rotation_value->setText(QString());
    label_best_rotation->setText(
        QApplication::translate("MainApp", "Best Rotation(deg)", 0));
    menu_File->setTitle(QApplication::translate("MainApp", "&File", 0));
    menu_View->setTitle(QApplication::translate("MainApp", "&View", 0));
    menu_Tools->setTitle(QApplication::translate("MainApp", "&Tools", 0));
    menu_Help->setTitle(QApplication::translate("MainApp", "&Help", 0));
  } // retranslateUi
};

namespace Ui {
class MainApp : public Ui_MainApp {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINAPP_H
