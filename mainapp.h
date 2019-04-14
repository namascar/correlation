#ifndef MAINAPP_H
#define MAINAPP_H


#ifndef QT_NO_PRINTER
#include <QPrinter>
#include <QPrintDialog>
#endif

#include <QFileDialog>
#include <QMainWindow>
#include <QtWidgets>
#include <QPixmap>
#include <QApplication>
#include <QLabel>
#include <QSizePolicy>
#include <QPainter>
#include <QPaintEvent>

#include "imageLabel.h"
#include "manager_class.h"
#include "subdi.h"
#include "parameters.hpp"
#include "ui_mainapp.h"

#include <sstream>
#include <fstream>
#include <iostream>

namespace Ui {
class MainApp;
}

class MainApp : public QMainWindow
{
    Q_OBJECT
    QThread managerThread;

#if CUDA_ENABLED
    CudaClass* cuda_manager;
#endif

public:
    explicit MainApp( QWidget* parent = 0 );
    ~MainApp();

private slots:
    void open                           ();
    void print                          ();
    void zoomIn                         ();
    void zoomOut                        ();
    void normalSize                     ();
    void fitToWindow                    ();
    void close_app                      ();
    void about                          ();
    void open_subdivisions              ();
    void close_subdivisions             ();
    void reverse_sets                   ();
    void correlate                      ();
    void enableRectangularCorrelate     ( float center_x_in , float center_y_in,
                                          float right_x_in  , float right_y_in , float left_x_in, float left_y_in ,
                                          float scale_x_in  , float scale_y_in );

    void enableAnnularCorrelate         ( float center_x_in , float center_y_in ,
                                          float inside_radius_in, float outside_radius_in, float ri_by_ro_in,
                                          float scale_x_in  , float scale_y_in);

    void enableBlobCorrelate            ( v_points xy_blob_in, float scale_x_in, float scale_y_in );
    void correlation_done               ( bool error );
    void makeAnalysisReport             ();
    void saveAnalysisReport             ();
    void change_correlation_model       ( int model_in );
    void change_preload_images          ( int preload_images_in );
    void change_color                   ( int color_in );
    void change_Initial_Guess           ( int guess_in );
    void change_interpolation_model     ( int interpolation_model_in );
    void change_update                  ( int update_in );
    void change_domain                  ( int domain_in );
    void change_max_iters               ( int max_iters_in );
    void change_precision               ( double precision_in );

    void change_star_pyramid            ( int star_py_in );
    void change_step_pyramid            ( int step_py_in );
    void change_stop_pyramid            ( int stop_py_in );

    void change_deformation_description ( int deformation_description_in );
    void change_processor               ( int processor_in );
    void change_devices_used            ( int devicesUsed_in );

    void change_rect_horizontal         ( int h_subdi_in );
    void change_rect_vertical           ( int v_subdi_in );
    void change_rect_x0                 ( double x0_in );
    void change_rect_y0                 ( double y0_in );
    void change_rect_xend               ( double xend_in );
    void change_rect_yend               ( double yend_in );

    void change_ann_radial              ( int r_subdi_in );
    void change_ann_angular             ( int a_subdi_in );
    void change_ann_center_x            ( double xc_in );
    void change_ann_center_y            ( double yc_in );
    void change_ann_ro                  ( double ro_in );
    void change_ann_ri                  ( double ri_in );

    void change_blob_center_x           ( double xc_in );
    void change_blob_center_y           ( double yc_in );
    void change_blob_x_scale            ( double xscale_in );
    void change_blob_y_scale            ( double yscale_in );

    bool display_two_images             ( QString und_file_QString_in, QString def_file_QString_in );
    bool display_undeformed_image       ( QString und_file_QString_in );
    bool display_deformed_image         ( QString def_file_QString_in );
    void stop_correlate                 ();
    void stop_correlation_display       ();
    void updateResultDisplay            ();
    void updateResultDisplay            ( float angle, float *result_parameters );
    void refresh_images                 ();

private:
    void createActions                  ();
    void scaleImage                     ( double factor );
    void updateActions                  ();
    void adjustScrollBar                ( QScrollBar* scrollBar , double factor );
    void selectInitialGuess             ();
    void updateDomains                  ();
    void updateRectangularCenter        ();
    void clearLabels                    ();
    void initialize_domains             ();
    void updateGpuPyramids              ();
    void loadNxtGpuImages               ();

    Ui::MainApp*                        ui;

    ImageLabel                         *und_imageLabel;
    ImageLabel                         *def_imageLabel;
    managerClass                       *manager;
    QPainter                           *painter;
    QStringList                         fileNames;
    std::ofstream                       report;
    QString                             reportFile;
    QString                             last_directory_name;
    QString                             first_und_file_QString;
    QString                             first_def_file_QString;
    QString                             first_nxt_file_QString;
    QString                             current_und_file_QString;
    QString                             current_def_file_QString;

#ifndef QT_NO_PRINTER
    QPrinter                            printer;
#endif

    QAction                            *openAct;
    QAction                            *subdivisionsAct;

    QPixmap                             und_pixmap;
    QPixmap                             def_pixmap;

    Subdi                              *subdivisions;

    bool                                drawing;
    bool                                sets;

    colorEnum                           color_mode_iv;
    int                                 number_of_colors;
    initialGuessEnum                    initial_guess_iv;
    updateEnum                          update_iv;
    domainEnum                          domain_iv;
    deformationDescriptionEnum          deformation_description_iv;
    errorHandlingModeEnum               error_handlingMode_iv;
    referenceImageEnum                  reference_image_iv;
    QStandardItemModel                 *comboBox_reference_image;
    QStandardItem                      *item_first_ref_image;
    bool                                arrow_iv;
    bool                                update_images_iv;
    bool                                preload_images_iv;
    double                              arrow_magnification_iv;
    bool                                plot_inside_points_iv;
    bool                                plot_contour_points_iv;
    processorEnum                       processor_iv;

    fittingModelEnum                    model_iv;
    QStandardItemModel                 *comboBox_correlation_model;

    interpolationModelEnum              interpolation_iv;

    float                               scale_x;
    float                               scale_y;

    rectangularDomainStruct             rectangularDomain;
    annularDomainStruct                 annularDomain;
    blobDomainStruct                    blobDomain;

    int                                 py_start;
    int                                 py_step;
    int                                 py_stop;
    float                               precision;
    int                                 max_iters;
    int                                 number_of_model_parameters;
    int                                 deviceCount;
    int                                 devicesUsed;
    int                                 number_of_threads;
    float                              *initial_guess = nullptr;
    float                             **initial_guess_archive = nullptr;

    QStringList                         icList;
    QStringList                         colorList;
    QStringList                         updateList;
    QStringList                         fit_modelList;
    QStringList                         interpolation_modelList;
    QStringList                         domainList;
    QStringList                         deformation_descriptionList;
    QStringList                         error_handlingModeList;
    QStringList                         no_yesList;
    QStringList                         reference_imageList;
    QStringList                         processorList;

signals:
    void                                start_correlation();
    void                                valueChanged_scaleSelection();
    void                                valueChanged_domain();
};

#endif // MAINAPP_H
