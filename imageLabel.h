/****************************************************************************
**
**  This class was developed by Javier Gonzalez on 2018 from
**  QT examples
**
**
****************************************************************************/
#ifndef IMAGELABEL_H
#define IMAGELABEL_H

#include "interpolation_class.hpp"
#include "model_class.hpp"
#include "parameters.hpp"

#include <assert.h>
#include <iostream>

#include <QLabel>
#include <QPaintEvent>
#include <QPainter>
#include <QPixmap>
#include <QtWidgets>

#ifndef QT_NO_PRINTER
#include <QPrintDialog>
#include <QPrinter>
#endif

QT_BEGIN_NAMESPACE
class QLabel;
QT_END_NAMESPACE

class ImageLabel : public QLabel {
  Q_OBJECT

public:
  std::string name;
  ImageLabel(QWidget *parent);
  bool selecting;
  bool mirroring;
  bool correlating;
  bool selecting_ic;
  bool adjusting_domain;
  // undeformed geometry - need this to apply ic
  float right_x, right_y, left_x, left_y;
  float prev_left_x, prev_left_y;
  float prev_right_x, prev_right_y;
  float adjust_domain_right_x, adjust_domain_right_y;
  float adjust_domain_left_x, adjust_domain_left_y;
  float center_x, center_y;
  float prev_center_x, prev_center_y;
  float blob_scale_x, blob_scale_y;
  float inside_radius, outside_radius, ri_by_ro;
  float prev_inside_radius, prev_outside_radius;
  float blob_x_max, blob_x_min, blob_y_max, blob_y_min;
  v_points xy_blob;
  v_points prev_xy_blob;
  std::vector<v_points> inside_points;
  std::vector<v_points> contour_points;
  std::vector<bool> inside_points_error;
  std::vector<bool> contour_points_error;

  // deformed geometry
  float def_lo_right_x, def_lo_right_y, def_hi_left_x, def_hi_left_y;
  float def_hi_right_x, def_hi_right_y, def_lo_left_x, def_lo_left_y;
  float def_center_x, def_center_y;
  float def_ri_by_ro, def_q, def_outside_radius, def_inside_radius;
  float def_u, def_v, def_dudx, def_dudy, def_dvdx, def_dvdy;

  float realScale_x, realScale_y;
  float clickScale_x, clickScale_y;
  domainEnum domain;
  int h_subdivisions, v_subdivisions, r_subdivisions, a_subdivisions;
  fittingModelEnum model;
  float *model_parameters = nullptr;
  float prev_model_parameters[6];
  int ic_def_x, ic_def_y, ic_und_x, ic_und_y;
  bool suppress_selection_display;

private:
protected:
  void mousePressEvent(QMouseEvent *event);
  void mouseMoveEvent(QMouseEvent *event);
  void mouseReleaseEvent(QMouseEvent *event);
  void paintEvent(QPaintEvent *event);
  void drawOutline_rectangular(QPainter &painter);
  void drawOutline_annular(QPainter &painter);
  void drawOutline_blob(QPainter &painter);
  void paint_inside_points(QPainter &painter);
  void paint_contour_points(QPainter &painter);
  void applyModelRectangular();
  void applyModelAnnular();
  void applyModelBlob();

  UVUxUyVxVyInitialGuess UVUxUyVxVyIC;
  annularInitialGuess annularIC;
  bool error;

signals:
  void valueChanged_rectangularSelected(float center_x, float center_y,
                                        float right_x, float right_y,
                                        float left_x, float left_y,
                                        float clickScale_x, float clickScale_y);
  void valueChanged_annularSelected(float center_x, float center_y,
                                    float inside_radius, float outside_radius,
                                    float ri_by_ro, float clickScale_x,
                                    float clickScale_y);
  void valueChanged_blobSelected(v_points xy_blob, float clickScale_x,
                                 float clickScale_y);

  void valueChanged_selectingRectangular(float center_x_in, float center_y_in,
                                         float right_x_in, float right_y_in,
                                         float left_x_in, float left_y_in,
                                         float clickScale_x_in,
                                         float clickScale_y_in);
  void valueChanged_selectingAnnular(float center_x_in, float center_y_in,
                                     float r_inside_in, float r_outside_in,
                                     float ri_by_ro_in, float clickScale_x_in,
                                     float clickScale_y_in);
  void valueChanged_selectingBlob(float center_x_in, float center_y_in,
                                  v_points xy_blob_in, float clickScale_x_in,
                                  float clickScale_y_in);
  void update_mirror();
  void stop_correlation_display();
  void new_initial_conditions();

private slots:
  void scaleSquare();
  void mirrorSelectionRectangular(float center_x_in, float center_y_in,
                                  float right_x_in, float right_y_in,
                                  float left_x_in, float left_y_in,
                                  float clickScale_x_in, float clickScale_y_in);
  void mirrorSelectionAnnular(float center_x_in, float center_y_in,
                              float r_inside_in, float r_outside_in,
                              float ri_by_ro_in, float clickScale_x_in,
                              float clickScale_y_in);
  void mirrorSelectionBlob(float center_x_in, float center_y_in,
                           v_points xy_blob_in, float clickScale_x_in,
                           float clickScale_y_in);

  void GUIupdated();

  void set_inside_points(v_points inside_points_vector_in, bool error_in);
  void set_contour_points(v_points contour_points_vector_in, bool error_in);

public slots:
  void clear_inside_points();
  void clear_contour_points();
};

#endif // IMAGELABEL_H
