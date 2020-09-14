#ifndef SUBDI_H
#define SUBDI_H

#include "parameters.hpp"

// Qt includes
#include <QComboBox>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QWidget>

class Subdi : public QWidget {
  Q_OBJECT

public:
  explicit Subdi(QWidget *parent = 0);
  QLabel *colorLabel;
  QLabel *updateLabel;
  QLabel *blur_start_Label;
  QLabel *blur_step_Label;
  QLabel *blur_stop_Label;
  QLabel *max_iters_Label;
  QLabel *precision_Label;
  QLabel *update_images_Label;
  QLabel *preload_images_Label;
  QLabel *interpolation_model_Label;
  QLabel *deformation_descriptionLabel;
  QLabel *error_handlingModeLabel;
  QLabel *arrowLabel;
  QLabel *arrow_magnificationLabel;
  QLabel *plot_inside_pointsLabel;
  QLabel *plot_contour_pointsLabel;
  QLabel *reference_imageLabel;
  QLabel *processorLabel;
  QLabel *number_of_GPUs_threads_Label;

  QSpinBox *blur_start_Box;
  QSpinBox *blur_step_Box;
  QSpinBox *blur_stop_Box;
  QSpinBox *max_iters_Box;
  QSpinBox *number_of_GPUs_threads_Box;

  QDoubleSpinBox *arrow_magnificationBox;
  QDoubleSpinBox *precision_Box;

  QComboBox *colorBox;
  QComboBox *updateBox;
  QComboBox *interpolation_model_Box;
  QComboBox *deformation_descriptionBox;
  QComboBox *error_handlingModeBox;
  QComboBox *arrowBox;
  QComboBox *plot_inside_pointsBox;
  QComboBox *plot_contour_pointsBox;
  QComboBox *update_images_Box;
  QComboBox *preload_images_Box;
  QComboBox *reference_imageBox;
  QComboBox *processorBox;

  QPushButton *okButton;

  QFormLayout *leftlayout;
  QFormLayout *centerlayout;
  QFormLayout *rightlayout;
  QHBoxLayout *toplayout;
  QFormLayout *bottomlayout;
  QVBoxLayout *layout;

signals:
  void valueChanged_domain();

private slots:

protected:
};

#endif // SUBDI_H
