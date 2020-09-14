/****************************************************************************
**
**  This software was developed by Javier Gonzalez on 2018
**
**  Creates the parameter window.
**
****************************************************************************/

#include "subdi.h"

Subdi::Subdi(QWidget *parent) : QWidget(parent) {
  setWindowTitle("Enter Parameters");
  resize(800, 100);
  move(1300, 70);

  // left layout
  max_iters_Label = new QLabel(tr("Max. Iterations"));
  max_iters_Box = new QSpinBox();
  max_iters_Box->setMaximum(999);

  deformation_descriptionLabel = new QLabel(tr("Deformation description"));
  deformation_descriptionBox = new QComboBox();

  reference_imageLabel = new QLabel(tr("Reference Image"));
  reference_imageBox = new QComboBox();

  processorLabel = new QLabel(tr("Target Processor"));
  processorBox = new QComboBox();

  number_of_GPUs_threads_Label = new QLabel(tr("Number of threads"));
  number_of_GPUs_threads_Box = new QSpinBox();

  QFormLayout *leftlayout = new QFormLayout;
  leftlayout->addRow(max_iters_Label);
  leftlayout->addRow(max_iters_Box);
  leftlayout->addRow(deformation_descriptionLabel);
  leftlayout->addRow(deformation_descriptionBox);
  leftlayout->addRow(reference_imageLabel);
  leftlayout->addRow(reference_imageBox);
  leftlayout->addRow(processorLabel);
  leftlayout->addRow(processorBox);
  leftlayout->addRow(number_of_GPUs_threads_Label);
  leftlayout->addRow(number_of_GPUs_threads_Box);

  // center layout
  updateLabel = new QLabel(tr("Update Type"));
  updateBox = new QComboBox();

  blur_start_Label = new QLabel(tr("Start Pyramid"));
  blur_start_Box = new QSpinBox();

  blur_step_Label = new QLabel(tr("Step Pyramid"));
  blur_step_Box = new QSpinBox();

  blur_stop_Label = new QLabel(tr("Stop Pyramid"));
  blur_stop_Box = new QSpinBox();

  interpolation_model_Label = new QLabel(tr("Interpolation Model"));
  interpolation_model_Box = new QComboBox();

  QFormLayout *centerlayout = new QFormLayout;
  centerlayout->addRow(updateLabel);
  centerlayout->addRow(updateBox);
  centerlayout->addRow(blur_start_Label);
  centerlayout->addRow(blur_start_Box);
  centerlayout->addRow(blur_step_Label);
  centerlayout->addRow(blur_step_Box);
  centerlayout->addRow(blur_stop_Label);
  centerlayout->addRow(blur_stop_Box);
  centerlayout->addRow(interpolation_model_Label);
  centerlayout->addRow(interpolation_model_Box);

  // right layout
  colorLabel = new QLabel(tr("Color"));
  colorBox = new QComboBox();

  precision_Label = new QLabel(tr("Precision"));
  precision_Box = new QDoubleSpinBox();
  precision_Box->setDecimals(6);

  update_images_Label = new QLabel(tr("Update Images"));
  update_images_Box = new QComboBox();

  preload_images_Label = new QLabel(tr("Preload Images"));
  preload_images_Box = new QComboBox();

  QFormLayout *rightlayout = new QFormLayout;
  rightlayout->addRow(colorLabel);
  rightlayout->addRow(colorBox);
  rightlayout->addRow(precision_Label);
  rightlayout->addRow(precision_Box);
  rightlayout->addRow(update_images_Label);
  rightlayout->addRow(update_images_Box);
  rightlayout->addRow(preload_images_Label);
  rightlayout->addRow(preload_images_Box);

  // far right layout
  error_handlingModeLabel = new QLabel(tr("Error Handling"));
  error_handlingModeBox = new QComboBox();

  arrowLabel = new QLabel(tr("Display Arrows"));
  arrowBox = new QComboBox();

  arrow_magnificationLabel = new QLabel(tr("Arrow Magnification"));
  arrow_magnificationBox = new QDoubleSpinBox();

  plot_inside_pointsLabel = new QLabel(tr("Display Inside Points"));
  plot_inside_pointsBox = new QComboBox();

  plot_contour_pointsLabel = new QLabel(tr("Display Correlation Results"));
  plot_contour_pointsBox = new QComboBox();

  QFormLayout *farRightLayout = new QFormLayout;
  farRightLayout->addRow(error_handlingModeLabel);
  farRightLayout->addRow(error_handlingModeBox);
  farRightLayout->addRow(arrowLabel);
  farRightLayout->addRow(arrowBox);
  farRightLayout->addRow(arrow_magnificationLabel);
  farRightLayout->addRow(arrow_magnificationBox);
  farRightLayout->addRow(plot_inside_pointsLabel);
  farRightLayout->addRow(plot_inside_pointsBox);
  farRightLayout->addRow(plot_contour_pointsLabel);
  farRightLayout->addRow(plot_contour_pointsBox);

  QHBoxLayout *toplayout = new QHBoxLayout;
  toplayout->addItem(leftlayout);
  toplayout->addItem(centerlayout);
  toplayout->addItem(rightlayout);
  toplayout->addItem(farRightLayout);

  okButton = new QPushButton("Enter Values");

  QFormLayout *bottomlayout = new QFormLayout;
  bottomlayout->addRow(okButton);

  QVBoxLayout *layout = new QVBoxLayout;
  layout->addItem(toplayout);
  layout->addItem(bottomlayout);

  Subdi::setLayout(layout);
}
