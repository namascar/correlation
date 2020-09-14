#ifndef MANAGER_CLASS_H
#define MANAGER_CLASS_H
//----------------------------------------------------------------------
//
//   include
//
//----------------------------------------------------------------------
#include "correlation_class.hpp"
#include "interpolation_class.hpp"
#include "parameters.hpp"
#include "polygon_class.h"

#if CUDA_ENABLED
#include "cuda_class.cuh"
#endif

#include <QApplication>
#include <QMainWindow>
#include <QObject>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <chrono>
#include <iostream>
#include <sstream>

typedef std::vector<std::vector<float>> polyEquations;

//----------------------------------------------------------------------
//
//   classes
//
//----------------------------------------------------------------------
class managerClass : public QObject {
  Q_OBJECT

  QStringList *fileNames = nullptr;

  cv::Mat undeformed_opencvMat;
  std::string und_file_string;

  cv::Mat deformed_opencvMat;
  std::string def_file_string;

  cv::Mat next_opencvMat;

  colorEnum color_mode;
  int number_of_colors;
  int color_flag;
  updateEnum update;
  fittingModelEnum model;
  int number_of_model_parameters;
  interpolationModelEnum interpolation;
  processorEnum processor;
  domainEnum domain_type;
  float *global_initial_guess = nullptr;
  float *initial_guess = nullptr;
  deformationDescriptionEnum deformationDescription;
  errorHandlingModeEnum error_handling_mode;
  referenceImageEnum referenceImage;
  int pyramid_start{0};
  int pyramid_step{0};
  int pyramid_stop{0};

  bool plot_inside_points;
  bool plot_contour_points;

  float precision;
  int max_iters;
  int number_of_threads;

  rectangularDomainStruct rectangularDomain;
  annularDomainStruct annularDomain;
  blobDomainStruct blobDomain;

  bool error;
  errorEnum errorType;

  CorrelationClass *correlator = nullptr;

#if CUDA_ENABLED
  CudaClass *cuda_manager;
#endif

  // timing variables
  float time_all_point_selection;

  void set_undeformed_image(std::string und_fileName_in,
                            int current_frame_order,
                            referenceImageEnum referenceImage);

  void set_deformed_image(std::string def_fileName_in, int current_frame_order);
  bool set_next_image(std::string next_fileName_in);
  bool perform_single_frame_correlation_rectangular(int hs, int vs,
                                                    frame_results *results_in,
                                                    int current_frame_order);

  bool perform_single_frame_correlation_annular(int rs, int as,
                                                frame_results *results_in,
                                                int current_frame_order);

  bool perform_single_frame_correlation_blob(frame_results *results_in,
                                             int current_frame_order);

  void adjust_rectangular_domain(int &center_x, int &center_y,
                                 frame_results *previous_results,
                                 int current_frame_order);

  void adjust_annular_domain(int i, int j, float &r, float &a, float &ri,
                             float dr, float da, float &center_x,
                             float &center_y, int as,
                             frame_results *all_previous_results,
                             int current_frame_order);

  void adjust_blob_domain(frame_results *previous_results,
                          int current_frame_order);

  void adjust_initial_guess(float *initial_guess,
                            frame_results *previous_results,
                            int current_frame_order);

  v_points get_inside_points_blobDomain(v_points xy_contour_in);
  v_points get_inside_points_blobDomain_non_convex(v_points xy_contour_in,
                                                   int max_x, int min_x,
                                                   int max_y, int min_y);
  v_points get_inside_points_blobDomain_convex(v_points xy_contour_in,
                                               int max_x, int min_x, int max_y,
                                               int min_y);
  v_points get_inside_points_rectangularDomain(int x0, int y0, int x1, int y1);
  v_points get_inside_points_annularDomain(float r, float dr, float a, float da,
                                           float cx, float cy, int as);

  v_points get_contour_points_rectangularDomain(int x0, int y0, int x1, int y1);
  v_points get_contour_points_annularDomain(float r, float dr, float a,
                                            float da, float cx, float cy);

  v_points deformPoints(v_points contour, float cx, float cy, float ro = 0.f);

  bool check_polygon_convex(v_points xy_contour_in);
  polyEquations makeLineEquations(v_points xy_contour_in);
  bool check_inside_polygon(const v_points &xy_contour_in, int ix, int iy);
  bool check_inside_polygon(const v_points &xy_contour_in, int ix, int iy,
                            const polyEquations &contourLineEquations);

  intersectionEnum check_segment_intersection(float v1x1, float v1y1,
                                              float v1x2, float v1y2,
                                              float v2x1, float v2y1,
                                              float v2x2, float v2y2);

  intersectionEnum
  check_segment_intersection(float v1x2, float v1y2, float v2y1, float v2y2,
                             const std::vector<float> &lineEquation);

  void update_results(frame_results *results_in,
                      CorrelationResult *correlationResult = nullptr);
  void update_global_results(int hrs, int vas, frame_results *results_in);
  void addFrameToReport(int frame_number_in, int results_i_in, int results_j_in,
                        frame_results *results_in);
  void initializeReport();
  int estimate_number_of_points(rectangularDomainStruct rectangularDomain_in);
  int estimate_number_of_points(annularDomainStruct annularDomain_in);
  int estimate_number_of_points(blobDomainStruct blobDomain_in);

public:
  explicit managerClass(QObject *parent = 0);
  ~managerClass();

  void set_filenames(QStringList *fileNames_in);

  bool set_domain(rectangularDomainStruct rectangularDomain);
  bool set_domain(annularDomainStruct annularDomain);
  bool set_domain(blobDomainStruct blobDomain);

#if CUDA_ENABLED
  void set_cuda_manager(CudaClass *cuda_manager_in);
#endif

  void set_color_mode(colorEnum color_mode_in, int number_of_colors_in);
  void set_update(updateEnum update);
  void set_model(fittingModelEnum model_in);
  void set_interpolation(interpolationModelEnum interpolation_in);
  void set_global_initial_guess(float *global_initial_guess_in);
  void set_processor(processorEnum processor_in);

  void set_precision(float precision_in);
  void set_max_iters(int max_iters_in);
  void set_number_of_threads(int number_of_threads_in);
  void set_deformation_description(
      deformationDescriptionEnum deformation_description_in);
  void set_plot_inside_points(bool plot_inside_points_in);
  void set_plot_contour_points(bool plot_contour_points_in);
  void set_error_handling_mode(errorHandlingModeEnum error_handling_mode_in);
  void set_referenceImage(referenceImageEnum referenceImage_in);
  void set_pyramid(int pyramid_start_in, int pyramid_step_in,
                   int pyramid_stop_in);

  bool stop_flag;
  std::stringstream report;

  errorEnum get_errorType();

signals:
  void correlation_is_done(bool error);
  void send_und_inside_points(v_points inside_points_vector, bool error);
  void send_und_contour_points(v_points contour_points_vector, bool error);
  void send_def_inside_points(v_points inside_points_vector, bool error);
  void send_def_contour_points(v_points contour_points_vector, bool error);
  void clear_und_inside_points();
  void clear_def_inside_points();
  void clear_und_contour_points();
  void clear_def_contour_points();
  void display_images(QString und_file_QString, QString def_file_QString);
  void display_results(float angle, float *parameters);

private slots:
  void
  perform_multiframe_correlation(); /**< Private method can only be called via
                                       "emit" so that it runs on the manager
                                       thread */
};

#endif // MANAGER_CLASS_H
