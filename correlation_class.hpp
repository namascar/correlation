#ifndef CORRELATION_CLASS_H
#define CORRELATION_CLASS_H

// local includes
#include "interpolation_class.hpp"
#include "model_class.hpp"
#include "parameters.hpp"
#include <pyramid_class.h>

// opencv includes
#include "opencv2/core/core.hpp"

// eigen includes
#include <Dense>

// usual includes
#include <ctime>
#include <iostream>
#include <limits>

// multithread includes
#include <future>
#include <thread>

// This struct contains the parameters to call each correlation job.
//	Each job has an object with all information in its member data
struct s_correlation_thread_data {
  bool runInterpolation;

  int thread_id;
  int first_point_this_thread;
  int number_of_points_this_thread;
  InterpolationClass *interpolator;
  ModelClass *modeler;
  float *contribution_mat_A = nullptr;
  float *contribution_vec_B = nullptr;
  float contribution_chi;
  bool error_status;
  errorEnum error_code;
  float model_time;
  float interpolation_time;
};

class CorrelationClass {
private:
  // general variables
  bool error_status;
  bool set_center_flag;
  errorEnum error_code;
  int allocated_points{0};
  float *und_intensities = nullptr;
  float *def_xy_positions = nullptr;
  float *solution = nullptr; // solver solution

  // image variables
  cv::Mat def_image;
  cv::Mat und_image;
  cv::Mat nxt_image;
  int number_of_colors;

  // model fiting variables
  float *mat_A = nullptr;
  float *vec_B = nullptr;
  float *vec_x = nullptr;
  float chi;

  // interpolation object and variables
  interpolationModelEnum interpolationModel;
  std::vector<InterpolationClass *> interpolators;
  int number_of_interpolation_parameters =
      InterpolationClass::get_number_of_interpolation_parameters(
          interpolationModel);

  float *w_results = nullptr;
  float *dwdxy_results = nullptr;

  // modeler object and variables
  fittingModelEnum fittingModel;
  ModelClass *modelers = nullptr;
  int number_of_model_parameters;
  float *model_parameters = nullptr;
  float *dTxydp = nullptr;
  int number_of_threads;

  float required_precision;
  int maximum_iterations;
  int reached_iterations;
  float last_good_chi;

  // pyramid
  int pyramid_start{0};
  int pyramid_step{0};
  int pyramid_stop{0};

  s_correlation_thread_data *correlation_thread_data = nullptr;
  void *status;

  // timing variables
  float time_all_model;
  float time_all_interpolation;
  float start_chi;
  float duration_chi;
  float start_new_parameters;
  float duration_new_parameters;
  float start_new_parameters_assembly;
  float duration_new_parameters_assembly;
  float time_all_new_parameters_assembly;
  float start_new_parameters_solver;
  float duration_new_parameters_solver;
  float time_all_new_parameters_solver;
  float start_newton_raphson;
  float duration_newton_raphson;

  void apply_model_and_interpolate(int pyramid_level, bool runInterpolation);
  void flush_A_B();
  void compute_model_parameters(float lambda, float scaling);
  float *solve();
  void debug_correlation(int pyramid_level);
  void debug_correlation_info(int pyramid_level);
  void allocate_point_dependent_arrays();
  void delete_point_dependent_arrays();
  void Newton_Raphson_dump(int pyramid_level);

  Pyramid_class pyramid{pyramid_start,
                        pyramid_step,
                        pyramid_stop,
                        number_of_colors,
                        number_of_interpolation_parameters,
                        fittingModel};

public:
  CorrelationClass(int allocated_points_in, int number_of_colors_in,
                   interpolationModelEnum interpolationModel_in,
                   fittingModelEnum fittingModel_in, int number_of_threads_in,
                   float required_precision_in, int maximum_iterations_in,
                   int pyramid_start_in, int pyramid_step_in,
                   int pyramid_stop_in)
      :

        allocated_points(allocated_points_in),
        number_of_colors(number_of_colors_in),
        interpolationModel(interpolationModel_in),
        fittingModel(fittingModel_in), number_of_threads(number_of_threads_in),
        required_precision(required_precision_in),
        maximum_iterations(maximum_iterations_in),
        pyramid_start(pyramid_start_in), pyramid_step(pyramid_step_in),
        pyramid_stop(pyramid_stop_in) {
    initialize_correlation();
  }

  ~CorrelationClass();

  void initialize_correlation();
  void set_undeformed_image(cv::Mat &und_image_in);
  void set_und_image_from_def();
  void set_deformed_image(cv::Mat &def_image_in);
  void set_def_image_from_nxt();
  void set_next_image(cv::Mat &nxt_image_in);

  float *get_model_parameters();

  float *Newton_Raphson(float *model_parameters_in);

  float *Newton_Raphson(float *model_parameters_in, // blob correlation
                        int number_of_points_in, float *xy_points_in);

  float *Newton_Raphson(float *model_parameters_in, // blob with known center
                        int number_of_points_in, float und_x_center_in,
                        float und_y_center_in, float *und_xy_positions_in);

  bool get_error_status();
  float get_chi();
  int get_number_of_points();
  float get_und_x_center();
  float get_und_y_center();
  int get_iterations();
  v_points getUndXY0();
  v_points getDefXY0();
  errorEnum get_error_code();
};

#endif
