#ifndef INTERPOLATION_CLASS_H
#define INTERPOLATION_CLASS_H

// local includes
#include "parameters.hpp"

// opencv includes
#include "opencv2/core/core.hpp"

// eigen includes
#include <Dense>

// timing
#include <chrono>
#include <iostream>

float modelU_distort_x(float x, float y, float cx, float cy, float ro,
                       float *model_parameters);
float modelU_distort_y(float x, float y, float cx, float cy, float ro,
                       float *model_parameters);

float modelUV_distort_x(float x, float y, float cx, float cy, float ro,
                        float *model_parameters);
float modelUV_distort_y(float x, float y, float cx, float cy, float ro,
                        float *model_parameters);

float modelUVQ_distort_x(float x, float y, float cx, float cy, float ro,
                         float *model_parameters);
float modelUVQ_distort_y(float x, float y, float cx, float cy, float ro,
                         float *model_parameters);

float modelUVUxUyVxVy_distort_x(float x, float y, float cx, float cy, float ro,
                                float *model_parameters);
float modelUVUxUyVxVy_distort_y(float x, float y, float cx, float cy, float ro,
                                float *model_parameters);

class InterpolationClass {
protected:
  //  Common space
  int thread_id;
  bool error_status;
  errorEnum error_code;
  int number_of_colors;
  int number_of_points;
  unsigned char *und_image_ptr = nullptr;
  int und_image_step = 0;
  unsigned char *def_image_ptr = nullptr;
  int def_image_rows = 0;
  int def_image_cols = 0;
  int def_image_step = 0;

  //  Interpolation space
  interpolationModelEnum interpolation_model;
  int number_of_interpolation_parameters;
  float *interpolation_matrix = nullptr;
  float *interpolation_vector = nullptr;
  float *all_parameters = nullptr;
  float *local_all_parameters = nullptr;

  float *und_xy_positions =
      nullptr; // Array with all the undeformed locations for all points
  float *und_intensities = nullptr; // Array with all the undeformed intensities
                                    // for all points - has to go
  float *w_results = nullptr; // Array with all the deformed intensities for all
                              // points - has to go
  float *dwdxy_results = nullptr; // Array with all the deformed intensities
                                  // gradients for all points - has to go
  float *def_xy_positions = nullptr; // Array with all the deformed locations
                                     // for all points - has to go

  float *und_w = nullptr; // array with undeformed intensity for one point - one
                          // entry per color
  float *def_w = nullptr; // array with deformed intensity for one point - one
                          // entry per color
  float *def_dwdxy = nullptr; // array with deformed intensity gradients (wrt
                              // x,y) for one point - two entries per color
  float *dTxydp = nullptr;    // array with transformation gradients (wrt p) for
                           // one point - two entries per number_of_parameters

  float *vec_B;
  float *mat_A;
  float chi;

  //  Modeling space
  fittingModelEnum fitting_model;
  int number_of_model_parameters;

  float *model_parameters = nullptr;
  float und_x_center;
  float def_x_center;
  float und_y_center;
  float def_y_center;

  // timing variables
  float time_duration_interpolation;

  // Methods
  virtual void make_interpolation_matrix() = 0;
  inline void get_interpolation_parameters(int x_pos, int y_pos);
  void debug_interpolation();
  virtual void get_new_interpolation_parameters(float *part_of_all_parameters,
                                                int x_pos, int y_pos,
                                                int color_in) = 0;
  virtual void get_interpolation(float *part_of_w_results,
                                 float *part_of_dwdxy_results, float xdef,
                                 float ydef) = 0;

public:
  InterpolationClass(const int number_of_colors_in,
                     const fittingModelEnum &fittingModel_in)
      : number_of_colors(number_of_colors_in), fitting_model(fittingModel_in) {
    error_status = false;
    number_of_model_parameters = get_number_of_model_parameters(fitting_model);

    // dTxydp = new float [ number_of_model_parameters * 2];
    vec_B = new float[number_of_model_parameters];
    mat_A = new float[number_of_model_parameters * number_of_model_parameters];
  }

  virtual ~InterpolationClass() = 0;

  int get_thread_id();
  float get_time();
  void set_thread_id(int thread_id_in);
  void set_def_image(unsigned char *def_image_ptr_in, int def_image_rows_in,
                     int def_image_cols_in, int def_image_step_in,
                     float *all_parameters_in);
  void set_und_image(unsigned char *und_image_ptr_in, int und_image_step_in);
  void get_multiple_interpolations();
  void set_multiple_interpolations(
      int number_of_points_in, float *und_intensities_in,
      float *und_xy_positions_in, float *def_xy_positions_in,
      float *w_results_in, float *dwdxy_results_in, float *model_parameters_in,
      float und_x_center_in, float und_y_center_in, float *dTxydp_in);
  bool get_error_status();
  void set_error_status(bool error_status_in);
  errorEnum get_error_code();
  void set_error_code(errorEnum error_code_in);
  int get_number_of_colors();
  static int get_number_of_interpolation_parameters(
      interpolationModelEnum interpolationModel_in);
  static int get_number_of_model_parameters(fittingModelEnum fittingModel_in);
  static std::vector<InterpolationClass *>
  new_InterpolationClass(const interpolationModelEnum &interpolationModel_in,
                         const fittingModelEnum &fittingModel_in,
                         const int number_of_colors_in,
                         const int number_of_threads);
  float *get_mat_A();
  float *get_vec_B();
  float get_chi();
};

class InterpolationClass_bicubic : public InterpolationClass {
private:
  void make_interpolation_matrix();

protected:
  inline void get_new_interpolation_parameters(float *part_of_all_parameters,
                                               int x_pos, int y_pos,
                                               int color_in);
  inline void get_interpolation(float *part_of_w_results,
                                float *part_of_dwdxy_results, float xdef,
                                float ydef);

public:
  InterpolationClass_bicubic(const int number_of_colors_in,
                             const fittingModelEnum &fittingModel_in)
      : InterpolationClass(number_of_colors_in, fittingModel_in) {
    interpolation_model = im_bicubic;
    number_of_interpolation_parameters =
        get_number_of_interpolation_parameters(interpolation_model);

    interpolation_matrix = new float[number_of_interpolation_parameters *
                                     number_of_interpolation_parameters];
    interpolation_vector = new float[number_of_interpolation_parameters];
    local_all_parameters = new float[number_of_interpolation_parameters];

    make_interpolation_matrix();
  }

  ~InterpolationClass_bicubic();
};

class InterpolationClass_bilinear : public InterpolationClass {
private:
  void make_interpolation_matrix();

protected:
  inline void get_new_interpolation_parameters(float *part_of_all_parameters,
                                               int x_pos, int y_pos,
                                               int color_in);
  inline void get_interpolation(float *part_of_w_results,
                                float *part_of_dwdxy_results, float xdef,
                                float ydef);

public:
  InterpolationClass_bilinear(const int number_of_colors_in,
                              const fittingModelEnum &fittingModel_in)
      : InterpolationClass(number_of_colors_in, fittingModel_in) {
    interpolation_model = im_bilinear;
    number_of_interpolation_parameters =
        get_number_of_interpolation_parameters(interpolation_model);

    local_all_parameters = new float[number_of_interpolation_parameters];
  }
  ~InterpolationClass_bilinear();
};

class InterpolationClass_nearest : public InterpolationClass {
private:
  void make_interpolation_matrix();

protected:
  inline void get_new_interpolation_parameters(float *part_of_all_parameters,
                                               int x_pos, int y_pos,
                                               int color_in);
  inline void get_interpolation(float *part_of_w_results,
                                float *part_of_dwdxy_results, float xdef,
                                float ydef);

public:
  InterpolationClass_nearest(const int number_of_colors_in,
                             const fittingModelEnum &fittingModel_in)
      : InterpolationClass(number_of_colors_in, fittingModel_in) {
    interpolation_model = im_nearest;
    number_of_interpolation_parameters =
        get_number_of_interpolation_parameters(interpolation_model);

    local_all_parameters = new float[number_of_interpolation_parameters];
  }
  ~InterpolationClass_nearest();
};

#endif
