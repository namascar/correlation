#ifndef PYRAMID_CLASS_H
#define PYRAMID_CLASS_H

#include <algorithm> //swap
#include <iostream>
#include <vector>

// multithread includes
#include <future>
#include <thread>

// openCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

// project includes
#include "enums.hpp"
#include "interpolation_class.hpp"
#include "parameters.hpp"

class Pyramid_class {
  int start;
  int step;
  int stop;

  int number_of_colors;
  int number_of_interpolation_parameters;

  cv::Mat und_image;
  cv::Mat def_image;
  cv::Mat nxt_image;

  std::vector<unsigned char *> und_images;
  int und_rows{0};
  int und_cols{0};
  int und_step{0};

  std::vector<unsigned char *> def_images;
  int def_rows{0};
  int def_cols{0};
  int def_step{0};

  std::vector<unsigned char *> nxt_images;
  int nxt_rows{0};
  int nxt_cols{0};
  int nxt_step{0};

  std::vector<float *> all_interpolation_parameters;
  int allocated_all_interpolation_parameters{0};

  std::vector<float *> xy_positions;
  std::vector<int> number_of_points;
  v_points xy_center;

  fittingModelEnum fittingModel;
  int number_of_model_parameters{0};

  void clear_und_images();
  void clear_def_images();
  void clear_nxt_images();
  void clear_xy_positions();
  void clear_all_interpolation_parameters();
  void set_all_interpolation_parameters();
  void reset_all_interpolation_parameters();

  std::vector<unsigned char *>
  make_pyramid(std::vector<unsigned char *> pyramid_in, ImageType imageType);

public:
  Pyramid_class(int start_in, int step_in, int stop_in, int number_of_colors_in,
                int number_of_interpolation_parameters_in,
                fittingModelEnum fittingModel_in)
      :

        start(start_in),
        step(step_in), stop(stop_in), number_of_colors(number_of_colors_in),
        number_of_interpolation_parameters(
            number_of_interpolation_parameters_in),
        fittingModel(fittingModel_in) {}

  // Setting methods
  void set_und_image(const cv::Mat &und_image_in);
  void set_def_image(const cv::Mat &def_image_in);
  void set_nxt_image(const cv::Mat &nxt_image_in);
  void und_from_def();
  void def_from_nxt();
  void set_xy_positions(float *xy_positions_in, int number_of_points_in);
  void set_number_of_model_parameters(int number_of_model_parameters_in);
  void set_und_center(float und_x_center_in, float und_y_center_in);
  void set_und_center();

  // Translation methods
  void translate_model_parameters(float *model_parameters,
                                  int pyramid_level_src, int pyramid_level_dst);
  // Getting methods
  unsigned char *get_und_ptr(int level);
  unsigned char *get_def_ptr(int level);
  float *get_all_param(int level);
  float *get_xy_positions(int level);
  int get_number_of_points(int level);
  int get_rows(int level, ImageType type);
  int get_cols(int level, ImageType type);
  int get_step(int level, ImageType type);
  void get_und_center(float &und_x_center_in, float &und_y_center_in,
                      int level);

  ~Pyramid_class() {
    clear_und_images();
    clear_def_images();
    clear_xy_positions();
    clear_all_interpolation_parameters();
  }
};

#endif // PYRAMID_CLASS_H
