// usual includes
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>

// local includes
//#include "interpolation_class.hpp"
#include "correlation_class.hpp"
#include "parameters.hpp"

// opencv includes
#include "core.hpp"
#include "highgui.hpp"

std::string errorStrings[4] = {"error_none", "error_model_out_of_image",
                               "error_interpolation_out_of_image",
                               "error_correlation_max_iters_reached"};

int main() {
  // model variables
  fittingModelEnum fittingModel = fm_QXYE; // UVUxUyVxVy;
  int number_of_fitting_parameters = ModelClass::get_number_of_model_parameters(
      fittingModel); // get #parameters class method

  // make some dummy model parameters for testing - these are the initial guess
  float *model_parameters = new float[number_of_fitting_parameters];
#if DEBUG_MEMORY
  std::cout << "allocating model_parameters  in driver= " << model_parameters
            << std::endl;
#endif

  // define interpolation model and variables
  interpolationModelEnum interpolationModel = im_bicubic;

  // Read images
  // std::string filename_und = "../../../../data/apple_crop2bw.png";
  // std::string filename_und = "../../../../data/SMRT101_35/Dock.png";
  // std::string filename_und =
  // "../../../../data/checkerboard_blurred_bw_100x100.png";
  std::string filename_und = "../../../../data/checkered_ssmooth.png";
  cv::Mat und_image = cv::imread(filename_und, cv::IMREAD_ANYCOLOR);
  if (!und_image.data) {
    // std::string filename_und = "../../../data/apple_crop2bw.png";
    // std::string filename_und = "../../../data/SMRT101_35/Dock.png";
    // std::string filename_und =
    // "../../../data/checkerboard_blurred_bw_100x100.png";
    std::string filename_und = "../../../data/checkered_ssmooth.png";
    und_image = cv::imread(filename_und, cv::IMREAD_ANYCOLOR);
  }
  std::cout << "und_image cols = " << und_image.cols << std::endl;

  // std::string filename_def = "../../../../data/apple_crop2bw.png";
  // std::string filename_def = "../../../../data/SMRT101_35/Dock_15deg.png";
  // std::string filename_def = "../../../../data/SMRT101_35/Dock.png";
  // std::string filename_def =
  // "../../../../data/checkerboard_blurred_bw_100x100_1x_5deg.png";
  std::string filename_def = "../../../../data/checkered_ssmooth_5degcw.png";
  cv::Mat def_image = cv::imread(filename_def, cv::IMREAD_ANYCOLOR);
  if (!def_image.data) {
    // std::string filename_def = "../../../data/apple_crop2bw.png";
    // std::string filename_def = "../../../data/SMRT101_35/Dock_15deg.png";
    // std::string filename_def = "../../../data/SMRT101_35/Dock.png";
    // std::string filename_def =
    // "../../../data/checkerboard_blurred_bw_100x100_1x_5deg.png";
    std::string filename_def = "../../../data/checkered_ssmooth_5degcw.png";
    def_image = cv::imread(filename_def, cv::IMREAD_ANYCOLOR);
  }
  std::cout << "def_image cols = " << def_image.cols << std::endl;

  // // test rectangular domain

  // model_parameters[0] = 0.f;
  // model_parameters[1] = 0.0f;
  // model_parameters[2] = 0.001f;
  // model_parameters[3] = 0.002f;
  // model_parameters[4] = 0.003f;
  // model_parameters[5] = 0.004f;

  // CorrelationClass correlator(und_image,
  // 							def_image,
  // 							50,
  // 							50,
  // 							interpolationModel,
  // 							fittingModel,
  // 							10,
  // 							10,
  // 							model_parameters,
  // 							0.001,	//required
  // precission
  // 							50);	//max iterations

  // float * new_pars = correlator.Newton_Raphson(model_parameters);

  // std::cout << "resulting model parameters = ";
  // for(int p = 0 ; p < number_of_fitting_parameters ; ++p )
  // 	std::cout << new_pars [ p ] << " ";
  // std::cout << std::endl;

  // //define the correlation problem - xy points in the undeformed image to
  // correlate
  // int x_start = 5;
  // int x_end = 15;

  // int y_start = 25;
  // int y_end = 35;

  // int number_of_points = (x_end - x_start + 1) * (y_end - y_start + 1);

  // //Dynamically allocate the interpolation parameters array and xy-position
  // (und) for all threads to share
  // float * xy_positions = new float[ number_of_points * 2];
  // #if DEBUG_MEMORY
  // std::cout << "allocating xy_positions in driver= " << xy_positions <<
  // std::endl;
  // #endif

  // //define correlation point locations
  // int i_point = -1;
  // for(int i = x_start ; i <= x_end; ++i)
  // 	for(int j = y_start ; j <= y_end; ++j)
  // 	{
  // 		xy_positions[ ++i_point ] = (float)i;
  // 		xy_positions[ ++i_point ] = (float)j;
  // 	}

  // //instanciate a correlator object
  // CorrelationClass correlator(und_image,
  // 							def_image,
  // 							number_of_points,
  // 							xy_positions,
  // 							interpolationModel,
  // 							fittingModel,
  // 							model_parameters,
  // 							0.001,	//required
  // precission
  // 							50);	//max iterations

  // float * new_pars = correlator.Newton_Raphson(model_parameters);

  // std::cout << "resulting model parameters = ";
  // for(int p = 0 ; p < number_of_fitting_parameters ; ++p )
  // 	std::cout << new_pars [ p ] << " ";
  // std::cout << std::endl;

  // //define a new correlation problem with the same images -
  // // xy points in the undeformed image to correlate
  // x_start = 15.f;
  // x_end = 25.f;

  // y_start = 25.f;
  // y_end = 35.f;

  // number_of_points = (x_end - x_start + 1) * (y_end - y_start + 1);

  // //Dynamically allocate the interpolation parameters array and xy-position
  // (und) for all threads to share
  // xy_positions = new float[ number_of_points * 2];
  // #if DEBUG_MEMORY
  // 	std::cout << "allocating xy_positions in driver= " << xy_positions <<
  // std::endl;
  // #endif

  // //define correlation point locations
  // i_point = -1;
  // for(int i = x_start ; i <= x_end; ++i)
  // 	for(int j = y_start ; j <= y_end; ++j)
  // 	{
  // 		xy_positions[ ++i_point ] = (float)i;
  // 		xy_positions[ ++i_point ] = (float)j;
  // 	}

  // model_parameters[0] = 0.5f;
  // model_parameters[1] = 1.0f;
  // model_parameters[2] = 0.001f;
  // model_parameters[3] = 0.002f;
  // model_parameters[4] = 0.003f;
  // model_parameters[5] = 0.004f;

  // new_pars = correlator.Newton_Raphson(model_parameters, number_of_points,
  // xy_positions);

  // std::cout << "resulting model parameters = ";
  // for(int p = 0 ; p < number_of_fitting_parameters ; ++p )
  // 	std::cout << new_pars [ p ] << " ";
  // std::cout << std::endl;

  // //test the window correlation

  // model_parameters[0] = 0.5f;
  // model_parameters[1] = 1.0f;
  // model_parameters[2] = 0.001f;
  // model_parameters[3] = 0.002f;
  // model_parameters[4] = 0.003f;
  // model_parameters[5] = 0.004f;

  // new_pars = correlator.Newton_Raphson(model_parameters, 23, 13, 6, 6);

  // std::cout << "resulting model parameters = ";
  // for(int p = 0 ; p < number_of_fitting_parameters ; ++p )
  // 	std::cout << new_pars [ p ] << " ";
  // std::cout << std::endl;

  // test annular correlation
  model_parameters[0] = -0.01f; // angle in radians
  model_parameters[1] = 112.f;  // Txc
  model_parameters[2] = 112.f;  // Txc
  model_parameters[3] = 0.00f;  // e

  CorrelationClass correlator(und_image, // annular correlation
                              def_image,
                              112,   // x center
                              112,   // y center
                              20.f,  // r in
                              100.f, // r out
                              interpolationModel, fittingModel,
                              model_parameters, 0.0000001, 500);

  float *new_pars = correlator.Newton_Raphson(model_parameters);

  std::cout << std::endl
            << "status = " << correlator.get_error_status() << std::endl
            << "error code  = " << errorStrings[correlator.get_error_code()]
            << std::endl
            << "resulting model parameters = ";
  for (int p = 0; p < number_of_fitting_parameters; ++p)
    std::cout << new_pars[p] << " ";
  std::cout << std::endl;

#if DEBUG_MEMORY
  std::cout << "deleting model_parameters in driver = " << model_parameters
            << std::endl;
#endif
  delete[] model_parameters;

  return 0;
}