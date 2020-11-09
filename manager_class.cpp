/*-------------------------------------------------------------------

  File        : manager_class.cpp

  Description : Manages the correlation process to be performed either on
                the CPU or the GPU. Given a user selected domain, from
                MainApp.cpp, selects the individual points to be used on
                the correlation. Can also select the bounding contour of
                the correlation domain for plotting purposes.
                The class inherits from Qt Object, to be able to emit and
                receive Qt signals, to communicate with the MainApp GUI.

  Author  : Javier Gonzalez, 29-October-2017 - 21-May-2018

  -----------------------------------------------------------------*/

//----------------------------------------------------------------------
//
//   include
//
//----------------------------------------------------------------------

#include "manager_class.h"

//----------------------------------------------------------------------
//
//   functors
//
//----------------------------------------------------------------------

//  This functor is used to aggregate the deformation points resulting from one
//  correlation step with
//      correlation parameters into the next set of undeformed points. Next
//      undeformed points need to
//      be integer values, since the undeformed intensity values are queried
//      from the undeformed image
//      at the nodal points, without interpolation.
struct add_pair {
  add_pair(std::pair<float, float> x_in) : x(x_in) {}
  std::pair<float, float> operator()(std::pair<float, float> y) const {
    return std::make_pair((int)(x.first + y.first + 0.5f),
                          (int)(x.second + y.second + 0.5f));
  }

private:
  std::pair<float, float> x;
};

//----------------------------------------------------------------------
//
//   classes
//
//----------------------------------------------------------------------

managerClass::managerClass(QObject *parent) : QObject(parent) {}

managerClass::~managerClass() {
  delete[] global_initial_guess;
  global_initial_guess = nullptr;

  delete[] initial_guess;
  initial_guess = nullptr;
}

void managerClass::set_filenames(QStringList *fileNames_in) {
  fileNames = fileNames_in;
}

#if CUDA_ENABLED
void managerClass::set_cuda_manager(CudaClass *cuda_manager_in) {
  cuda_manager = cuda_manager_in;
}
#endif

bool managerClass::set_domain(rectangularDomainStruct rectangularDomain_in) {
  domain_type = domain_rectangular;
  rectangularDomain = rectangularDomain_in;

  return true;
}

bool managerClass::set_domain(annularDomainStruct annularDomain_in) {
  domain_type = domain_annular;
  annularDomain = annularDomain_in;

  return true;
}

bool managerClass::set_domain(blobDomainStruct blobDomain_in) {
  domain_type = domain_blob;
  blobDomain = blobDomain_in;

  return true;
}

void managerClass::set_color_mode(colorEnum color_mode_in,
                                  int number_of_colors_in) {
  color_mode = color_mode_in;

  switch (color_mode) {
  case color_monochrome:
    color_flag = cv::IMREAD_GRAYSCALE;
    number_of_colors = 1;
    break;

  case color_color:
    color_flag = cv::IMREAD_ANYCOLOR;
    number_of_colors = number_of_colors_in;
    break;

  default:
    assert(false);
    break;
  }
}

void managerClass::set_update(updateEnum update_in) { update = update_in; }

void managerClass::set_processor(processorEnum processor_in) {
  processor = processor_in;
}

void managerClass::set_model(fittingModelEnum model_in) {
  model = model_in;
  number_of_model_parameters =
      ModelClass::get_number_of_model_parameters(model);
}

void managerClass::set_interpolation(interpolationModelEnum interpolation_in) {
  interpolation = interpolation_in;
}

void managerClass::set_global_initial_guess(float *global_initial_guess_in) {

  delete[] global_initial_guess;
  global_initial_guess = nullptr;

  delete[] initial_guess;
  initial_guess = nullptr;

  global_initial_guess = new float[number_of_model_parameters];
  initial_guess = new float[number_of_model_parameters];

  for (int p = 0; p < number_of_model_parameters; ++p) {
    global_initial_guess[p] = global_initial_guess_in[p];
  }
}

void managerClass::set_max_iters(int max_iters_in) { max_iters = max_iters_in; }
void managerClass::set_precision(float precision_in) {
  precision = precision_in;
}

void managerClass::set_plot_inside_points(bool plot_inside_points_in) {
  plot_inside_points = plot_inside_points_in;
}

void managerClass::set_plot_contour_points(bool plot_contour_points_in) {
  plot_contour_points = plot_contour_points_in;
}

void managerClass::set_error_handling_mode(
    errorHandlingModeEnum error_handling_mode_in) {
  error_handling_mode = error_handling_mode_in;
}

void managerClass::set_undeformed_image(std::string und_fileName_in,
                                        int current_frame_order,
                                        referenceImageEnum referenceImage) {
  switch (processor) {
  case processor_CPU:
    if (current_frame_order == 0) {
      // Load und image from disk
      undeformed_opencvMat = cv::imread(und_fileName_in, color_flag);
      // Call the cpu pyramid to make the pyramid
      correlator->set_undeformed_image(undeformed_opencvMat);
    } else if (referenceImage == refImage_Previous) {
      // Call the cpu pyramid to repoint the defPyramid to undPyramid
      correlator->set_und_image_from_def();
    }
    break;

  case processor_GPU:
#if CUDA_ENABLED
    if (current_frame_order == 0) {
      // First undeformed pyramid is already made by the
      // mainApp::updateGpuPyramids, which is called
      // whenever we 1)load new images or 2)change the pyramid parameters or
      // 3)change the color mode
      // or 4)are done correlating or 5)change processor type. Via
      // cuda_manager::resetImagePyramids
    } else if (referenceImage == refImage_Previous) {
      // Rename the defPyramid as undPyramid and delete the old undPyramid
      cuda_manager->makeUndPyramidFromDef();
    }
#endif
    break;

  default:
    assert(false);
    break;
  }
}

void managerClass::set_deformed_image(std::string def_fileName_in,
                                      int current_frame_order) {
  switch (processor) {
  case processor_CPU:
    if (current_frame_order == 0) {
      // Load und image from disk
      deformed_opencvMat = cv::imread(def_fileName_in, color_flag);
      // Call the cpu pyramid to make the pyramid
      correlator->set_deformed_image(deformed_opencvMat);
    } else {
      // Call the cpu pyramid to repoint the defPyramid to undPyramid
      correlator->set_def_image_from_nxt();
    }
    break;

  case processor_GPU:
#if CUDA_ENABLED
    if (current_frame_order == 0) {
      // First deformed pyramid is already made by the
      // mainApp::updateGpuPyramids, which is called
      // whenever we 1) load new images or 2)change the pyramid parameters or
      // 3)change the color mode
      // or 4)are done correlating or 5)change processor type. Via
      // cuda_manager::resetImagePyramids

      // Temporary code while the cuda pyramids are implemented
      // deformed_opencvMat = cv::imread( def_fileName_in, color_flag );
      // end of temporary code
    } else {
      cuda_manager->makeDefPyramidFromNxt();
    }
#endif
    break;

  default:
    assert(false);
    break;
  }
}

bool managerClass::set_next_image(std::string next_fileName_in) {
  bool error = true;

  switch (processor) {
  case processor_CPU:
    next_opencvMat = cv::imread(next_fileName_in, color_flag);
    correlator->set_next_image(next_opencvMat);
    error = next_opencvMat.empty();
    break;

  case processor_GPU:
#if CUDA_ENABLED
    cuda_manager->resetNextPyramid(next_fileName_in);
    error = false; // for now
#endif
    break;

  default:
    assert(false);
    break;
  }

  return error;
}

void managerClass::set_referenceImage(referenceImageEnum referenceImage_in) {
  referenceImage = referenceImage_in;
}

bool managerClass::perform_single_frame_correlation_rectangular(
    int hs, int vs, frame_results *results_in, int current_frame_order) {
  int x1 = (int)rectangularDomain.x_end;
  int x0 = (int)rectangularDomain.x_begin;

  int y1 = (int)rectangularDomain.y_end;
  int y0 = (int)rectangularDomain.y_begin;

  // Integer size required for correlation constructor
  int xdim = (abs(x1 - x0) / hs - 1) / 2;
  int ydim = (abs(y1 - y0) / vs - 1) / 2;

  // Float set of square dimensions, with float arithmetic for precise
  // positioning of centers
  //      when using subdivisions
  float fx1 = rectangularDomain.x_end;
  float fx0 = rectangularDomain.x_begin;
  float fhs = (float)hs;

  float fy1 = rectangularDomain.y_end;
  float fy0 = rectangularDomain.y_begin;
  float fvs = (float)vs;

  // Float size needed for precise positioning of correlation
  //      squares. Otherwhise rounding error aggregates over squares.
  float fxdim = (fabs(fx1 - fx0) / fhs - 1.f) / 2.f;
  float fydim = (fabs(fy1 - fy0) / fvs - 1.f) / 2.f;

  time_all_point_selection = 0;

  for (int i = 0; i < hs; ++i) {
    int center_x = (int)(0.5f + fx0 + fxdim + (2.f * fxdim + 1.f) * (float)i);

    for (int j = 0; j < vs; ++j) {
      int iSector = i * vs + j;

      int center_y = (int)(0.5f + fy0 + fydim + (2.f * fydim + 1.f) * (float)j);

      // Define the center_x and center_y. Note "adjust_rectangular_domain"
      // modifies sector's center_x and center_y.
      adjust_rectangular_domain(center_x, center_y, &results_in[iSector],
                                current_frame_order); // want later to make
                                                      // parameters i and j, not
                                                      // center_x and center_y

      adjust_initial_guess(initial_guess, &results_in[iSector],
                           current_frame_order);

      // Definition of the undeformed xy points and number_of_points for this
      // sector.
      std::chrono::system_clock::time_point start_point_selection =
          std::chrono::system_clock::now();

      if (current_frame_order == 0) {
        results_in[iSector].und_contour = get_contour_points_rectangularDomain(
            center_x - xdim, center_y - ydim, center_x + xdim, center_y + ydim);
        switch (processor) {
        case processor_CPU:
          results_in[iSector].und_inside_points =
              get_inside_points_rectangularDomain(
                  center_x - xdim, center_y - ydim, center_x + xdim,
                  center_y + ydim);
          break;

        case processor_GPU:
#if CUDA_ENABLED
          cuda_manager->resetPolygon(iSector, center_x - xdim, center_y - ydim,
                                     center_x + xdim, center_y + ydim);

          if (plot_inside_points) {
            results_in[iSector].und_inside_points =
                cuda_manager->getUndXY0ToCPU(iSector);
          }
#endif
          break;

        default:
          assert(false);
          break;
        }
      } else // Logic to get the undeformed points on frames other than the
             // first
      {
        switch (deformationDescription) {
        case def_Eulerian:
          //  Do nothing
          break;

        case def_strict_Lagrangian:
          results_in[iSector].und_inside_points =
              results_in[iSector].def_inside_points;
          results_in[iSector].und_contour = results_in[iSector].def_contour;

          if (processor == processor_GPU) {
#if CUDA_ENABLED
            cuda_manager->updatePolygon(iSector, def_strict_Lagrangian);
            if (plot_inside_points) {
              results_in[iSector].und_inside_points =
                  cuda_manager->getUndXY0ToCPU(iSector);
            }
#endif
          }
          break;

        case def_Lagrangian: {
          std::pair<float, float> center_offset =
              std::make_pair(results_in[iSector].und_center_x -
                                 results_in[iSector].past_und_center_x,

                             results_in[iSector].und_center_y -
                                 results_in[iSector].past_und_center_y);

          std::transform(results_in[iSector].und_contour.begin(),
                         results_in[iSector].und_contour.end(),
                         results_in[iSector].und_contour.begin(),
                         add_pair(center_offset));

          switch (processor) {
          case processor_CPU:
            std::transform(results_in[iSector].und_inside_points.begin(),
                           results_in[iSector].und_inside_points.end(),
                           results_in[iSector].und_inside_points.begin(),
                           add_pair(center_offset));
            break;

          case processor_GPU:
#if CUDA_ENABLED
            cuda_manager->updatePolygon(iSector, def_Lagrangian);
            if (plot_inside_points) {
              results_in[iSector].und_inside_points =
                  cuda_manager->getUndXY0ToCPU(iSector);
            }
#endif
            break;

          default:
            assert(false);
            break;
          }
        } break;

        default:
          assert(false);
          break;
        }
      }

      std::chrono::system_clock::time_point duration_point_selection =
          std::chrono::system_clock::now();
      time_all_point_selection +=
          (float)std::chrono::duration_cast<std::chrono::milliseconds>(
              duration_point_selection - start_point_selection)
              .count() /
          1000.f;
#if DEBUG_TIME_POINT_SELECTION
      std::cout << "manager()     :point selection wall execution time(s): "
                << time_all_point_selection << '\n';
#endif

      CorrelationResult *correlationResults = nullptr;

      // Call the correlator with these points
      switch (processor) {
      case processor_CPU:
        initial_guess = correlator->Newton_Raphson(
            initial_guess, (int)results_in[iSector].und_inside_points.size(),
            center_x, center_y,
            (float *)results_in[iSector].und_inside_points.data());

        error = correlator->get_error_status();
        break;

      case processor_GPU:
#if CUDA_ENABLED

        correlationResults = cuda_manager->correlate(iSector, initial_guess,
                                                     results_in[iSector]);
        errorType = correlationResults->errorCode;
        error = (errorType != error_none);

#endif
        break;

      default:
        assert(true);
        break;
      }

      // Updating center and other correlation results from the correlator for
      // the report
      update_results(&results_in[iSector], correlationResults);

      switch (processor) {
      case processor_CPU:
        if (plot_inside_points ||
            deformationDescription == def_strict_Lagrangian) {
          results_in[iSector].def_inside_points = correlator->getDefXY0();
        }
        break;

      case processor_GPU:
#if CUDA_ENABLED
        if (plot_inside_points) {
          results_in[iSector].def_inside_points =
              cuda_manager->getDefXY0ToCPU(iSector);
        }
#endif
        break;

      default:
        assert(false);
        break;
      }

      if (plot_inside_points) {
        // connect(manager, SIGNAL( send_und_inside_points (v_points, bool,
        // float, float) ), und_imageLabel, SLOT( set_inside_points (v_points,
        // bool, float, float) ) );
        // connect(manager, SIGNAL( send_def_inside_points (v_points, bool,
        // float, float) ), def_imageLabel, SLOT( set_inside_points (v_points,
        // bool, float, float) ) );
        if (current_frame_order == 0 || referenceImage == refImage_Previous) {
          emit send_und_inside_points(results_in[iSector].und_inside_points,
                                      error);
        }
        emit send_def_inside_points(results_in[iSector].def_inside_points,
                                    error);
      }

      if (plot_contour_points) {
        results_in[iSector].def_contour = deformPoints(
            results_in[iSector].und_contour, results_in[iSector].und_center_x,
            results_in[iSector].und_center_y);

        // connect(manager, SIGNAL( send_und_contour_points(v_points, bool) ),
        // und_imageLabel, SLOT( set_contour_points(v_points, bool) ) );
        // connect(manager, SIGNAL( send_def_contour_points(v_points, bool) ),
        // def_imageLabel, SLOT( set_contour_points(v_points, bool) ) );
        if (current_frame_order == 0 || referenceImage == refImage_Previous) {
          emit send_und_contour_points(results_in[iSector].und_contour, error);
        }
        emit send_def_contour_points(results_in[iSector].def_contour, error);
      }

      if (stop_flag)
        break;
      if (error) {
        switch (processor) {
        case processor_CPU:
          errorType = correlator->get_error_code();
          break;

        case processor_GPU:
          // errorType is already read
          break;

        default:
          assert(false);
          break;
        }

        if (error_handling_mode == errorMode_stopAll ||
            error_handling_mode == errorMode_stopFrame)
          break;
      }

    } // This completes all the iterations of a correlation event

    if (stop_flag)
      break;
    if (error && (error_handling_mode == errorMode_stopAll ||
                  error_handling_mode == errorMode_stopFrame))
      break;
  }

  // Update the global correlation results, that averages through all sector's
  // results, like center.
  update_global_results(hs, vs, results_in);

  // This completes one frame worth of correlations
  return error;
}

bool managerClass::perform_single_frame_correlation_annular(
    int rs, int as, frame_results *results_in, int current_frame_order) {
  float ri = annularDomain.r_inside;
  float ro = annularDomain.r_outside;

  float cx = annularDomain.x_center;
  float cy = annularDomain.y_center;

  float dr = (ro - ri) / (float)rs;
  float da = 2.f * PI / (float)as;

  time_all_point_selection = 0;

  for (int i = 0; i < rs; ++i) {
    for (int j = 0; j < as; ++j) {
      int iSector = i * as + j;

      // Need to refresh r every sector because of the Lagrangian description
      float r; // inside radius of the sector
      float a; // angle of the sector

      // The undeformed domain is selected depending on the deformation
      // description mode and
      // whether we are in the first frame. Note "adjust_annular_domain" defines
      // r and a.
      adjust_annular_domain(i, j, r, a, ri, dr, da, cx, cy, as, results_in,
                            current_frame_order);
      // Adjust_initial_guess customizes global_initial_guess for this sector,
      // into initial_guess
      adjust_initial_guess(initial_guess, &results_in[iSector],
                           current_frame_order);

      //  Compute the undeformed points only on the first frame
      std::chrono::system_clock::time_point start_point_selection =
          std::chrono::system_clock::now();
      if (current_frame_order == 0) {
        results_in[iSector].und_contour =
            get_contour_points_annularDomain(r, dr, a, da, cx, cy);

        switch (processor) {
        case processor_CPU:
          results_in[iSector].und_inside_points =
              get_inside_points_annularDomain(r, dr, a, da, cx, cy, as);

          break;

        case processor_GPU:
#if CUDA_ENABLED
          cuda_manager->resetPolygon(iSector, r, dr, a, da, cx, cy, as);

          if (plot_inside_points) {
            results_in[i * as + j].und_inside_points =
                cuda_manager->getUndXY0ToCPU(iSector);
          }
#endif
          break;

        default:
          assert(false);
          break;
        }
      } else // Logic to get the undeformed points on frames other than the
             // first
      {
        switch (deformationDescription) {
        case def_Eulerian:
          //  Do nothing
          break;

        case def_strict_Lagrangian:
          results_in[iSector].und_inside_points =
              results_in[iSector].def_inside_points;
          results_in[iSector].und_contour = results_in[iSector].def_contour;
          if (processor == processor_GPU) {
#if CUDA_ENABLED
            cuda_manager->updatePolygon(iSector, def_strict_Lagrangian);
            if (plot_inside_points) {
              results_in[iSector].und_inside_points =
                  cuda_manager->getUndXY0ToCPU(iSector);
            }
#endif
          }
          break;

        case def_Lagrangian: {
          std::pair<float, float> center_offset =
              std::make_pair(results_in[iSector].und_center_x -
                                 results_in[iSector].past_und_center_x,

                             results_in[iSector].und_center_y -
                                 results_in[iSector].past_und_center_y);

          std::transform(results_in[iSector].und_contour.begin(),
                         results_in[iSector].und_contour.end(),
                         results_in[iSector].und_contour.begin(),
                         add_pair(center_offset));

          switch (processor) {
          case processor_CPU:
            std::transform(results_in[iSector].und_inside_points.begin(),
                           results_in[iSector].und_inside_points.end(),
                           results_in[iSector].und_inside_points.begin(),
                           add_pair(center_offset));
            break;

          case processor_GPU:
#if CUDA_ENABLED
            cuda_manager->updatePolygon(iSector, def_Lagrangian);
            if (plot_inside_points) {
              results_in[iSector].und_inside_points =
                  cuda_manager->getUndXY0ToCPU(iSector);
            }
#endif
            break;

          default:
            assert(false);
            break;
          }
        } break;

        default:
          assert(false);
          break;
        }
      }

      std::chrono::system_clock::time_point duration_point_selection =
          std::chrono::system_clock::now();
      time_all_point_selection +=
          (float)std::chrono::duration_cast<std::chrono::milliseconds>(
              duration_point_selection - start_point_selection)
              .count() /
          1000.f;
#if DEBUG_TIME_POINT_SELECTION
      std::cout << "manager()     :point selection wall execution time(s): "
                << time_all_point_selection << '\n';
#endif

      CorrelationResult *correlationResults = nullptr;

      switch (processor) {
      case processor_CPU:
        correlator->Newton_Raphson(
            initial_guess, (int)results_in[iSector].und_inside_points.size(),
            (float *)results_in[iSector].und_inside_points.data());

        error = correlator->get_error_status();
        break;

      case processor_GPU:
#if CUDA_ENABLED

        correlationResults = cuda_manager->correlate(iSector, initial_guess,
                                                     results_in[iSector]);
        errorType = correlationResults->errorCode;
        error = (errorType != error_none);

#endif
        break;

      default:
        assert(true);
        break;
      }

      // Updating center and other correlation results from the correlator
      update_results(&results_in[iSector], correlationResults);

      if (plot_inside_points ||
          deformationDescription == def_strict_Lagrangian) {
        switch (processor) {
        case processor_CPU:
          results_in[iSector].def_inside_points = correlator->getDefXY0();
          break;

        case processor_GPU:
#if CUDA_ENABLED
          if (plot_inside_points) {
            results_in[iSector].def_inside_points =
                cuda_manager->getDefXY0ToCPU(iSector);
          }
#endif
          break;

        default:
          assert(false);
          break;
        }
      }

      if (plot_inside_points) {
        // connect(manager, SIGNAL( send_und_inside_points (v_points, bool,
        // float, float) ), und_imageLabel, SLOT( iside_points (v_points, bool,
        // float, float) ) );
        // connect(manager, SIGNAL( send_def_inside_points (v_points, bool,
        // float, float) ), def_imageLabel, SLOT( set_inside_points (v_points,
        // bool, float, float) ) );
        if (current_frame_order == 0 || referenceImage == refImage_Previous) {
          emit send_und_inside_points(results_in[iSector].und_inside_points,
                                      error);
        }
        emit send_def_inside_points(results_in[iSector].def_inside_points,
                                    error);
      }

      if (plot_contour_points) {
        results_in[iSector].def_contour = deformPoints(
            results_in[iSector].und_contour, results_in[iSector].und_center_x,
            results_in[iSector].und_center_y, ro);

        // connect( manager , SIGNAL( send_und_contour_points( v_points , bool)
        // ), und_imageLabel, SLOT( set_contour_points( v_points , bool ) ) );
        // connect( manager , SIGNAL( send_def_contour_points( v_points , bool)
        // ), def_imageLabel, SLOT( set_contour_points( v_points , bool ) ) );
        if (current_frame_order == 0 || referenceImage == refImage_Previous) {
          emit send_und_contour_points(results_in[iSector].und_contour, error);
        }
        emit send_def_contour_points(results_in[iSector].def_contour, error);
      }

      if (stop_flag)
        break;
      if (error) {
        switch (processor) {
        case processor_CPU:
          errorType = correlator->get_error_code();
          break;

        case processor_GPU:
          // errorType is already read
          break;

        default:
          assert(false);
          break;
        }

        if (error_handling_mode == errorMode_stopAll ||
            error_handling_mode == errorMode_stopFrame)
          break;
      }
    } // sectors loop

    if (stop_flag)
      break;
    if (error && (error_handling_mode == errorMode_stopAll ||
                  error_handling_mode == errorMode_stopFrame))
      break;
  }

  // Update the global correlation results, that averages through all sector's
  // results, like center,
  // aggregate angle, and dilation parameter
  update_global_results(rs, as, results_in);

  return error;
}

v_points managerClass::get_inside_points_annularDomain(float r, float dr,
                                                       float a, float da,
                                                       float cx, float cy,
                                                       int as) {
#if DEBUG_MANAGER_INSIDE_POINTS
  printf("manager(): get_inside_points_annularDomain\n");
#endif

#if DEBUG_TIME_POINT_SELECTION
  std::chrono::system_clock::time_point start =
      std::chrono::system_clock::now();
#endif

  int x0, y0, x1, y1;

  float corner_00_x;
  float corner_01_x;
  float corner_10_x;
  float corner_11_x;

  float corner_00_y;
  float corner_01_y;
  float corner_10_y;
  float corner_11_y;

  float arc_x;
  float arc_y;

  v_points inside_points_vector;

  switch (as) {
  case 0:

    assert(false);

    break;

  case 1:

    x0 = cx - (r + dr);
    x1 = cx + (r + dr);
    y0 = cy - (r + dr);
    y1 = cy + (r + dr);

    break;

  default: {
    float sin0 = (float)sin(a);
    float cos0 = (float)cos(a);
    float sin1 = (float)sin(a + da);
    float cos1 = (float)cos(a + da);
    float sin2 = (float)sin(a + da / 2.f);
    float cos2 = (float)cos(a + da / 2.f);

    corner_00_x = cx + (r)*cos0;
    corner_01_x = cx + (r)*cos1;
    corner_10_x = cx + (r + dr) * cos0 * 1.2f; // Cheap sag
    corner_11_x = cx + (r + dr) * cos1 * 1.2f;

    corner_00_y = cy + (r)*sin0;
    corner_01_y = cy + (r)*sin1;
    corner_10_y = cy + (r + dr) * sin0 * 1.2f;
    corner_11_y = cy + (r + dr) * sin1 * 1.2f;

    arc_x = cx + (r + dr) * cos2;
    arc_y = cy + (r + dr) * sin2;

    x0 = std::min(arc_x, std::min(std::min(corner_00_x, corner_01_x),
                                  std::min(corner_10_x, corner_11_x)));
    x1 = std::max(arc_x, std::max(std::max(corner_00_x, corner_01_x),
                                  std::max(corner_10_x, corner_11_x)));

    y0 = std::min(arc_y, std::min(std::min(corner_00_y, corner_01_y),
                                  std::min(corner_10_y, corner_11_y)));
    y1 = std::max(arc_y, std::max(std::max(corner_00_y, corner_01_y),
                                  std::max(corner_10_y, corner_11_y)));

    break;
  }
  }

  float ro2 = (r + dr) * (r + dr);
  float ri2 = r * r;

  inside_points_vector.reserve(PI * (ro2 - ri2) / (float)as * 1.1f);

#pragma omp parallel
  for (float i = x0; i < x1; ++i) {
    v_points inside_points_vector_private;

#pragma omp for nowait
    for (int j = y0; j < y1; ++j) {
      float r2 = (i - cx) * (i - cx) + (j - cy) * (j - cy);
      if (r2 > ri2 && r2 < ro2) {
        float cross1 = (corner_11_x - i) * (corner_01_y - corner_11_y) -
                       (corner_11_y - j) * (corner_01_x - corner_11_x);
        float cross2 = (corner_00_x - i) * (corner_10_y - corner_00_y) -
                       (corner_00_y - j) * (corner_10_x - corner_00_x);

        if (cross1 * cross2 > 0 || as == 1) {
          inside_points_vector_private.push_back(std::make_pair(i, (float)j));
        }
      }
    }

#pragma omp critical
    inside_points_vector.insert(inside_points_vector.end(),
                                inside_points_vector_private.begin(),
                                inside_points_vector_private.end());
  }

#if DEBUG_TIME_POINT_SELECTION
  std::chrono::system_clock::time_point duration =
      std::chrono::system_clock::now();
  float total = (float)std::chrono::duration_cast<std::chrono::milliseconds>(
                    duration - start)
                    .count() /
                1000.f;
  std::cout << "manager()     :get_inside_points_annularDomain: wall execution "
               "time(s): "
            << total << '\n';
#endif

  return inside_points_vector;
}

v_points managerClass::get_contour_points_annularDomain(float r, float dr,
                                                        float a, float da,
                                                        float cx, float cy) {
  v_points contour;

  float sin0 = (float)sin(a);
  float cos0 = (float)cos(a);
  float sin1 = (float)sin(a + da);
  float cos1 = (float)cos(a + da);

  float corner_00_x = cx + (r)*cos0;
  float corner_01_x = cx + (r)*cos1;
  float corner_10_x = cx + (r + dr) * cos0;
  float corner_11_x = cx + (r + dr) * cos1;

  float corner_00_y = cy + (r)*sin0;
  float corner_01_y = cy + (r)*sin1;
  float corner_10_y = cy + (r + dr) * sin0;
  float corner_11_y = cy + (r + dr) * sin1;

  // The contour of the angular sector approximates the circular arcs as
  // polylines.
  // Every segment in the polyline is the smallest of 2 options. Either covering
  // a min arc angle or
  // having a length specified by the global blob segment distance.
  float angular_step_out =
      std::min(da / ((float)sqrt(min_blob_segment_squared) / (r + dr)),
               min_angular_step_deg * PI / 180.f);
  int number_of_intermediate_angular_steps_out =
      (int)floor(da / angular_step_out);

  float angular_step_in =
      std::min(da / ((float)sqrt(min_blob_segment_squared) / (r)),
               min_angular_step_deg * PI / 180.f);
  int number_of_intermediate_angular_steps_in =
      (int)floor(da / angular_step_in);

  contour.reserve(number_of_intermediate_angular_steps_out +
                  number_of_intermediate_angular_steps_in + 4);

  contour.push_back(std::make_pair(corner_00_x, corner_00_y));
  contour.push_back(std::make_pair(corner_10_x, corner_10_y));

  for (int i = 1; i <= number_of_intermediate_angular_steps_out; ++i) {
    contour.push_back(
        std::make_pair(cx + (r + dr) * cos(a + i * angular_step_out),
                       cy + (r + dr) * sin(a + i * angular_step_out)));
  }

  contour.push_back(std::make_pair(corner_11_x, corner_11_y));
  contour.push_back(std::make_pair(corner_01_x, corner_01_y));

  for (int i = number_of_intermediate_angular_steps_in; i >= 1; --i) {
    contour.push_back(std::make_pair(cx + (r)*cos(a + i * angular_step_in),
                                     cy + (r)*sin(a + i * angular_step_in)));
  }

  return contour;
}

bool managerClass::perform_single_frame_correlation_blob(
    frame_results *results_in, int current_frame_order) {
  int iSector = 0;

  bool error = false;

  auto start_time = std::chrono::system_clock::now();
  if (blobDomain.xy_contour.size() >= 3)
  // If the contour has 3 or more points (has a body) then find undeformed
  // domain,
  // initial guess and the inside points
  {
    adjust_blob_domain(&results_in[iSector], current_frame_order);
    adjust_initial_guess(initial_guess, &results_in[iSector],
                         current_frame_order);

    //  Compute the undeformed points only on the first frame
    if (current_frame_order == 0) {
      // Code that computes the blob inside points is disabled. Now it is done
      // with the polygon class
      // results_in[ 0 ].und_inside_points = get_inside_points_blobDomain(
      // results_in[ 0 ].und_contour );

      switch (processor) {
      case processor_CPU: {
        polygonBlob_class polygon(results_in[iSector].und_contour);
        error = polygon.getError();
        if (error) {
          errorType = error_bad_domain;
          return error;
        } else {
          results_in[iSector].und_inside_points = polygon.getInsidePoints();
        }
      } break;

      case processor_GPU:
#if CUDA_ENABLED
        cuda_manager->resetPolygon(results_in[iSector].und_contour);

        if (plot_inside_points) {
          results_in[0].und_inside_points =
              cuda_manager->getUndXY0ToCPU(iSector);
        }
#endif
        break;

      default:
        assert(false);
        break;
      }
    } else // Logic to get the undeformed points on frames other than the first
    {
      switch (deformationDescription) {
      case def_Eulerian:
        //  Do nothing
        break;

      case def_strict_Lagrangian:
        results_in[iSector].und_inside_points =
            results_in[iSector].def_inside_points;
        results_in[iSector].und_contour = results_in[iSector].def_contour;

        if (processor == processor_GPU) {
#if CUDA_ENABLED
          cuda_manager->updatePolygon(iSector, def_strict_Lagrangian);
          if (plot_inside_points) {
            results_in[iSector].und_inside_points =
                cuda_manager->getUndXY0ToCPU(iSector);
          }
#endif
        }
        break;

      case def_Lagrangian: {
        std::pair<float, float> center_offset =
            std::make_pair(results_in[iSector].und_center_x -
                               results_in[iSector].past_und_center_x,

                           results_in[iSector].und_center_y -
                               results_in[iSector].past_und_center_y);

        std::transform(results_in[iSector].und_contour.begin(),
                       results_in[iSector].und_contour.end(),
                       results_in[iSector].und_contour.begin(),
                       add_pair(center_offset));

        switch (processor) {
        case processor_CPU:
          std::transform(results_in[iSector].und_inside_points.begin(),
                         results_in[iSector].und_inside_points.end(),
                         results_in[iSector].und_inside_points.begin(),
                         add_pair(center_offset));
          break;

        case processor_GPU:
#if CUDA_ENABLED
          cuda_manager->updatePolygon(iSector, def_Lagrangian);
          if (plot_inside_points) {
            results_in[iSector].und_inside_points =
                cuda_manager->getUndXY0ToCPU(iSector);
          }
#endif
          break;

        default:
          assert(false);
          break;
        }
      } break;

      default:
        assert(false);
        break;
      }
    }
  } else {
    // assert( blobDomain.xy_contour.size() >= 3 );

    errorType = error_bad_domain;
    return true;
  }

  auto end_time = std::chrono::system_clock::now();
  float duration_point_selection =
      (float)std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                   start_time)
          .count() /
      1000000.f;
  time_all_point_selection += duration_point_selection;
#if DEBUG_TIME_POINT_SELECTION
  std::cout << "manager()     :point selection execution time(s): "
            << duration_point_selection << '\n';
#endif

  CorrelationResult *correlationResults = nullptr;

  // Call the correlator with these points
  switch (processor) {
  case processor_CPU:
    correlator->Newton_Raphson(
        initial_guess, (int)results_in[iSector].und_inside_points.size(),
        (float *)results_in[iSector].und_inside_points.data());

    error = correlator->get_error_status();
    break;

  case processor_GPU:
#if CUDA_ENABLED

    correlationResults =
        cuda_manager->correlate(iSector, initial_guess, results_in[iSector]);
    errorType = correlationResults->errorCode;
    error = (errorType != error_none);

#endif
    break;

  default:
    assert(false);
    break;
  }

  // Updating center and other correlation results from the correlator
  update_results(&results_in[iSector], correlationResults);

  // Update the global correlation results, that averages through all sector's
  // results, like center,
  // aggregate angle, and dilation parameter
  update_global_results(1, 1, results_in);

  if (plot_inside_points || deformationDescription == def_strict_Lagrangian) {
    switch (processor) {
    case processor_CPU:
      results_in[iSector].def_inside_points = correlator->getDefXY0();
      break;

    case processor_GPU:
#if CUDA_ENABLED
      if (plot_inside_points) {
        results_in[iSector].def_inside_points =
            cuda_manager->getDefXY0ToCPU(iSector);
      }
#endif
      break;

    default:
      assert(false);
      break;
    }
  }

  if (plot_inside_points) {
    // connect(manager, SIGNAL( send_und_inside_points (v_points, bool, float,
    // float) ), und_imageLabel, SLOT( set_inside_points (v_points, bool, float,
    // float) ) );
    // connect(manager, SIGNAL( send_def_inside_points (v_points, bool, float,
    // float) ), def_imageLabel, SLOT( set_inside_points (v_points, bool, float,
    // float) ) );
    if (current_frame_order == 0 || referenceImage == refImage_Previous) {
      emit send_und_inside_points(results_in[iSector].und_inside_points, error);
    }
    emit send_def_inside_points(results_in[iSector].def_inside_points, error);
  }

  if (plot_contour_points) {
    results_in[iSector].def_contour = deformPoints(
        results_in[iSector].und_contour, results_in[iSector].und_center_x,
        results_in[iSector].und_center_y);

    // connect(manager, SIGNAL( send_und_contour_points(v_points, bool) ),
    // und_imageLabel, SLOT( set_contour_points(v_points, bool) ) );
    // connect(manager, SIGNAL( send_def_contour_points(v_points, bool) ),
    // def_imageLabel, SLOT( set_contour_points(v_points, bool) ) );
    if (current_frame_order == 0 || referenceImage == refImage_Previous) {
      emit send_und_contour_points(results_in->und_contour, error);
    }
    emit send_def_contour_points(results_in->def_contour, error);
  }

  if (error)
    switch (processor) {
    case processor_CPU:
      errorType = correlator->get_error_code();
      break;

    case processor_GPU:
      // errorType is already read
      break;

    default:
      assert(false);
      break;
    }

  return error;
}

int managerClass::estimate_number_of_points(
    rectangularDomainStruct rectangularDomain_in) {
  int x1 = (int)rectangularDomain_in.x_end;
  int x0 = (int)rectangularDomain_in.x_begin;

  int y1 = (int)rectangularDomain_in.y_end;
  int y0 = (int)rectangularDomain_in.y_begin;

  int hs = rectangularDomain_in.horizontal_subdivisions;
  int vs = rectangularDomain_in.vertical_subdivisions;

  // Integer size required for correlation constructor
  int xdim = (abs(x1 - x0) / hs - 1) / 2;
  int ydim = (abs(y1 - y0) / vs - 1) / 2;

  //  compute size to allocate xy_positions for the hole frame using no marging
  //  since rectangular domains are always of same size.
  return (2 * xdim + 1) * (2 * ydim + 1);
}

int managerClass::estimate_number_of_points(
    annularDomainStruct annularDomain_in) {
  float ri = annularDomain_in.r_inside;
  float ro = annularDomain_in.r_outside;

  float as = (float)annularDomain_in.angular_subdivisions;
  float rs = (float)annularDomain_in.radial_subdivisions;

  float dr = (ro - ri) / rs;

  // Allocate enough memory to compute correlation for any sector. Allow a
  // growth factor of
  // 2 due to dilation. If more space is required, the correlator object takes
  // care of reallocating
  // memory.
  return PI * (ro * ro - (ro - dr) * (ro - dr)) / (float)as * 2.f;
}

int managerClass::estimate_number_of_points(blobDomainStruct blobDomain_in) {
  float x_max = blobDomain_in.xy_contour[0].first;
  float x_min = blobDomain_in.xy_contour[0].first;
  float y_max = blobDomain_in.xy_contour[0].second;
  float y_min = blobDomain_in.xy_contour[0].second;

  for (unsigned int p = 1; p < blobDomain_in.xy_contour.size(); ++p) {
    if (blobDomain_in.xy_contour[p].first > x_max)
      x_max = blobDomain_in.xy_contour[p].first;
    if (blobDomain_in.xy_contour[p].first < x_min)
      x_min = blobDomain_in.xy_contour[p].first;
    if (blobDomain_in.xy_contour[p].second > y_max)
      y_max = blobDomain_in.xy_contour[p].second;
    if (blobDomain_in.xy_contour[p].second < y_min)
      y_min = blobDomain_in.xy_contour[p].second;
  }

  return (x_max - x_min + 1.f) * (y_max - y_min + 1.f) * 1.2f;
}

void managerClass::perform_multiframe_correlation() {
#if DEBUG_TIME_MULTI_FRAME_CORRELATION
  std::chrono::system_clock::time_point chrono_start_multi_frame_correlation =
      std::chrono::system_clock::now();
#endif

  time_all_point_selection = 0.f;

  initializeReport();

  // Measure size to store a frame worth of results
  int results_i;
  int results_j;

  int number_of_points = 0;

  switch (domain_type) {
  case domain_rectangular:
    results_i = rectangularDomain.horizontal_subdivisions;
    results_j = rectangularDomain.vertical_subdivisions;
    number_of_points = estimate_number_of_points(rectangularDomain);
    break;

  case domain_annular:
    results_i = annularDomain.radial_subdivisions;
    results_j = annularDomain.angular_subdivisions;
    number_of_points = estimate_number_of_points(annularDomain);
    break;

  case domain_blob:
    results_i = 1;
    results_j = 1;
    number_of_points = estimate_number_of_points(blobDomain);
    break;

  default:
    assert(false);
    break;
  }

  switch (processor) {
  case processor_CPU:
    correlator =
        new CorrelationClass(number_of_points, number_of_colors, interpolation,
                             model, number_of_threads, precision, max_iters,
                             pyramid_start, pyramid_step, pyramid_stop);
    break;

  case processor_GPU:
#if CUDA_ENABLED
// GPU mode loads precision, max_iters, fitting model, interpolation model and
// number_of_color at mainApp::correlate.
//  Pyramid start, step and stop are loaded at mainApp::updateGPUPyramids
#endif
    break;

  default:
    assert(false);
    break;
  }

  // Allocating memory for the resulting correlation parameters for one frame (a
  // pair of images)
  frame_results *results = new frame_results[results_i * results_j];

  for (int i = 0; i < results_i; ++i) {
    for (int j = 0; j < results_j; ++j) {
      results[i * results_j + j].resulting_parameters =
          new float[number_of_model_parameters];
      results[i * results_j + j].initial_guess =
          new float[number_of_model_parameters];
      results[i * results_j + j].previous_resulting_parameters =
          new float[number_of_model_parameters];
      results[i * results_j + j].empty = true;

      for (int p = 0; p < number_of_model_parameters; ++p) {
        results[i * results_j + j].resulting_parameters[p] = 0.f;
        results[i * results_j + j].initial_guess[p] = 0.f;
        results[i * results_j + j].previous_resulting_parameters[p] = 0.f;
      }
    }
  }

  int total_frame_pairs = fileNames->size() - 1;
  for (int current_frame_order = 0; current_frame_order < total_frame_pairs;
       ++current_frame_order) {
    QString und_file_QString;
    QString def_file_QString;

    switch (referenceImage) {
    case refImage_First:
      und_file_QString = fileNames->at(0);
      break;

    case refImage_Previous:
      und_file_QString = fileNames->at(current_frame_order);
      break;

    default:
      assert(false);
      break;
    }

    und_file_string = und_file_QString.toStdString();
    def_file_QString = fileNames->at(current_frame_order + 1);
    def_file_string = def_file_QString.toStdString();

    // Set images. They are read from file on the first frame. Subsequently,
    // they are set from the next image
    set_undeformed_image(und_file_string, current_frame_order, referenceImage);
    set_deformed_image(def_file_string, current_frame_order);

    // connect( manager , SIGNAL( display_images(QString, QString) ), this,
    // SLOT( display_two_images( QString, QString ) ) );
    emit display_images(und_file_QString, def_file_QString);

    if (plot_inside_points) {
      //    connect(manager, SIGNAL( clear_und_inside_points () ),
      //    und_imageLabel, SLOT( clear_inside_points () ) );
      //    connect(manager, SIGNAL( clear_def_inside_points () ),
      //    def_imageLabel, SLOT( clear_inside_points () ) );
      if (current_frame_order == 0 || referenceImage == refImage_Previous) {
        emit clear_und_inside_points();
      }
      emit clear_def_inside_points();
    }

    if (plot_contour_points) {
      // connect(manager, SIGNAL( clear_und_contour_points() ), und_imageLabel,
      // SLOT( clear_contour_points() ) );
      // connect(manager, SIGNAL( clear_def_contour_points() ), def_imageLabel,
      // SLOT( clear_contour_points() ) );
      if (current_frame_order == 0 || referenceImage == refImage_Previous) {
        emit clear_und_contour_points();
      }
      emit clear_def_contour_points();
    }
#if DEBUG_TIME_FRAME_CORRELATION
    float start_frame_correlation = std::clock();
#endif

    std::future<bool> nextImageLoaderThread;

    // Load the next image asynchroniously with the correlation number crunching
    if (current_frame_order + 2 < fileNames->size()) {
      std::string next_file_string =
          fileNames->at(current_frame_order + 2).toStdString();
      nextImageLoaderThread =
          std::async(std::launch::async, &managerClass::set_next_image, this,
                     next_file_string);
    }

    switch (domain_type) {
    case domain_rectangular:
      perform_single_frame_correlation_rectangular(
          results_i, results_j, results, current_frame_order);

      break;

    case domain_annular:
      perform_single_frame_correlation_annular(results_i, results_j, results,
                                               current_frame_order);
      break;

    case domain_blob:
      perform_single_frame_correlation_blob(results, current_frame_order);
      break;

    default:
      assert(false);
      break;
    }

    if (current_frame_order + 2 < fileNames->size()) {
      if (nextImageLoaderThread.get()) {
        errorType = error_multiThread;
        error = true;
      }
    }

#if DEBUG_TIME_FRAME_CORRELATION
    float duration_frame_correlation =
        (std::clock() - start_frame_correlation) / (float)CLOCKS_PER_SEC;
    std::cout << "manager()     :Frame ";
    std::cout << current_frame_order << " correlation execution time(s): "
              << duration_frame_correlation << '\n';
#endif

    // Writes the text display in mainApp with parameters and angle per
    // connect(manager, SIGNAL( display_results( float , float* ) ), this ,
    // SLOT( updateResultDisplay( float , float* ) ) );
    emit display_results(results[0].def_global_angle, initial_guess);

    // Write report contribution from this frame
    addFrameToReport(current_frame_order, results_i, results_j, results);

    if (stop_flag || (error && error_handling_mode == errorMode_stopAll))
      break;

  } // for - loop all single-frame correlations

  if (processor == processor_CPU) {
    delete correlator;
    correlator = nullptr;
  }

  // After the report is written, delete the results
  for (int i = 0; i < results_i; ++i)
    for (int j = 0; j < results_j; ++j) {
      delete[] results[i * results_j + j].resulting_parameters;
      results[i * results_j + j].resulting_parameters = nullptr;

      delete[] results[i * results_j + j].initial_guess;
      results[i * results_j + j].initial_guess = nullptr;

      delete[] results[i * results_j + j].previous_resulting_parameters;
      results[i * results_j + j].previous_resulting_parameters = nullptr;
    }

  delete[] results;
  results = nullptr;

  // connect(manager, SIGNAL( correlation_is_done(bool) ), this, SLOT(
  // correlation_done(bool) ) );
  emit correlation_is_done(error);

#if DEBUG_TIME_MULTI_FRAME_CORRELATION
  std::chrono::system_clock::time_point
      chrono_duration_multi_frame_correlation =
          std::chrono::system_clock::now();
  std::cout << "manager()     :Multi-frame correlation wall execution time(s): "
            << (float)std::chrono::duration_cast<std::chrono::milliseconds>(
                   chrono_duration_multi_frame_correlation -
                   chrono_start_multi_frame_correlation)
                       .count() /
                   1000.f
            << '\n';
  fflush(stdout);
#endif

#if DEBUG_TIME_ALL_POINT_SELECTION
  std::cout << "manager()     :ALL point selection execution time(s): "
            << time_all_point_selection << '\n';
#endif
}

void managerClass::set_deformation_description(
    deformationDescriptionEnum deformationDescription_in) {
  deformationDescription = deformationDescription_in;
}

v_points managerClass::get_inside_points_blobDomain(v_points xy_contour_in) {
  // Find max and min xy
  float f_max_x = xy_contour_in[0].first;
  float f_min_x = xy_contour_in[0].first;
  float f_max_y = xy_contour_in[0].second;
  float f_min_y = xy_contour_in[0].second;

  for (unsigned int i = 1; i < xy_contour_in.size(); ++i) {
    if (xy_contour_in[i].first > f_max_x)
      f_max_x = xy_contour_in[i].first;
    if (xy_contour_in[i].first < f_min_x)
      f_min_x = xy_contour_in[i].first;
    if (xy_contour_in[i].second > f_max_y)
      f_max_y = xy_contour_in[i].second;
    if (xy_contour_in[i].second < f_min_y)
      f_min_y = xy_contour_in[i].second;
  }

  int max_x = (int)floor((double)f_max_x);
  int min_x = (int)ceil((double)f_min_x);
  int max_y = (int)floor((double)f_max_y);
  int min_y = (int)ceil((double)f_min_y);

  // Two different algorithms depending on convexity
  bool convex = check_polygon_convex(xy_contour_in);
  v_points inside_points;
  if (convex) {
    inside_points = get_inside_points_blobDomain_convex(xy_contour_in, max_x,
                                                        min_x, max_y, min_y);
  } else {
    inside_points = get_inside_points_blobDomain_non_convex(
        xy_contour_in, max_x, min_x, max_y, min_y);
  }

#if DEBUG_MANAGER_POINTS
  std::cout << "Convex = " << convex << std::endl;
  std::cout << "min_x = " << min_x << " max_x = " << max_x
            << " min_y = " << min_y << " max_y = " << max_y << std::endl;
  for (unsigned int i = 0; i < xy_contour_in.size(); ++i)
    std::cout << "Contour Point " << i << " = " << xy_contour_in[i].first << " "
              << xy_contour_in[i].second << std::endl;
  for (unsigned int i = 0; i < inside_points.size(); ++i)
    std::cout << "Body Point " << i << " = " << inside_points[i].first << " "
              << inside_points[i].second << std::endl;
#endif
  return inside_points;
}

v_points managerClass::get_inside_points_rectangularDomain(int x0, int y0,
                                                           int x1, int y1) {
#if DEBUG_MANAGER_INSIDE_POINTS
  printf("manager(): get_inside_points_rectangularDomain\n");
#endif

  unsigned int size = (x1 - x0) * (y1 - y0);

  v_points points;
  points.reserve(size);

  for (int ix = x0; ix <= x1; ++ix) {
    for (int iy = y0; iy <= y1; ++iy) {
      points.push_back(std::make_pair(ix, iy));
    }
  }

  return points;
}

v_points managerClass::get_contour_points_rectangularDomain(int x0, int y0,
                                                            int x1, int y1) {
  v_points contour;
  contour.reserve(4);

  contour.push_back(std::make_pair(x0, y0));
  contour.push_back(std::make_pair(x1, y0));
  contour.push_back(std::make_pair(x1, y1));
  contour.push_back(std::make_pair(x0, y1));

  return contour;
}

v_points managerClass::get_inside_points_blobDomain_non_convex(
    v_points xy_contour_in, int max_x, int min_x, int max_y, int min_y) {
#if DEBUG_MANAGER_INSIDE_POINTS
  printf("manager(): get_inside_points_blobDomain_non_convex\n");
#endif

  int size = (max_x - min_x) * (max_y - min_y);

  polyEquations contourLineEquations = makeLineEquations(xy_contour_in);

  v_points points;
  points.reserve(size);

#pragma omp parallel
  {
    v_points points_private;
    v_points xy_contour_private = xy_contour_in;
    polyEquations contourLineEquations_private = contourLineEquations;

#pragma omp for nowait
    for (int iy = min_y; iy <= max_y; ++iy) {
      for (int ix = min_x; ix <= max_x; ++ix) {
        if (check_inside_polygon(xy_contour_private, ix, iy,
                                 contourLineEquations_private)) {
          points_private.push_back(std::make_pair(ix, iy));
        }
      }
    }

#pragma omp critical
    points.insert(points.end(), points_private.begin(), points_private.end());
  }

  return points;
}

// Recursive function to get points inside a convex contour
v_points managerClass::get_inside_points_blobDomain_convex(
    v_points xy_contour_in, int max_x, int min_x, int max_y, int min_y) {
#if DEBUG_MANAGER_INSIDE_POINTS
  printf("manager(): get_inside_points_blobDomain_convex\n");
#endif

  v_points points;

  bool check_minx_miny = check_inside_polygon(xy_contour_in, min_x, min_y);

  // Check area is a single point
  if (min_x == max_x && min_y == max_y) {
    if (check_minx_miny) {
      points.push_back(std::make_pair(min_x, min_y));
    }

    return points;
  }

  bool check_maxx_miny = check_inside_polygon(xy_contour_in, max_x, min_y);

  // Check area is an horizontal line
  if (min_y == max_y) {
    if (check_minx_miny && check_maxx_miny) {
      points.reserve(max_x - min_x + 1);

      for (int ix = min_x; ix <= max_x; ++ix) {
        points.push_back(std::make_pair(ix, min_y));
      }
    } else {
      int x0 = min_x;
      int x1 = (min_x + max_x) / 2;
      int x2 = max_x;

      v_points quadrant1 = get_inside_points_blobDomain_convex(
          xy_contour_in, x1, x0, max_y, min_y);
      v_points quadrant2 = get_inside_points_blobDomain_convex(
          xy_contour_in, x2, x1 + 1, max_y, min_y);

      points.insert(points.end(), quadrant1.begin(), quadrant1.end());
      points.insert(points.end(), quadrant2.begin(), quadrant2.end());
    }

    return points;
  }

  bool check_minx_maxy = check_inside_polygon(xy_contour_in, min_x, max_y);

  // Check area is an vertical line
  if (min_x == max_x) {
    if (check_minx_miny && check_minx_maxy) {
      points.reserve(max_y - min_y + 1);

      for (int iy = min_y; iy <= max_y; ++iy) {
        points.push_back(std::make_pair(min_x, iy));
      }
    } else {
      int y0 = min_y;
      int y1 = (min_y + max_y) / 2;
      int y2 = max_y;

      v_points quadrant1 = get_inside_points_blobDomain_convex(
          xy_contour_in, max_x, min_x, y1, y0);
      v_points quadrant3 = get_inside_points_blobDomain_convex(
          xy_contour_in, max_x, min_x, y2, y1 + 1);

      points.insert(points.end(), quadrant1.begin(), quadrant1.end());
      points.insert(points.end(), quadrant3.begin(), quadrant3.end());
    }

    return points;
  }

  bool check_maxx_maxy = check_inside_polygon(xy_contour_in, max_x, max_y);

  // Check area is not a point nor a line
  if (check_minx_miny && check_minx_maxy && check_maxx_miny &&
      check_maxx_maxy) {
    points.reserve((max_x - min_x + 1) * (max_y - min_y + 1));

    for (int ix = min_x; ix <= max_x; ++ix) {
      for (int iy = min_y; iy <= max_y; ++iy) {
        points.push_back(std::make_pair(ix, iy));
      }
    }
  } else {
    int x0 = min_x;
    int x1 = (min_x + max_x) / 2;
    int x2 = max_x;
    int y0 = min_y;
    int y1 = (min_y + max_y) / 2;
    int y2 = max_y;

    v_points quadrant1 =
        get_inside_points_blobDomain_convex(xy_contour_in, x1, x0, y1, y0);
    v_points quadrant2 =
        get_inside_points_blobDomain_convex(xy_contour_in, x2, x1 + 1, y1, y0);
    v_points quadrant3 =
        get_inside_points_blobDomain_convex(xy_contour_in, x1, x0, y2, y1 + 1);
    v_points quadrant4 = get_inside_points_blobDomain_convex(
        xy_contour_in, x2, x1 + 1, y2, y1 + 1);

    points.insert(points.end(), quadrant1.begin(), quadrant1.end());
    points.insert(points.end(), quadrant2.begin(), quadrant2.end());
    points.insert(points.end(), quadrant3.begin(), quadrant3.end());
    points.insert(points.end(), quadrant4.begin(), quadrant4.end());
  }

  return points;
}

bool managerClass::check_polygon_convex(v_points xy_contour_in) {
  int n = (int)xy_contour_in.size();

  if (n < 4)
    return true;

  float dx1 = xy_contour_in[0].first - xy_contour_in[n - 1].first;
  float dy1 = xy_contour_in[0].second - xy_contour_in[n - 1].second;
  float dx2 = xy_contour_in[1].first - xy_contour_in[0].first;
  float dy2 = xy_contour_in[1].second - xy_contour_in[0].second;
  float zcrossproduct = dx1 * dy2 - dy1 * dx2;

  dx1 = xy_contour_in[n - 1].first - xy_contour_in[n - 2].first;
  dy1 = xy_contour_in[n - 1].second - xy_contour_in[n - 2].second;
  dx2 = xy_contour_in[0].first - xy_contour_in[n - 1].first;
  dy2 = xy_contour_in[0].second - xy_contour_in[n - 1].second;
  if (zcrossproduct * (dx1 * dy2 - dy1 * dx2) < 0.f)
    return false;

  for (int i = 2; i < n; ++i) {
    dx1 = xy_contour_in[i - 1].first - xy_contour_in[i - 2].first;
    dy1 = xy_contour_in[i - 1].second - xy_contour_in[i - 2].second;
    dx2 = xy_contour_in[i].first - xy_contour_in[i - 1].first;
    dy2 = xy_contour_in[i].second - xy_contour_in[i - 1].second;
    if (zcrossproduct * (dx1 * dy2 - dy1 * dx2) < 0.f)
      return false;
  }

  return true;
}

polyEquations managerClass::makeLineEquations(v_points xy_contour_in) {
  polyEquations contourLineEquations(xy_contour_in.size(),
                                     std::vector<float>(3, 0));

  for (unsigned int i = 0; i < xy_contour_in.size() - 1; ++i) {
    float v1x1 = xy_contour_in[i].first;
    float v1y1 = xy_contour_in[i].second;
    float v1x2 = xy_contour_in[i + 1].first;
    float v1y2 = xy_contour_in[i + 1].second;

    contourLineEquations[i][0] = v1y2 - v1y1;
    contourLineEquations[i][1] = v1x1 - v1x2;
    contourLineEquations[i][2] = (v1x2 * v1y1) - (v1x1 * v1y2);
  }

  float v1x1 = xy_contour_in[xy_contour_in.size() - 1].first;
  float v1y1 = xy_contour_in[xy_contour_in.size() - 1].second;
  float v1x2 = xy_contour_in[0].first;
  float v1y2 = xy_contour_in[0].second;

  contourLineEquations[xy_contour_in.size() - 1][0] = v1y2 - v1y1;
  contourLineEquations[xy_contour_in.size() - 1][1] = v1x1 - v1x2;
  contourLineEquations[xy_contour_in.size() - 1][2] =
      (v1x2 * v1y1) - (v1x1 * v1y2);

  return contourLineEquations;
}

bool managerClass::check_inside_polygon(const v_points &xy_contour_in, int ix,
                                        int iy) {
  int n = (int)xy_contour_in.size();
  int number_intersections = 0;

  // Horizontal line from outside image to test pixel ix,iy
  float v1x1 = -1.f;
  float v1y1 = (float)iy;
  float v1x2 = (float)ix;
  float v1y2 = (float)iy;

  float v2x1 = xy_contour_in[n - 1].first;
  float v2y1 = xy_contour_in[n - 1].second;
  float v2x2 = xy_contour_in[0].first;
  float v2y2 = xy_contour_in[0].second;

  if (check_segment_intersection(v1x1, v1y1, v1x2, v1y2, v2x1, v2y1, v2x2,
                                 v2y2) == intersection_yes) {
    number_intersections++;
  }

  for (int i = 1; i < n; ++i) {
    v2x1 = xy_contour_in[i - 1].first;
    v2y1 = xy_contour_in[i - 1].second;
    v2x2 = xy_contour_in[i].first;
    v2y2 = xy_contour_in[i].second;

    if (check_segment_intersection(v1x1, v1y1, v1x2, v1y2, v2x1, v2y1, v2x2,
                                   v2y2) == intersection_yes) {
      number_intersections++;
    }
  }

  // Point is inside if number of intersections is odd
  if (number_intersections % 2 != 0)
    return true;

  return false;
}

bool managerClass::check_inside_polygon(
    const v_points &xy_contour_in, int ix, int iy,
    const polyEquations &contourLineEquations) {
  int n = (int)contourLineEquations.size();
  int number_intersections = 0;

  // Horizontal line from outside image to test pixel ix,iy

  for (int i = 0; i < n; ++i) {
    float v2y1 = xy_contour_in[i].second;
    float v2y2 = xy_contour_in[(i < n - 1) ? (i + 1) : 0].second;

    if (check_segment_intersection((float)ix, (float)iy, v2y1, v2y2,
                                   contourLineEquations[i]) ==
        intersection_yes) {
      number_intersections++;
    }
  }

  // Point is inside if number of intersections is odd
  if (number_intersections % 2 != 0)
    return true;

  return false;
}

intersectionEnum
managerClass::check_segment_intersection(float v1x1, float v1y1, float v1x2,
                                         float v1y2, float v2x1, float v2y1,
                                         float v2x2, float v2y2) {
  // From
  // https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon

  float d1, d2;
  float a1, a2, b1, b2, c1, c2;

  // Convert vector 1 to a line (line 1) of infinite length.
  // We want the line in linear equation standard form: A*x + B*y + C = 0
  // See: http://en.wikipedia.org/wiki/Linear_equation
  a1 = v1y2 - v1y1;
  b1 = v1x1 - v1x2;
  c1 = (v1x2 * v1y1) - (v1x1 * v1y2);

  // Every point (x,y), that solves the equation above, is on the line,
  // every point that does not solve it, is not. The equation will have a
  // positive result if it is on one side of the line and a negative one
  // if is on the other side of it. We insert (x1,y1) and (x2,y2) of vector
  // 2 into the equation above.
  d1 = (a1 * v2x1) + (b1 * v2y1) + c1;
  d2 = (a1 * v2x2) + (b1 * v2y2) + c1;

  // If d1 and d2 both have the same sign, they are both on the same side
  // of our line 1 and in that case no intersection is possible. Careful,
  // 0 is a special case, that's why we don't test ">=" and "<=",
  // but "<" and ">".
  if (d1 > 0 && d2 > 0)
    return intersection_no;
  // JG -> Extend checking for vertex points. Only one of the vertex points
  // of the polygon segment is checked to avoid doble crossings at the
  // vertex
  // if ( d1 >= 0 && d2 >= 0 ) return intersection_no;
  if (d1 < 0 && d2 < 0)
    return intersection_no;

  // The fact that vector 2 intersected the infinite line 1 above doesn't
  // mean it also intersects the vector 1. Vector 1 is only a subset of that
  // infinite line 1, so it may have intersected that line before the vector
  // started or after it ended. To know for sure, we have to repeat the
  // the same test the other way round. We start by calculating the
  // infinite line 2 in linear equation standard form.
  a2 = v2y2 - v2y1;
  b2 = v2x1 - v2x2;
  c2 = (v2x2 * v2y1) - (v2x1 * v2y2);

  // Calculate d1 and d2 again, this time using points of vector 1.
  d1 = (a2 * v1x1) + (b2 * v1y1) + c2;
  d2 = (a2 * v1x2) + (b2 * v1y2) + c2;

  // Again, if both have the same sign (and neither one is 0),
  // no intersection is possible.
  if (d1 > 0 && d2 > 0)
    return intersection_no;
  if (d1 < 0 && d2 < 0)
    return intersection_no;

  // If we get here, only two possibilities are left. Either the two
  // vectors intersect in exactly one point or they are collinear, which
  // means they intersect in any number of points from zero to infinite.
  // if ( ( a1 * b2 ) - ( a2 * b1 ) == 0.0f ) return intersection_colliear;
  if (d1 == 0.f && d2 == 0.f)
    return intersection_colliear;

  // If they are not collinear, they must intersect in exactly one point.
  return intersection_yes;
}

intersectionEnum managerClass::check_segment_intersection(
    float v1x2, float v1y2, float v2y1, float v2y2,
    const std::vector<float> &lineEquation) {
  // From
  // https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon

  if (v2y1 > v1y2 && v2y2 > v1y2)
    return intersection_no;
  if (v2y1 < v1y2 && v2y2 < v1y2)
    return intersection_no;

  // The fact that vector 2 intersected the infinite line 1 above doesn't
  // mean it also intersects the vector 1. Vector 1 is only a subset of that
  // infinite line 1, so it may have intersected that line before the vector
  // started or after it ended. To know for sure, we have to repeat the
  // the same test the other way round. We start by calculating the
  // infinite line 2 in linear equation standard form.
  // a = lineEquation[ 0 ];
  // b = lineEquation[ 1 ];
  // c = lineEquation[ 2 ];

  // Calculate d1 and d2 again, this time using points of vector 1.
  // d1 = ( a2 * v1x1 ) + ( b2 * v1y1 ) + c2;
  // d2 = ( a2 * v1x2 ) + ( b2 * v1y2 ) + c2;
  float temp = lineEquation[1] * v1y2 + lineEquation[2];
  float d1 = -lineEquation[0] + temp; // v1x1 = -1 , v1y1 = v1y2
  float d2 = (lineEquation[0] * v1x2) + temp;

  // Again, if both have the same sign (and neither one is 0),
  // no intersection is possible.
  if (d1 > 0 && d2 > 0)
    return intersection_no;
  if (d1 < 0 && d2 < 0)
    return intersection_no;

  // If we get here, only two possibilities are left. Either the two
  // vectors intersect in exactly one point or they are collinear, which
  // means they intersect in any number of points from zero to infinite.
  // if ( ( a1 * b2 ) - ( a2 * b1 ) == 0.0f ) return intersection_colliear;
  if (d1 == 0.f && d2 == 0.f)
    return intersection_colliear;

  // If they are not collinear, they must intersect in exactly one point.
  return intersection_yes;
}

void managerClass::adjust_rectangular_domain(int &center_x, int &center_y,
                                             frame_results *previous_results,
                                             int current_frame_order) {
  // The "previous results" struct have not been set yet.
  if (current_frame_order == 0) {
    previous_results->und_global_center_x = rectangularDomain.x_center;
    previous_results->und_global_center_y = rectangularDomain.y_center;
    previous_results->und_global_angle = 0.f;
    previous_results->und_global_e = 0.f;

    previous_results->und_center_x = center_x;
    previous_results->und_center_y = center_y;
    previous_results->und_angle = 0.f;

    previous_results->empty = false;

    previous_results->past_und_center_x = previous_results->und_center_x;
    previous_results->past_und_center_y = previous_results->und_center_y;
  }
  // After the first Frame, we use the "pevious_results" to define the sector
  // domains.
  else {
    switch (deformationDescription) {
    // The undeformed domain is always the same in the Eulerian description
    case def_Eulerian: {
      // Nothing changes in th Eularian deformation description
      previous_results->und_global_center_x =
          previous_results->und_global_center_x;
      previous_results->und_global_center_y =
          previous_results->und_global_center_y;
      previous_results->und_global_angle = previous_results->und_global_angle;

      previous_results->und_center_x = previous_results->und_center_x;
      previous_results->und_center_y = previous_results->und_center_y;
      previous_results->und_angle = previous_results->und_angle;

      break;

    } // Eulerian description brackets

    // In Lagrangian mode, the correlation follows material points. So the
    // correlation of square
    // x is started where it ended up in the previous frame.
    case def_strict_Lagrangian:
    case def_Lagrangian: {

      previous_results->und_global_center_x =
          previous_results->def_global_center_x;
      previous_results->und_global_center_y =
          previous_results->def_global_center_y;
      previous_results->und_global_angle = previous_results->def_global_angle;

      previous_results->past_und_center_x = previous_results->und_center_x;
      previous_results->past_und_center_y = previous_results->und_center_y;

      previous_results->und_center_x = previous_results->def_center_x;
      previous_results->und_center_y = previous_results->def_center_y;

      previous_results->und_angle = previous_results->def_angle;

      break;

    } // Lagrangian description brackets
    default:
      assert(false);
      break;

    } // deformation description switch brackets
  }

  center_x = (int)(previous_results->und_center_x + 0.5f);
  center_y = (int)(previous_results->und_center_y + 0.5f);
}

void managerClass::adjust_annular_domain(int i, int j, float &r, float &a,
                                         float &ri, float dr, float da,
                                         float &center_x, float &center_y,
                                         int as,
                                         frame_results *all_previous_results,
                                         int current_frame_order) {
  // This method modifies the annular domain in a cohesive way. New center for
  // the annulus is computed.
  if (current_frame_order == 0) {
    all_previous_results[i * as + j].und_global_center_x =
        annularDomain.x_center;
    all_previous_results[i * as + j].und_global_center_y =
        annularDomain.y_center;

    all_previous_results[i * as + j].und_global_ro = annularDomain.r_outside;
    all_previous_results[i * as + j].und_global_ri = annularDomain.r_inside;
    all_previous_results[i * as + j].und_global_e =
        0.f; // first frames assumes zero a and e.
    all_previous_results[i * as + j].und_global_angle = 0.f;

    // temporary value to compute sector initial_guess. After correlation, value
    // gets updated with values from
    // correlator object
    if (as > 1) // 2 or more angular subdivisions
    {
      float center_angle = 0 + j * da + da / 2.f;
      float center_r = ri + i * dr + dr / 2.f;

      all_previous_results[i * as + j].und_center_x =
          all_previous_results[i * as + j].und_global_center_x +
          center_r * (float)cos((double)center_angle);
      all_previous_results[i * as + j].und_center_y =
          all_previous_results[i * as + j].und_global_center_y +
          center_r * (float)sin((double)center_angle);
    }
    //  If we only have one angular subdivision, the local center is the same as
    //  the global center
    else {
      all_previous_results[i * as + j].und_center_x =
          all_previous_results[i * as + j].und_global_center_x;
      all_previous_results[i * as + j].und_center_y =
          all_previous_results[i * as + j].und_global_center_y;
    }

    all_previous_results[i * as + j].past_und_center_x =
        all_previous_results[i * as + j].und_center_x;
    all_previous_results[i * as + j].past_und_center_y =
        all_previous_results[i * as + j].und_center_y;

    all_previous_results[i * as + j].und_e =
        0.f; // first frames assumes zero a and e.
    all_previous_results[i * as + j].und_angle = 0.f;

    all_previous_results[i * as + j].empty = false;
  }
  //  Frame orders other than the first
  else {
    switch (deformationDescription) {
    // The undeformed domain is always the same in the Eulerian description
    case def_Eulerian: {

      all_previous_results[i * as + j].und_global_center_x =
          all_previous_results[i * as + j].und_global_center_x;
      all_previous_results[i * as + j].und_global_center_y =
          all_previous_results[i * as + j].und_global_center_y;

      all_previous_results[i * as + j].und_global_ro =
          all_previous_results[i * as + j].und_global_ro;
      all_previous_results[i * as + j].und_global_ri =
          all_previous_results[i * as + j].und_global_ri;
      all_previous_results[i * as + j].und_global_e =
          all_previous_results[i * as + j].und_global_e;
      all_previous_results[i * as + j].und_global_angle =
          all_previous_results[i * as + j].und_global_angle;

      // temporary value to compute sector initial_guess. After correlation,
      // value gets updated with values from
      // correlator object
      all_previous_results[i * as + j].und_center_x =
          all_previous_results[i * as + j].und_center_x;
      all_previous_results[i * as + j].und_center_y =
          all_previous_results[i * as + j].und_center_y;

      all_previous_results[i * as + j].und_e =
          all_previous_results[i * as + j].und_e;
      all_previous_results[i * as + j].und_angle =
          all_previous_results[i * as + j].und_angle;

      break;

    } // Eulerian description brackets

    // In Lagrangian mode, the correlation follows material points. So the
    // correlation of the blob
    // starts where it ended up in the previous frame.
    case def_strict_Lagrangian:
    case def_Lagrangian: {
      all_previous_results[i * as + j].und_global_center_x =
          all_previous_results[i * as + j].def_global_center_x;
      all_previous_results[i * as + j].und_global_center_y =
          all_previous_results[i * as + j].def_global_center_y;

      all_previous_results[i * as + j].und_global_ro =
          all_previous_results[i * as + j].def_global_ro;
      all_previous_results[i * as + j].und_global_ri =
          all_previous_results[i * as + j].def_global_ri;
      all_previous_results[i * as + j].und_global_e =
          all_previous_results[i * as + j].def_global_e;
      all_previous_results[i * as + j].und_global_angle =
          all_previous_results[i * as + j].def_global_angle;

      // temporaty value to compute sector initial_guess. After correlation,
      // value gets updated with values from
      // correlator object
      all_previous_results[i * as + j].past_und_center_x =
          all_previous_results[i * as + j].und_center_x;
      all_previous_results[i * as + j].past_und_center_y =
          all_previous_results[i * as + j].und_center_y;

      all_previous_results[i * as + j].und_center_x =
          all_previous_results[i * as + j].def_center_x;
      all_previous_results[i * as + j].und_center_y =
          all_previous_results[i * as + j].def_center_y;

      all_previous_results[i * as + j].und_e =
          all_previous_results[i * as + j].def_e;
      all_previous_results[i * as + j].und_angle =
          all_previous_results[i * as + j].def_angle;

      break;

    } // Lagrangian description brackets

    default:
      assert(false);
      break;

    } // deformation description switch brackets
  }
  center_x = all_previous_results[i * as + j].und_global_center_x;
  center_y = all_previous_results[i * as + j].und_global_center_y;

  // Base polar coordinates
  r = ri + i * dr;
  a = all_previous_results[i * as + j].und_global_angle + j * da;
}

void managerClass::adjust_blob_domain(frame_results *previous_results,
                                      int current_frame_order) {
  // The "previous results" struct have not been set yet.
  if (current_frame_order == 0) {
    previous_results->und_global_center_x = blobDomain.x_center;
    previous_results->und_global_center_y = blobDomain.y_center;
    previous_results->und_global_angle = 0.f;

    previous_results->und_center_x = blobDomain.x_center;
    previous_results->und_center_y = blobDomain.y_center;
    previous_results->past_und_center_x = previous_results->und_center_x;
    previous_results->past_und_center_y = previous_results->und_center_y;
    previous_results->und_angle = 0.f;

    previous_results->und_contour = blobDomain.xy_contour;

    previous_results->empty = false;
  }
  // After the first Frame, we use the "pevious_results" to define the blob
  // domains.
  else {
    switch (deformationDescription) {
    // The undeformed domain is always the same in the Eulerian description
    case def_Eulerian: {

      previous_results->und_global_center_x =
          previous_results->und_global_center_x;
      previous_results->und_global_center_y =
          previous_results->und_global_center_y;
      previous_results->und_global_angle = previous_results->und_global_angle;

      previous_results->und_center_x = previous_results->und_center_x;
      previous_results->und_center_y = previous_results->und_center_y;
      previous_results->und_angle = previous_results->und_angle;

      previous_results->und_contour = previous_results->und_contour;

      break;

    } // Eulerian description brackets

    // In Lagrangian mode, the correlation follows material points. So the
    // correlation of the blob
    // starts where it ended up in the previous frame.
    case def_strict_Lagrangian:
    case def_Lagrangian: {

      previous_results->und_global_center_x =
          previous_results->def_global_center_x;
      previous_results->und_global_center_y =
          previous_results->def_global_center_y;
      previous_results->und_global_angle = previous_results->def_global_angle;

      previous_results->past_und_center_x = previous_results->und_center_x;
      previous_results->past_und_center_y = previous_results->und_center_y;

      previous_results->und_center_x = previous_results->def_center_x;
      previous_results->und_center_y = previous_results->def_center_y;

      previous_results->und_angle = previous_results->def_angle;

      break;

    } // Lagrangian description brackets
    default:
      assert(false);
      break;

    } // deformation description switch brackets

  } // else condition: frames other than the first brackets
}

void managerClass::update_results(frame_results *results_in,
                                  CorrelationResult *correlationResult) {
  // Updating the results array to keep track of last frames results

  // Storing the correlation paramenters
  float *temp_model_parameters;

  switch (processor) {
  case processor_CPU:
    temp_model_parameters = correlator->get_model_parameters();

    results_in->chi = correlator->get_chi();
    results_in->number_of_points = correlator->get_number_of_points();
    results_in->iterations = correlator->get_iterations();
    results_in->error_status = correlator->get_error_status();
    results_in->error_code = correlator->get_error_code();
    results_in->und_center_x = correlator->get_und_x_center();
    results_in->und_center_y = correlator->get_und_y_center();

    break;

  case processor_GPU:
#if CUDA_ENABLED
    temp_model_parameters = correlationResult->resultingParameters;

    results_in->chi = correlationResult->chi;
    results_in->number_of_points = correlationResult->numberOfPoints;
    results_in->iterations = correlationResult->iterations;
    results_in->error_code = correlationResult->errorCode;
    results_in->error_status = (results_in->error_code != error_none);
    results_in->und_center_x = correlationResult->undCenterX;
    results_in->und_center_y = correlationResult->undCenterY;
#endif
    break;

  default:
    assert(false);
    break;
  }

  memcpy(results_in->resulting_parameters, temp_model_parameters,
         number_of_model_parameters);

  memcpy(initial_guess, temp_model_parameters, number_of_model_parameters);

  for (int p = 0; p < number_of_model_parameters; ++p) {
    results_in->resulting_parameters[p] = temp_model_parameters[p];
  }

  // Define the pointer to the transformation model
  float (*distortX)(float, float, float, float, float, float *);
  float (*distortY)(float, float, float, float, float, float *);

  switch (model) {
  case fm_U:
    distortX = modelU_distort_x;
    distortY = modelU_distort_y;
    results_in->def_angle = 0.f;
    results_in->def_e = 0.f;
    break;

  case fm_UV:
    distortX = modelUV_distort_x;
    distortY = modelUV_distort_y;
    results_in->def_angle = 0.f;
    results_in->def_e = 0.f;
    break;

  case fm_UVQ:
    distortX = modelUVQ_distort_x;
    distortY = modelUVQ_distort_y;
    results_in->def_angle =
        results_in->resulting_parameters[2] + results_in->und_angle;
    results_in->def_e = 0.f;
    break;

  case fm_UVUxUyVxVy:
    distortX = modelUVUxUyVxVy_distort_x;
    distortY = modelUVUxUyVxVy_distort_y;
    results_in->def_angle =
        best_rotation_UVUxUyVxVy(results_in->resulting_parameters) +
        results_in->und_angle;
    results_in->def_e = 0.f;
    break;

  default:
    assert(false);
    break;
  }

  // Short names to variables that are required for the computation of the
  // sectors centers.
  float x = results_in->und_center_x;
  float y = results_in->und_center_y;
  float cx = results_in->und_global_center_x;
  float cy = results_in->und_global_center_y;
  float ro = results_in->und_global_ro;

  results_in->def_center_x =
      distortX(x, y, x, y, ro, results_in->resulting_parameters);
  results_in->def_center_y =
      distortY(x, y, x, y, ro, results_in->resulting_parameters);

  // apply deformation to the contour
  results_in->def_contour.clear();
  results_in->def_contour.reserve(results_in->und_contour.size());

  for (unsigned int i = 0; i < results_in->und_contour.size(); ++i) {
    float def_x = distortX(results_in->und_contour[i].first,
                           results_in->und_contour[i].second, cx, cy, ro,
                           results_in->resulting_parameters);
    float def_y = distortY(results_in->und_contour[i].first,
                           results_in->und_contour[i].second, cx, cy, ro,
                           results_in->resulting_parameters);
    results_in->def_contour.push_back(std::make_pair(def_x, def_y));
  }
}

void managerClass::addFrameToReport(int frame_number_in, int results_i_in,
                                    int results_j_in,
                                    frame_results *results_in) {
  for (int i = 0; i < results_i_in; ++i)
    for (int j = 0; j < results_j_in; ++j) {
      report << frame_number_in << "," << und_file_string << ","
             << def_file_string << ","

             << results_in[i * results_j_in + j].und_global_center_x << ","
             << results_in[i * results_j_in + j].und_global_center_y << ","

             << results_in[i * results_j_in + j].und_center_x << ","
             << results_in[i * results_j_in + j].und_center_y << ","

             << results_in[i * results_j_in + j].def_global_center_x << ","
             << results_in[i * results_j_in + j].def_global_center_y << ","
             << results_in[i * results_j_in + j].def_center_x << ","
             << results_in[i * results_j_in + j].def_center_y << ",";

      for (int p = 0; p < number_of_model_parameters; ++p) {
        report << results_in[i * results_j_in + j].resulting_parameters[p]
               << ",";
      }

      for (int p = 0; p < number_of_model_parameters; ++p) {
        report << results_in[i * results_j_in + j].initial_guess[p] << ",";
      }

      report << results_in[i * results_j_in + j].und_global_angle << ","
             << results_in[i * results_j_in + j].def_global_angle << ","
             << results_in[i * results_j_in + j].und_angle << ","
             << results_in[i * results_j_in + j].def_angle << ","
             << results_in[i * results_j_in + j].def_angle * 180 / PI << ",";
      ;

      report << results_in[i * results_j_in + j].chi << ","
             << results_in[i * results_j_in + j].number_of_points << ","
             << results_in[i * results_j_in + j].iterations << ","
             << results_in[i * results_j_in + j].error_status << ","
             << results_in[i * results_j_in + j].error_code << std::endl;
    }
}

void managerClass::initializeReport() {
  report.str("");

  report << "Frame#"
         << ","
         << "und_file_string"
         << ","
         << "def_file_string"
         << ","
         << "und_global_center_x"
         << ","
         << "und_global_center_y"
         << ","
         << "und_center_x"
         << ","
         << "und_center_y"
         << ","
         << "def_global_center_x"
         << ","
         << "def_global_center_y"
         << ","
         << "def_center_x"
         << ","
         << "def_center_y"
         << ",";

  for (int p = 0; p < number_of_model_parameters; ++p)
    report << "parameter_" << p << ",";

  for (int p = 0; p < number_of_model_parameters; ++p)
    report << "Initial_guess_" << p << ",";

  report << "und_global_angle(rad)"
         << ","
         << "def_global_angle(rad)"
         << ","
         << "und_angle(rad)"
         << ","
         << "def_angle(rad)"
         << ","
         << "def_angle(deg)"
         << ",";

  report << "chi"
         << ","
         << "number_of_points"
         << ","
         << "iterations"
         << ","
         << "error_status"
         << ","
         << "error_code" << std::endl;
}

v_points managerClass::deformPoints(v_points contour, float cx, float cy,
                                    float ro) {
  v_points deformed_contour;

  float *model_parameters;

  switch (processor) {
  case processor_CPU:
    model_parameters = correlator->get_model_parameters();
    break;

  case processor_GPU:
#if CUDA_ENABLED
    model_parameters = initial_guess;

#endif
    break;

  default:
    assert(false);
    break;
  }

  float (*def_x)(float, float, float, float, float, float *);
  float (*def_y)(float, float, float, float, float, float *);

  switch (model) {
  case fm_U:

    def_x = modelU_distort_x;
    def_y = modelU_distort_y;

    break;

  case fm_UV:

    def_x = modelUV_distort_x;
    def_y = modelUV_distort_y;

    break;

  case fm_UVQ:

    def_x = modelUVQ_distort_x;
    def_y = modelUVQ_distort_y;

    break;

  case fm_UVUxUyVxVy:

    def_x = modelUVUxUyVxVy_distort_x;
    def_y = modelUVUxUyVxVy_distort_y;

    break;

  default:

    assert(false);

    break;
  }

  deformed_contour.reserve(contour.size());

  for (unsigned int i = 0; i < contour.size(); ++i) {
    float x = def_x(contour[i].first, contour[i].second, cx, cy, ro,
                    model_parameters);
    float y = def_y(contour[i].first, contour[i].second, cx, cy, ro,
                    model_parameters);
    deformed_contour.push_back(std::make_pair(x, y));
  }

  return deformed_contour;
}

void managerClass::adjust_initial_guess(float *initial_guess,
                                        frame_results *previous_results,
                                        int current_frame_order) {
  // This method assumes a constant rate of deformation between image pairs,
  // so that the previously resulting deformation is a good guess for the next

  // In the first frame, the global initial guess is customized for the sector
  if (current_frame_order == 0) {
    switch (model) {
    case fm_U:
    case fm_UV:
    case fm_UVQ: {
      // Sectors on the UVQ models have the global initial guess plus the strain
      // contribution.
      copy_model_parameters(global_initial_guess,
                            previous_results->initial_guess,
                            number_of_model_parameters);

      float dx = previous_results->und_center_x -
                 previous_results->und_global_center_x;
      float dy = previous_results->und_center_y -
                 previous_results->und_global_center_y;

      float Vx = global_initial_guess[2];

      previous_results->initial_guess[0] += -dy * Vx;
      previous_results->initial_guess[1] += dx * Vx;

      break;
    }

    case fm_UVUxUyVxVy: {
      // Sectors on the UVUxUyVxVy models have the global initial guess plus the
      // strain contribution.
      copy_model_parameters(global_initial_guess,
                            previous_results->initial_guess,
                            number_of_model_parameters);

      float dx = previous_results->und_center_x -
                 previous_results->und_global_center_x;
      float dy = previous_results->und_center_y -
                 previous_results->und_global_center_y;

      float Ux = global_initial_guess[2];
      float Uy = global_initial_guess[3];
      float Vx = global_initial_guess[4];
      float Vy = global_initial_guess[5];

      previous_results->initial_guess[0] += dx * Ux + dy * Uy;
      previous_results->initial_guess[1] += dx * Vx + dy * Vy;

      break;
    }

    default:

      break;

    } // model switch brackets

    // To enable the Eulerian dIG approach
    copy_model_parameters(previous_results->initial_guess,
                          previous_results->previous_resulting_parameters,
                          number_of_model_parameters);

  } // if ( current_frame_order == 0 )

  // else case applies to frames other than the first. The relsuts from the
  // previous frame are a good guess for
  // the next frame.
  else {
    // Eulerian descriptions with a first image reference progresively increment
    // the resulting parameters.
    // Under the assumption of constant rate of deformetion, we can bias the
    // initial guess to account for this
    if (deformationDescription == def_Eulerian &&
        referenceImage == refImage_First) {

      for (int i = 0; i < number_of_model_parameters; ++i) {
        previous_results->initial_guess[i] =
            previous_results->resulting_parameters[i] +
            (previous_results->resulting_parameters[i] -
             previous_results->previous_resulting_parameters[i]);
      }

    } // if deformation and ref bracket
    else {

      for (int i = 0; i < number_of_model_parameters; ++i) {
        previous_results->initial_guess[i] =
            previous_results->resulting_parameters[i];
      }
    }

    // To mantain the Eulerian dIG approach
    copy_model_parameters(previous_results->resulting_parameters,
                          previous_results->previous_resulting_parameters,
                          number_of_model_parameters);
  }

  // For all types of frames, first and other, copy the computed initial guess
  // (stored in the results) into the
  // active variable "initial_guess" read by the correlator.
  copy_model_parameters(previous_results->initial_guess, initial_guess,
                        number_of_model_parameters);
}

void managerClass::update_global_results(int hrs, int vas,
                                         frame_results *all_results_in) {
  float average_angle = 0.f;
  float average_center_x = 0.f;
  float average_center_y = 0.f;
  float average_e = 0.f;

  float total_n = 0.f;

  for (int i = 0; i < hrs; ++i) {
    for (int j = 0; j < vas; ++j) {
      float n = all_results_in[i * vas + j].number_of_points;

      average_angle += all_results_in[i * vas + j].def_angle * n;
      average_center_x += all_results_in[i * vas + j].def_center_x * n;
      average_center_y += all_results_in[i * vas + j].def_center_y * n;
      average_e += all_results_in[i * vas + j].def_e * n;

      total_n += n;
    }
  }

  average_angle = average_angle / total_n;
  average_center_x = average_center_x / total_n;
  average_center_y = average_center_y / total_n;
  average_e = average_e / total_n;

  float und_ro = all_results_in[0].und_global_ro;
  float und_ri = all_results_in[0].und_global_ri;

  float def_ri = 1.f + average_e * (und_ro / und_ri - 1.f);

  for (int i = 0; i < hrs; ++i) {
    for (int j = 0; j < vas; ++j) {
      all_results_in[i * vas + j].def_global_angle = average_angle;
      all_results_in[i * vas + j].def_global_center_x = average_center_x;
      all_results_in[i * vas + j].def_global_center_y = average_center_y;
      all_results_in[i * vas + j].def_global_e = average_e;

      all_results_in[i * vas + j].def_global_ro =
          all_results_in[i * vas + j].und_global_ro;
      all_results_in[i * vas + j].def_global_ri = def_ri;
    }
  }
}

void managerClass::set_number_of_threads(int number_of_threads_in) {
  number_of_threads = number_of_threads_in;
}

void managerClass::set_pyramid(int pyramid_start_in, int pyramid_step_in,
                               int pyramid_stop_in) {
  pyramid_start = pyramid_start_in;
  pyramid_step = pyramid_step_in;
  pyramid_stop = pyramid_stop_in;
}

errorEnum managerClass::get_errorType() { return errorType; }
