// local includes
#include "correlation_class.hpp"

// COMPUTATION RECIPE FOR EACH THREAD
//	1) perform the transformation by the corresponding modeler
// 	2) interpolate the resulting transformed location in the deformed
//		image to get its intensity (w) and gradients (dwdxy)
//  3) Assembler puts togeter the thread contribution to the mat_A and vec_B
//  -coming soon
// void *h_interpolation( void *correlation_thread_data_in )
s_correlation_thread_data
h_interpolation(s_correlation_thread_data thread_data) {
  thread_data.modeler->compute_model();
  thread_data.model_time = thread_data.modeler->get_time();

  if (thread_data.modeler->get_error_status()) {
    thread_data.error_code = thread_data.modeler->get_error_code();
    thread_data.error_status = true;
  }

  if (thread_data.runInterpolation) {
    thread_data.interpolator->get_multiple_interpolations();
    thread_data.interpolation_time = thread_data.interpolator->get_time();

    thread_data.contribution_mat_A = thread_data.interpolator->get_mat_A();
    thread_data.contribution_vec_B = thread_data.interpolator->get_vec_B();
    thread_data.contribution_chi = thread_data.interpolator->get_chi();

    if (thread_data.interpolator->get_error_status()) {
      thread_data.error_code = thread_data.interpolator->get_error_code();
      thread_data.error_status = true;
    }
  }

  return thread_data;
}

void CorrelationClass::set_undeformed_image(cv::Mat &und_image_in) {
  und_image = und_image_in;
  pyramid.set_und_image(und_image);
}

void CorrelationClass::set_deformed_image(cv::Mat &def_image_in) {
  def_image = def_image_in;
  pyramid.set_def_image(def_image);
}

void CorrelationClass::set_next_image(cv::Mat &nxt_image_in) {
  nxt_image = nxt_image_in;
  pyramid.set_nxt_image(nxt_image);
}

void CorrelationClass::set_und_image_from_def() {
  und_image = def_image;
  pyramid.und_from_def();
}

void CorrelationClass::set_def_image_from_nxt() {
  def_image = nxt_image;
  pyramid.def_from_nxt();
}

void CorrelationClass::initialize_correlation() {
  error_status = false;
  set_center_flag = true;
  error_code = error_none;

  number_of_model_parameters =
      ModelClass::get_number_of_model_parameters(fittingModel);
  pyramid.set_number_of_model_parameters(number_of_model_parameters);

  allocate_point_dependent_arrays();

  mat_A = new float[number_of_model_parameters * number_of_model_parameters];
  vec_B = new float[number_of_model_parameters];
  solution = new float[number_of_model_parameters];
  vec_x = new float[number_of_model_parameters];

  // Factory implementation to create the selected interpolation objects, one
  // per thread
  interpolators = InterpolationClass::new_InterpolationClass(
      interpolationModel, fittingModel, number_of_colors, number_of_threads);

  // Factory implementation to create the selected model objects, one per thread
  modelers = ModelClass::new_ModelClass(fittingModel, number_of_threads);
}

void CorrelationClass::allocate_point_dependent_arrays() {
  def_xy_positions = new float[allocated_points * 2];
  w_results = new float[allocated_points * number_of_colors];
  und_intensities = new float[allocated_points * number_of_colors];
  dwdxy_results = new float[allocated_points * number_of_colors * 2];
  dTxydp = new float[allocated_points * number_of_model_parameters * 2];
}

void CorrelationClass::delete_point_dependent_arrays() {
  // und_xy_positions are deleted by the manager
  delete[] def_xy_positions;
  def_xy_positions = nullptr;

  delete[] und_intensities;
  und_intensities = nullptr;

  delete[] w_results;
  w_results = nullptr;

  delete[] dwdxy_results;
  dwdxy_results = nullptr;

  delete[] dTxydp;
  dTxydp = nullptr;
}

CorrelationClass::~CorrelationClass() {
  delete_point_dependent_arrays();

  delete[] modelers;
  delete[] mat_A;
  delete[] vec_B;
  delete[] solution;
  delete[] vec_x;

  for (int i = 0; i < number_of_threads; ++i) // interpolators is a vector of
                                              // dynamically allocated
                                              // InterpolationClass
  {
    delete interpolators[i];
  }
}

void CorrelationClass::apply_model_and_interpolate(int pyramid_level,
                                                   bool runInterpolation) {
#if DEBUG_TIME_MODEL_AND_INTERPOLATION
  auto start_model_and_interpolation = std::clock();
#endif

  correlation_thread_data = new s_correlation_thread_data[number_of_threads];
  std::vector<std::future<s_correlation_thread_data>> correlationThreads;

  // define number of points each thread is going to model and interpolate
  int number_of_points = pyramid.get_number_of_points(pyramid_level);
  float *und_xy_positions = pyramid.get_xy_positions(pyramid_level);
  float und_x_center, und_y_center;
  pyramid.get_und_center(und_x_center, und_y_center, pyramid_level);

  unsigned char *und_image_ptr = pyramid.get_und_ptr(pyramid_level);
  int und_image_step = pyramid.get_step(pyramid_level, imageType_und);

  unsigned char *def_image_ptr = pyramid.get_def_ptr(pyramid_level);
  int def_image_rows = pyramid.get_rows(pyramid_level, imageType_def);
  int def_image_cols = pyramid.get_cols(pyramid_level, imageType_def);
  int def_image_step = pyramid.get_step(pyramid_level, imageType_def);

#if DEBUG_CORRELATION_PLOTS
  int und_image_rows = pyramid.get_rows(pyramid_level, imageType_und);
  int und_image_cols = pyramid.get_cols(pyramid_level, imageType_und);

  cv::Mat undTemp(und_image_rows, und_image_cols, CV_8U, und_image_ptr,
                  und_image_step);
  cv::imshow("und image", undTemp);

  cv::Mat defTemp(def_image_rows, def_image_cols, CV_8U, def_image_ptr,
                  def_image_step);
  cv::imshow("def image", defTemp);
#endif

  float *all_interpolation_parameters = pyramid.get_all_param(pyramid_level);

  int points_per_thread = number_of_points / number_of_threads;

  for (int ithread = 0; ithread < number_of_threads; ++ithread) {
    correlation_thread_data[ithread].thread_id = ithread;
    correlation_thread_data[ithread].number_of_points_this_thread =
        points_per_thread;
    correlation_thread_data[ithread].modeler = &modelers[ithread];
    correlation_thread_data[ithread].interpolator = interpolators[ithread];
    correlation_thread_data[ithread].error_status = false;
    correlation_thread_data[ithread].error_code = error_none;
    correlation_thread_data[ithread].runInterpolation = runInterpolation;
  }

  // Left over points
  for (int ithread = 0; ithread < number_of_points % number_of_threads;
       ++ithread) {
    correlation_thread_data[ithread].number_of_points_this_thread++;
  }

  int first_point_this_thread = 0;
  for (int ithread = 0; ithread < number_of_threads; ++ithread) {
#if DEBUG_THREAD
    std::cout << "correlation() : creating thread, " << ithread << std::endl;
#endif

    correlation_thread_data[ithread].modeler->set_thread_id(ithread);
    correlation_thread_data[ithread].interpolator->set_thread_id(ithread);
    correlation_thread_data[ithread].first_point_this_thread =
        first_point_this_thread;

    // Divide the transformation problem and the interpolation problem for the
    // modeleres and interpolators.
    // 	Every thread gets a reference to the corresponding entry of the
    // und_xy_positions in the undeformed image
    //	which is the thread number multiplied by the number of points in the
    //threads, times 2 (x and y).
    //	This way, for example, if there are 15 points per thread, the reference
    //of thread 0th, "points" at the
    //	location 0, the next thread has a reference pointing to the entry 15 * 2
    //= 30. The last thread may have
    //	more items than the rest, but the begining of the array follows the same
    //rule. Similar rule applies for the
    //	def_xy_points, w_results and dwdxy_results.
    // 	NOTE THIS IS JUST SETTING THE PROBLEMS FOR EACH THREAD, NOT SOLVING THEM
    // - that is done by h_interpolation

    modelers[ithread].set_points(
        correlation_thread_data[ithread]
            .number_of_points_this_thread, // this object needs to go away
        &und_xy_positions[first_point_this_thread * 2],
        &def_xy_positions[first_point_this_thread * 2], model_parameters,
        und_x_center, und_y_center,
        &dTxydp[first_point_this_thread * number_of_model_parameters * 2]);
    if (runInterpolation) {
      interpolators[ithread]->set_und_image(und_image_ptr, und_image_step);

      interpolators[ithread]->set_def_image(def_image_ptr, def_image_rows,
                                            def_image_cols, def_image_step,
                                            all_interpolation_parameters);

      interpolators[ithread]->set_multiple_interpolations(
          correlation_thread_data[ithread].number_of_points_this_thread,
          &und_intensities[first_point_this_thread *
                           number_of_colors], // this needs to go away
          &und_xy_positions[first_point_this_thread * 2],
          &def_xy_positions[first_point_this_thread *
                            2], // this needs to go away
          &w_results[first_point_this_thread *
                     number_of_colors], // this needs to go away
          &dwdxy_results[first_point_this_thread * number_of_colors *
                         2], // this needs to go away
          model_parameters,
          und_x_center, und_y_center,
          &dTxydp[first_point_this_thread * number_of_model_parameters * 2]);
    }

    correlationThreads.push_back(std::async(std::launch::async, h_interpolation,
                                            correlation_thread_data[ithread]));

    first_point_this_thread +=
        correlation_thread_data[ithread].number_of_points_this_thread;
  }

  // join threads
  for (int ithread = 0; ithread < number_of_threads; ithread++) {
    s_correlation_thread_data correlationResult =
        correlationThreads[ithread].get();

    time_all_model += correlationResult.model_time;

    if (runInterpolation) {
      time_all_interpolation += correlationResult.interpolation_time;

      //  Aggregates contributions from all threads to construct global chi,
      //  mat_A and vec_B
      //      to be used by the solver
      chi += correlationResult.contribution_chi;

      for (int p1 = 0; p1 < number_of_model_parameters; ++p1) {
        vec_B[p1] += correlationResult.contribution_vec_B[p1];

        for (int p2 = p1; p2 < number_of_model_parameters; ++p2)
          mat_A[p1 * number_of_model_parameters + p2] +=
              correlationResult
                  .contribution_mat_A[p1 * number_of_model_parameters + p2];
      }
    }

    if (correlationResult.error_status)
      error_code = correlationResult.error_code;
    error_status = error_status || correlationResult.error_status;

#if DEBUG_THREAD
    std::cout << "correlation() : completed thread id : " << ithread
              << "  exiting with status : " << status << std::endl;
#endif
  }

#if DEBUG_TIME_MODEL_AND_INTERPOLATION
  auto duration_model_and_interpolation =
      (std::clock() - start_model_and_interpolation) / (float)CLOCKS_PER_SEC;
  std::cout << "correlation() :model and interpolation execution time(s): "
            << duration_model_and_interpolation << '\n';
#endif

#if DEBUG_CORRELATION
  debug_correlation(pyramid_level);
#endif

  delete[] correlation_thread_data;
  correlation_thread_data = nullptr;
}

float *CorrelationClass::get_model_parameters() { return model_parameters; }

// This method is for correlation of another rectangular window,
// maybe with different dimensions (xdim,ydim) than previously defined
float *CorrelationClass::Newton_Raphson(float *model_parameters_in,
                                        int number_of_points_in,
                                        float und_x_center_in,
                                        float und_y_center_in,
                                        float *und_xy_positions_in) {
  pyramid.set_und_center(und_x_center_in, und_y_center_in);

  set_center_flag = false; // consider making this not-a-global

  return CorrelationClass::Newton_Raphson(
      model_parameters_in, number_of_points_in, und_xy_positions_in);
}

// This method is for correlation of a new set of points, maybe with a different
// number
// of points than already allocated. We will delete the old und_xy_positions,
// def_xy_positions
// w_results, dwdxy_results and dTxydp arrays and reallocate new ones if the
// sizes are different.
// take a pointer of the new xy_position array.
float *CorrelationClass::Newton_Raphson(float *model_parameters_in,
                                        int number_of_points_in,
                                        float *und_xy_positions_in) {
  if (allocated_points < number_of_points_in) {
    allocated_points = number_of_points_in;
    delete_point_dependent_arrays();
    allocate_point_dependent_arrays();
  }

  pyramid.set_xy_positions(und_xy_positions_in, number_of_points_in);

  if (set_center_flag) {
    pyramid.set_und_center();
  }
  set_center_flag = true;

  return Newton_Raphson(model_parameters_in);
}

// This method assumes no changes on the correlation set, so und_xy_positions
// will
// still be used. Only change is the initial guess, passed as
// model_parameters_in
float *CorrelationClass::Newton_Raphson(float *model_parameters_in) {
  // start timing
  start_newton_raphson = std::clock();
  time_all_model = 0.f;
  time_all_interpolation = 0.f;
  time_all_new_parameters_assembly = 0.f;
  time_all_new_parameters_solver = 0.f;

  int pyramid_level_old = 0;
  bool runInterpolation = true;

  model_parameters = model_parameters_in;

  //  New parameters and chi are computed in the same loop since there is a lot
  //  of sinergy.
  //      We need to keep track of last two computed set of parameters.
  // Store these parameters, if the step doesn't converge
  // we will start increasing lambda from the last converging set of
  // parameters

  float *last_good_model_parameters = new float[number_of_model_parameters];
  float *tentative_model_parameters = new float[number_of_model_parameters];
  float *saved_model_parameters = new float[number_of_model_parameters];

  for (int pyramid_level = pyramid_stop; pyramid_level >= pyramid_start;
       pyramid_level -= pyramid_step) {
    pyramid.translate_model_parameters(model_parameters, pyramid_level_old,
                                       pyramid_level);

#if DEBUG_CORRELATION_INFO
    debug_correlation_info(pyramid_level);
#endif

    error_status = false;
    error_code = error_none;

    float lambda = 0.0001;
    float min_lambda = 1e-9;
    float max_lambda = 1e9;

    last_good_chi = std::numeric_limits<float>::max();

#if DEBUG_NEWTON_RAPHSON
    std::cout << std::endl << "NEWTON_RAPHSON" << std::endl;
    std::cout << "initial guess "
              << " -> ";
    for (int p = 0; p < number_of_model_parameters; ++p)
      std::cout << model_parameters[p] << " ";
    std::cout << std::endl;
#endif

    int number_of_points = pyramid.get_number_of_points(pyramid_level);
    // float scaling = 1.f / ( 10.f * (float) pow( number_of_points , 2 ) );
    float scaling = 1.f / ((float)number_of_points);

    //  Take the initial guess as the last good model parameters
    for (int p = 0; p < number_of_model_parameters; ++p) {
      last_good_model_parameters[p] = model_parameters[p];
    }

    //  Compute INITIAL CHI for the initial_guess - based on model_parameters
    flush_A_B();
    apply_model_and_interpolate(pyramid_level, runInterpolation);

    if (error_status) {
#if DEBUG_NEWTON_RAPHSON
      printf("Error code =  %2d\n", error_code);
#endif
      pyramid.translate_model_parameters(model_parameters, pyramid_level, 0);
      return model_parameters;
    }

    chi *= scaling;
    last_good_chi = chi;

#if DEBUG_NEWTON_RAPHSON_FAIL_DUMP
    if (last_good_chi > 1e8)
      Newton_Raphson_dump(pyramid_level);
// return model_parameters;
#endif

    //   Since we have the matrix_A and vector_B contributions from the chi
    //   calculation,
    //      may as well compute the NEXT PARAMETERS AND SAVE THEM.
    compute_model_parameters(lambda, scaling);

    for (int p = 0; p < number_of_model_parameters; ++p) {
      saved_model_parameters[p] = model_parameters[p];
    }

    bool use_saved_parameters = true;

    for (int iteration = 1; iteration <= maximum_iterations + 1; ++iteration) {

      if (iteration > maximum_iterations || lambda >= max_lambda) {
        error_status = true;
        error_code = error_correlation_max_iters_reached;
#if DEBUG_NEWTON_RAPHSON
        printf("Error code =  %2d\n", error_code);
#endif

        break;
      } else {
        reached_iterations = iteration;
      }

      // Sequencial computation of model_parameters and chi has a lot of
      // sinergy. The new
      //      model_parameters can be computed at the same time as chi, but they
      //      would be
      //      the future parameter set. Instead of running the multithreading
      //      engine twice,
      //      the future parameter set is saved and reused, only if the chi in
      //      last step was better.
      if (use_saved_parameters) {
        for (int p = 0; p < number_of_model_parameters; ++p) {
          tentative_model_parameters[p] = saved_model_parameters[p];
        }
      }
      //  If last chi was worst, we need to recompute the model parameters.
      //  Calls multithreading engine to assemble upper half of the correlation
      //  matrix for this iteration
      //      to compute the model parameters.
      //  Using und_xy_positions and the model_parameters,
      //      1) model : generates def_xy_positions and dTwdxy
      //      2) intepolation : generates w_results and dwdxy_results
      else {
        for (int p = 0; p < number_of_model_parameters; ++p) {
          model_parameters[p] = last_good_model_parameters[p];
        }

        flush_A_B();
        apply_model_and_interpolate(pyramid_level, runInterpolation);
        chi *= scaling;

        if (error_status) {
#if DEBUG_NEWTON_RAPHSON
          printf("Error code =  %2d\n", error_code);
#endif
          break;
        }

        // model_parameters -> model_parameters
        compute_model_parameters(lambda, scaling);

        for (int p = 0; p < number_of_model_parameters;
             ++p) // These are the new set of model_parameters
        {
          tentative_model_parameters[p] = model_parameters[p];
        }
      }

      //  Computation of CHI FOR THE TENTATIVE set of model parameters. ( put
      //  them in "model_parameteres" )
      for (int p = 0; p < number_of_model_parameters; ++p) {
        model_parameters[p] = tentative_model_parameters[p];
      }

      flush_A_B();
      apply_model_and_interpolate(pyramid_level, runInterpolation);
      chi *= scaling;

      if (error_status) {
#if DEBUG_NEWTON_RAPHSON
        printf("Error code =  %2d\n", error_code);
#endif
        break;
      }

      //   Since we have the matrix_A and vector_B contributions from the chi
      //   calculation,
      //      may as well compute the next parameters and save them for the
      //      likely case
      //      the new chi is better. Use the next lambda, assuming convergence.
      compute_model_parameters(std::max(lambda * 0.4f, min_lambda), scaling);

      for (int p = 0; p < number_of_model_parameters;
           ++p) // These are the new set of model_parameters
      {
        saved_model_parameters[p] = model_parameters[p];
      }

      //  Compares delta chi based on last_good_chi(last_good_model_parameters)
      //  and
      //      chi(model_parameters). However, it does not act on this info until
      //      the
      //      parameters are updated
      float delta_chi =
          std::abs((last_good_chi - chi) /
                   (std::max(last_good_chi, chi) + required_precision));

#if DEBUG_NEWTON_RAPHSON
      printf("iteration = %4d: last good p: ", iteration);
      for (int p = 0; p < number_of_model_parameters; ++p)
        printf("%12.4e  ", last_good_model_parameters[p]);

      printf("last good chi:%12.4e:    tent. p: ", last_good_chi);
      for (int p = 0; p < number_of_model_parameters; ++p)
        printf("%12.4e  ", tentative_model_parameters[p]);
      printf("tent. chi:%12.4e :delta chi: %12.4e lambda: %12.4e ", chi,
             delta_chi, lambda);
#endif

      if (chi <=
          last_good_chi) // converging step - record the new "best" parameters
      {
        last_good_chi = chi;
        lambda = std::max(lambda * 0.4f, min_lambda);

        for (int p = 0; p < number_of_model_parameters; ++p) {
          last_good_model_parameters[p] = tentative_model_parameters[p];
        }

        use_saved_parameters = true;

#if DEBUG_NEWTON_RAPHSON
        printf(" # CONVERGING\n");
#endif
      } else // diverging step - increase lambda and keep last good set of
             // parameters
      {
        lambda = std::min(lambda * 10.0f, max_lambda);

        use_saved_parameters = false;

#if DEBUG_NEWTON_RAPHSON
        printf("\n");
#endif
      }

      // Was convergence reached?
      if (delta_chi < required_precision) {
#if DEBUG_NEWTON_RAPHSON
        printf("Convergence reached at delta_chi =  %6f\n", delta_chi);
#endif
        break;
      }

    } // for iterations

    pyramid_level_old = pyramid_level;

  } // Pyramid iteration

  delete[] last_good_model_parameters;
  last_good_model_parameters = nullptr;
  delete[] tentative_model_parameters;
  tentative_model_parameters = nullptr;
  delete[] saved_model_parameters;
  saved_model_parameters = nullptr;

  duration_newton_raphson =
      (std::clock() - start_newton_raphson) / (float)CLOCKS_PER_SEC;
#if DEBUG_TIME_NEWTON_RAPHSON
  std::cout << "correlation() :Newton_Raphson execution time(s): "
            << duration_newton_raphson << '\n';
#endif

#if DEBUG_TIME_ALL_MODEL
  std::cout << "correlation() :ALL model execution time(s): " << time_all_model
            << " for n = " << pyramid.get_number_of_points(0) << " and "
            << number_of_threads << " threads"
            << " iterations = " << reached_iterations << '\n';
#endif

#if DEBUG_TIME_ALL_INTERPOLATION
  std::cout << "correlation() :ALL interpolation execution time(s): "
            << time_all_interpolation
            << " for n = " << pyramid.get_number_of_points(0) << " and "
            << number_of_threads << " threads"
            << " iterations = " << reached_iterations << '\n';
#endif

#if DEBUG_TIME_ALL_NEW_PARAMETERS_ASSEMBLY
  std::cout << "correlation() :ALL matrix assembly execution time(s): "
            << time_all_new_parameters_assembly
            << " for n = " << pyramid.get_number_of_points(0) << " and "
            << number_of_threads << " threads"
            << " iterations = " << reached_iterations << '\n';
#endif

#if DEBUG_TIME_ALL_NEW_PARAMETERS_SOLVER
  std::cout << "correlation() :ALL matrix solver execution time(s): "
            << time_all_new_parameters_solver
            << " for n = " << pyramid.get_number_of_points(0) << " and "
            << number_of_threads << " threads"
            << " iterations = " << reached_iterations << '\n';
#endif

  pyramid.translate_model_parameters(model_parameters, pyramid_level_old, 0);
  return model_parameters;
}

void CorrelationClass::compute_model_parameters(float lambda, float scaling) {
  start_new_parameters = std::clock();
  start_new_parameters_assembly = std::clock();

  // apply scaling to mat_A and vec_B for numerical precission
  for (int p1 = 0; p1 < number_of_model_parameters; ++p1) {
    vec_B[p1] *= scaling;
    for (int p2 = p1; p2 < number_of_model_parameters; ++p2)
      mat_A[p1 * (number_of_model_parameters) + p2] *= scaling;
  }

  //  Hessian matrix is symetric. Only upper part was computed, now copy to
  //  lower part.
  //  p1 index starts with 0 intentionally, to apply the LM scalling to mat_A[
  //  0, 0 ]
  for (int p1 = 0; p1 < number_of_model_parameters; ++p1) {
    // this loop will not be executed the first time, that is ok
    for (int p2 = 0; p2 < p1; ++p2)
      mat_A[p1 * (number_of_model_parameters) + p2] =
          mat_A[p2 * (number_of_model_parameters) + p1];

    //  Apply the Lebevenberg-Marquardt scaling to the diagonal
    mat_A[p1 * (number_of_model_parameters) + p1] *= (1.f + lambda);
  }

  duration_new_parameters_assembly =
      (std::clock() - start_new_parameters_assembly) / (float)CLOCKS_PER_SEC;
  time_all_new_parameters_assembly += duration_new_parameters_assembly;
#if DEBUG_TIME_NEW_PARAMETERS_ASSEMBLY
  std::cout << "correlation() :new parameters assembly execution time(s): "
            << duration_new_parameters_assembly << '\n';
#endif

#if DEBUG_SOLVER
  printf("Correlation: Used Lambda = %16.4e \n\n", lambda);
#endif

  start_new_parameters_solver = std::clock();

  //**********************************************************************************
  //   Solve the system
  float *dp = solve(); // dp now points to the
                       // solution[number_of_model_parameters] working space
  //**********************************************************************************

  for (int p = 0; p < number_of_model_parameters; ++p)
    model_parameters[p] += dp[p];

  duration_new_parameters_solver =
      (std::clock() - start_new_parameters_solver) / (float)CLOCKS_PER_SEC;
  time_all_new_parameters_solver += duration_new_parameters_solver;
#if DEBUG_TIME_NEW_PARAMETERS_SOLVER
  std::cout << "correlation() :new parameters solver execution time(s): "
            << duration_new_parameters_solver << '\n';
#endif

  duration_new_parameters =
      (std::clock() - start_new_parameters) / (float)CLOCKS_PER_SEC;
#if DEBUG_TIME_NEW_PARAMETERS
  std::cout << "correlation() :new parameters execution time(s): "
            << duration_new_parameters << '\n';
#endif
}

bool CorrelationClass::get_error_status() { return error_status; }

errorEnum CorrelationClass::get_error_code() { return error_code; }

void CorrelationClass::flush_A_B() {
  for (int p1 = 0; p1 < number_of_model_parameters; ++p1) {
    vec_B[p1] = 0.f;
    for (int p2 = 0; p2 < number_of_model_parameters; ++p2)
      mat_A[p1 * number_of_model_parameters + p2] = 0.f;
  }
  chi = 0.f;
}

float *CorrelationClass::solve() {
#if DEBUG_SOLVER
  printf("Correlation: matrix and vector - solver \n");
  for (int p1 = 0; p1 < number_of_model_parameters; ++p1) {
    for (int p2 = 0; p2 < number_of_model_parameters; ++p2)
      printf(" %16.4e ", mat_A[p1 * (number_of_model_parameters) + p2]);

    printf(" | %16.4e\n", vec_B[p1]);
  }

  printf("\n");
#endif

  //		//opencv solver
  //    cv::Mat A = cv::Mat(number_of_model_parameters,
  //    number_of_model_parameters ,CV_32FC1, mat_A);
  //    cv::Mat B = cv::Mat(number_of_model_parameters, 1, CV_32FC1, vec_B);
  //    cv::Mat x = A.inv() * B;
  //    float *temp = (float*)x.data;
  //    //opencv solver

  // Start eigen solver
  // map memory spaces of mat_A, vec_B and temp to Eigen objects for solver
  Eigen::Map<Eigen::MatrixXf> eigen_map_A(mat_A, number_of_model_parameters,
                                          number_of_model_parameters);
  Eigen::Map<Eigen::VectorXf> eigen_map_B(vec_B, number_of_model_parameters);
  Eigen::Map<Eigen::VectorXf> eigen_map_x(vec_x, number_of_model_parameters);

  eigen_map_x = eigen_map_A.colPivHouseholderQr().solve(eigen_map_B);

  // End of eigen solver

  for (int p = 0; p < number_of_model_parameters; ++p)
    solution[p] = vec_x[p];

#if DEBUG_SOLVER
  printf("Correlation: solution - solver\n");

  for (int p = 0; p < number_of_model_parameters; ++p)
    printf(" %16.4e ", solution[p]);

  std::cout << std::endl
            << " residuals " << std::endl
            << eigen_map_A * eigen_map_x - eigen_map_B << std::endl;
  std::cout << std::endl;

#endif

  return solution;
}

void CorrelationClass::debug_correlation_info(int pyramid_level) {
  int number_of_points = pyramid.get_number_of_points(pyramid_level);
  float und_x_center, und_y_center;
  pyramid.get_und_center(und_x_center, und_y_center, pyramid_level);

  std::cout << std::endl << "correlation(): INFO " << std::endl;
  std::cout << "number of points " << number_of_points << std::endl;
  std::cout << "center_x " << und_x_center << std::endl;
  std::cout << "center_y " << und_y_center << std::endl;
  std::cout << "precision " << required_precision << std::endl;
  std::cout << "maximum iterations " << maximum_iterations << std::endl;
  std::cout << "number of colors " << number_of_colors << std::endl;
  std::cout << "interpolation model " << interpolationModel << std::endl;
  std::cout << "number of interpolation parameters "
            << number_of_interpolation_parameters << std::endl;
  std::cout << "fitting model " << fittingModel << std::endl;
  std::cout << "number of model parameters " << number_of_model_parameters
            << std::endl;
  std::cout << "number of threads " << number_of_threads << std::endl;
}

void CorrelationClass::debug_correlation(int pyramid_level) {

  int number_of_points = pyramid.get_number_of_points(pyramid_level);
  float *und_xy_positions = pyramid.get_xy_positions(pyramid_level);
  unsigned char *und_image_ptr = pyramid.get_und_ptr(pyramid_level);

  std::cout << "Correlation results" << std::endl;

  std::cout << "model parameters ";
  for (int p = 0; p < number_of_model_parameters; ++p)
    std::cout << model_parameters[p] << " ";
  std::cout << std::endl;

  for (int ipoint = 0; ipoint < number_of_points; ++ipoint) {
    int x = (int)und_xy_positions[ipoint * 2 + 0];
    int y = (int)und_xy_positions[ipoint * 2 + 1];

    float defx = def_xy_positions[ipoint * 2 + 0];
    float defy = def_xy_positions[ipoint * 2 + 1];

    std::cout << "point " << ipoint << " und W (" << x << "," << y << ") = ";

    for (int c = 0; c < number_of_colors; ++c)
      std::cout << (float)und_image_ptr[und_image.step1() * y +
                                        x * number_of_colors + c]
                << ", ";

    std::cout << "-> def W (" << defx << "," << defy << ") = ";

    for (int c = 0; c < number_of_colors; ++c)
      std::cout << w_results[ipoint * number_of_colors + c] << ", ";

    std::cout << " dwdx = ";
    for (int c = 0; c < number_of_colors; ++c)
      std::cout << dwdxy_results[ipoint * (number_of_colors * 2) + c] << ", ";

    std::cout << " dwdy = ";
    for (int c = 0; c < number_of_colors; ++c)
      std::cout << dwdxy_results[ipoint * (number_of_colors * 2) +
                                 number_of_colors + c]
                << ", ";

    std::cout << ", dTxdp = ";
    for (int p = 0; p < number_of_model_parameters; ++p)
      std::cout << dTxydp[ipoint * (number_of_model_parameters * 2) + p] << " ";

    std::cout << ", dTydp = ";
    for (int p = 0; p < number_of_model_parameters; ++p)
      std::cout << dTxydp[ipoint * (number_of_model_parameters * 2) +
                          number_of_model_parameters + p]
                << " ";

    std::cout << std::endl;
  }
  std::cout << std::endl;
}

float CorrelationClass::get_chi() { return last_good_chi; }

int CorrelationClass::get_number_of_points() {
  return pyramid.get_number_of_points(0);
}

float CorrelationClass::get_und_x_center() {
  float und_x_center, und_y_center;

  pyramid.get_und_center(und_x_center, und_y_center, 0);

  return und_x_center;
}

float CorrelationClass::get_und_y_center() {
  float und_x_center, und_y_center;

  pyramid.get_und_center(und_x_center, und_y_center, 0);

  return und_y_center;
}

int CorrelationClass::get_iterations() { return reached_iterations; }

v_points CorrelationClass::getUndXY0() {
  v_points vxy(pyramid.get_number_of_points(0));

  float *temp = pyramid.get_xy_positions(0);

  for (unsigned int ipoint = 0; ipoint < vxy.size(); ++ipoint) {
    vxy[ipoint] = std::make_pair(temp[ipoint * 2], temp[ipoint * 2 + 1]);
  }

  return vxy;
}

v_points CorrelationClass::getDefXY0() {
  int numberOfPoints0 = pyramid.get_number_of_points(0);
  v_points defXY(numberOfPoints0);

  if (pyramid_start != 0) {
    bool runInterpolation = false;
    apply_model_and_interpolate(0, runInterpolation);
  }

  memcpy(defXY.data(), def_xy_positions, numberOfPoints0 * 2 * sizeof(float));

  return defXY;
}

void CorrelationClass::Newton_Raphson_dump(int pyramid_level) {
  float *all_interpolation_parameters = pyramid.get_all_param(pyramid_level);
  int number_of_points = pyramid.get_number_of_points(pyramid_level);
  float *und_xy_positions = pyramid.get_xy_positions(pyramid_level);

  printf("                            point       und_x       und_y       "
         "und_w       def_x       def_y       def_w       def_wx      "
         "def_wy\n");

  for (int ipoint = 0; ipoint < number_of_points; ++ipoint) {
    if (std::abs(w_results[ipoint * number_of_colors]) > 1000) {
      printf(" correlation:           %6d     %10.4f   %10.4f   %10.4f   "
             "%10.4f   %10.4f   %10.4f   %10.4f   %10.4f\n",
             ipoint, und_xy_positions[ipoint * 2],
             und_xy_positions[ipoint * 2 + 1],
             und_intensities[ipoint * number_of_colors],
             def_xy_positions[ipoint * 2], def_xy_positions[ipoint * 2 + 1],
             w_results[ipoint * number_of_colors], dwdxy_results[ipoint * 2],
             dwdxy_results[ipoint * 2 + 1]);

      float xdef = def_xy_positions[ipoint * 2];
      float ydef = def_xy_positions[ipoint * 2 + 1];

      int ix = (int)xdef;
      int iy = (int)ydef;

      float debug_w_results[3];
      float debug_dwdxy_results[6];

      int index0 =
          (ix + iy * def_image.cols) *
              (number_of_colors * number_of_interpolation_parameters + 1) +
          1;

      float dx = xdef - (int)xdef + 1.f;
      float dy = ydef - (int)ydef + 1.f;

      float px[4] = {1.f, dx, dx * dx, dx * dx * dx};
      float py[4] = {1.f, dy, dy * dy, dy * dy * dy};

      // Initialize results to 0
      for (int c = 0; c < number_of_colors; c++) {
        debug_w_results[c] = 0.f;
        debug_dwdxy_results[c] = 0.f;
        debug_dwdxy_results[number_of_colors + c] = 0.f;

        int index_c = index0 + c * number_of_interpolation_parameters;

        printf("%10.4f\n", all_interpolation_parameters[index_c - 1]);

        for (int jk = 0; jk < 4; jk++) {
          int index_c_jk = index_c + jk * 4;

          for (int ik = 0; ik < 4; ik++) {
            int index_c_jk_ik = index_c_jk + ik;

            //   w_result
            debug_w_results[c] +=
                all_interpolation_parameters[index_c_jk_ik] * py[jk] * px[ik];

            if (ik > 0) // dwdx_result
              debug_dwdxy_results[c] +=
                  ik * all_interpolation_parameters[index_c_jk_ik] * py[jk] *
                  px[ik - 1];

            if (jk > 0) // dwdy_result
              debug_dwdxy_results[number_of_colors + c] +=
                  jk * all_interpolation_parameters[index_c_jk_ik] *
                  py[jk - 1] * px[ik];

            printf("%10.4f\n", all_interpolation_parameters[index_c_jk_ik]);

          } // ik bracket
        }   // jk bracket

        printf("\n xdef=%10.4f      ydef=%10.4f      w=%10.4f      wx=%10.4f   "
               "   wy%10.4f      \n",
               xdef, ydef, debug_w_results[c], debug_dwdxy_results[c],
               debug_dwdxy_results[number_of_colors + c]);

      } // c  bracket

      // Dump thread info
      for (int ithread = 0; ithread < number_of_threads; ++ithread) {
        printf(" thread = %4d , n = %6d , first = %6d \n", ithread,
               correlation_thread_data[ithread].number_of_points_this_thread,
               correlation_thread_data[ithread].first_point_this_thread);

        for (int j = 0;
             j < correlation_thread_data[ithread].number_of_points_this_thread;
             ++j) {
          int ipoint =
              correlation_thread_data[ithread].first_point_this_thread + j;

          printf(" thread = %2d, xdef = %10.4f, ydef = %10.4f\n", ithread,
                 def_xy_positions[ipoint * 2],
                 def_xy_positions[ipoint * 2 + 1]);
        }
      }
    }
  }
  std::cout << std::endl;
}
