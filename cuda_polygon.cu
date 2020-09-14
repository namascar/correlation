#include "cuda_polygon.cuh"

void cudaPolygon::fillRectangle() {
#if DEBUG_CUDA_POLYGON
  printf("cudaPolygon::fillRectangle: Filling x0 = %d , y0 = %d , x1 = %d , y1 "
         "= %d , size = %d , sector = %d\n",
         x0, y0, x1, y1, size, sector);
#endif

  undeformed_xs.resize(stop + 1);
  undeformed_xs[0].resize(size);
  undeformed_ys.resize(stop + 1);
  undeformed_ys[0].resize(size);

  xy_center.resize(stop + 1);

  // Make x an indexed vector
  thrust::sequence(thrust::cuda::par.on(domainSelectionStream),
                   undeformed_xs[0].begin(), undeformed_xs[0].end());

  // Zip x and y and transform them
  thrust::for_each(thrust::cuda::par.on(domainSelectionStream),

                   thrust::make_zip_iterator(thrust::make_tuple(
                       undeformed_xs[0].begin(), undeformed_ys[0].begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(
                       undeformed_xs[0].end(), undeformed_ys[0].end())),
                   RectFunctor(x0, y0, x1, y1));
}

v_points cudaPolygon::getUndXY0ToCPU() {

#if DEBUG_CUDA_POLYGON
  printf("cudaPolygon::getUndXY0ToCPU\n");
#endif

  thrust::host_vector<float> h_xs = undeformed_xs[0];
  thrust::host_vector<float> h_ys = undeformed_ys[0];

  v_points vxy(h_xs.size());

  for (int i = 0; i < vxy.size(); ++i) {
    vxy[i] = std::make_pair(h_xs[i], h_ys[i]);
  }

  return vxy;
}

v_points cudaPolygon::getDefXY0ToCPU() {
/**
  Method is used to make a host copy of the deformed points for plotting
  purposes
  */
#if DEBUG_CUDA_POLYGON
  printf("cudaPolygon::getDefXY0ToCPU\n");
#endif

  // Generate deformed points for the level 0 as a copy of the undeformed
  // parameters
  thrust::device_vector<float> deformed_xs0 = undeformed_xs[0];
  thrust::device_vector<float> deformed_ys0 = undeformed_ys[0];

  float *d_parameters = getParameters(parType_lastGood);
  int numberOfPoints = getNumberOfPoints(0);
  float *defX_ptr = thrust::raw_pointer_cast(deformed_xs0.data());
  float *defY_ptr = thrust::raw_pointer_cast(deformed_ys0.data());
  float *undCenter = getUndCenter(0);
  int blocksPerGrid =
      (numberOfPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  kModel_inPlace<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_parameters,

                                                       fittingModel,

                                                       numberOfPoints, defX_ptr,
                                                       defY_ptr, undCenter);

  // Transfer deformed points to the CPU
  thrust::host_vector<float> h_xs = deformed_xs0;
  thrust::host_vector<float> h_ys = deformed_ys0;

  // Make a v_points and return it
  v_points vxy(h_xs.size());

  for (int i = 0; i < vxy.size(); ++i) {
    vxy[i] = std::make_pair(h_xs[i], h_ys[i]);
  }

  return vxy;
}

int cudaPolygon::getNumberOfPoints(int level) {
  return undeformed_xs[level].size();
}

float *cudaPolygon::getUndXPtr(int level) {
  return thrust::raw_pointer_cast(undeformed_xs[level].data());
}

float *cudaPolygon::getUndYPtr(int level) {
  return thrust::raw_pointer_cast(undeformed_ys[level].data());
}

float *cudaPolygon::getUndCenter(int level) {
  return thrust::raw_pointer_cast(xy_center[level].data());
}

float *cudaPolygon::getParameters(parameterTypeEnum parSrc) {
  return parameters[parSrc];
}

CorrelationResult *cudaPolygon::getCorrelationResultsToCPU() {
#if DEBUG_CUDA_POLYGON
  printf("cudaPolygon::getCorrelationResultsToCPU , sector = %d\n", sector);
  fflush(stdout);
#endif

  scaleParametersForLevel(0);

  // Copy last parameter set to gpuCorrelationResults
  cudaMemcpy(gpuCorrelationResults->resultingParameters,
             parameters[parType_lastGood],
             numberOfModelParameters * sizeof(float), cudaMemcpyDeviceToDevice);

  // Copy xy_center to gpuCorrelationResults
  cudaMemcpy(&gpuCorrelationResults->undCenterX,
             thrust::raw_pointer_cast(xy_center[0].data()), 2 * sizeof(float),
             cudaMemcpyDeviceToDevice);

  // Copy gpuCorrelationResults to cpuCorrelationResults
  cudaMemcpy(cpuCorrelationResults, gpuCorrelationResults,
             sizeof(CorrelationResult), cudaMemcpyDeviceToHost);

  // Number of points comes from the thrust device_vector, which has its size in
  // the cpu
  cpuCorrelationResults->numberOfPoints = getNumberOfPoints(0);

#if DEBUG_CUDA_POLYGON
  printf("cudaPolygon::getCorrelationResultsToCPU\n");
  printf("cpuCorrelationResults->numberOfPoints = %d\n",
         cpuCorrelationResults->numberOfPoints);
  printf("cpuCorrelationResults->undCenterX = %f\n",
         cpuCorrelationResults->undCenterX);
  printf("cpuCorrelationResults->undCenterY = %f\n",
         cpuCorrelationResults->undCenterY);
  for (int i = 0; i < numberOfModelParameters; ++i) {
    printf("%14.4e", cpuCorrelationResults->resultingParameters[i]);
  }
  printf("\n");
  fflush(stdout);
#endif

  return cpuCorrelationResults;
}

float *cudaPolygon::getGlobalABChi() { return globalABChi; }

void cudaPolygon::updateParameters(int numberOfModelParameters,
                                   parameterTypeEnum parSrc,
                                   parameterTypeEnum parDst,
                                   cudaStream_t stream) {
  kUpdateParameters<<<1, 32, 0, stream>>>(
      parameters[parSrc], parameters[parDst],
      &globalABChi[numberOfModelParameters * numberOfModelParameters],
      numberOfModelParameters);

#if DEBUG_CUDA_POLYGON
  printf("cudaPolygon::updateParameters type = %d , sector = %d\n", parDst,
         sector);

  float *h_par = new float[numberOfModelParameters];
  float *d_par = parameters[parDst];

  cudaMemcpy(h_par, d_par, numberOfModelParameters * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < numberOfModelParameters; ++i) {
    printf("%14.4e", h_par[i]);
  }
  printf("\n");
  fflush(stdout);

  delete[] h_par;
#endif
}

void cudaPolygon::scaleParametersForLevel(int level) {
  if (level == currentPyramidLevel)
    return;

#if DEBUG_CUDA_POLYGON
  printf("cudaPolygon::scaleParametersForLevel - before scale , sector = %d\n",
         sector);

  float *h_par = new float[numberOfModelParameters];
  float *d_par = parameters[parType_lastGood];
  cudaMemcpy(h_par, d_par, numberOfModelParameters * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < numberOfModelParameters; ++i) {
    printf("%14.4e", h_par[i]);
  }
  printf("\n");
  fflush(stdout);
#endif

  int numerator = 1 << currentPyramidLevel;
  int denominator = 1 << level;
  float multiplier = (float)numerator / (float)denominator;

  kScale<<<1, 1>>>(parameters[parType_lastGood], parameters[parType_tentative],
                   parameters[parType_saved], fittingModel, multiplier);

  currentPyramidLevel = level;

#if DEBUG_CUDA_POLYGON
  printf("cudaPolygon::scaleParametersForLevel - after scale\n");

  h_par = new float[numberOfModelParameters];

  cudaMemcpy(h_par, d_par, numberOfModelParameters * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < numberOfModelParameters; ++i) {
    printf("%14.4e", h_par[i]);
  }
  printf("\n");
  fflush(stdout);

  delete[] h_par;
#endif
}

void cudaPolygon::initializeParametersLevel0(float *initialGuess_) {

  // Put a marker on the nvvp CUDA profiler
  nvtxRangePushA("cudaPolygon::initializeParametersLevel0");

  cudaMemcpy(parameters[parType_lastGood], initialGuess_,
             numberOfModelParameters * sizeof(float), cudaMemcpyHostToDevice);

  transferParameters(parType_lastGood, parType_tentative);
  transferParameters(parType_lastGood, parType_saved);

  currentPyramidLevel = 0;

  nvtxRangePop();

#if DEBUG_CUDA_POLYGON
  printf("cudaPolygon::initializeParametersLevel0 , sector = %d\n", sector);

  float *h_par = new float[numberOfModelParameters];
  float *d_par = parameters[parType_lastGood];

  cudaMemcpy(h_par, d_par, numberOfModelParameters * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < numberOfModelParameters; ++i) {
    printf("%14.4e", h_par[i]);
  }
  printf("\n");
  fflush(stdout);

  delete[] h_par;
#endif
}

void cudaPolygon::updatePolygon(
    deformationDescriptionEnum deformationDescription) {
#if DEBUG_CUDA_POLYGON
  printf("cudaPolygon::updatePolygon\n");
  fflush(stdout);
#endif
  switch (deformationDescription) {
  case def_Eulerian:
    return;

  case def_Lagrangian: {
    float dxy[2]{0.f, 0.f};

    switch (fittingModel) {
    case fm_UVUxUyVxVy:
    case fm_UVQ:
    case fm_UV:

      cudaMemcpy(&dxy[0], parameters[parType_lastGood], 2 * sizeof(float),
                 cudaMemcpyDeviceToHost);
      xy_center[0][0] += dxy[0];
      xy_center[0][1] += dxy[1];

      break;

    case fm_U:

      cudaMemcpy(&dxy[0], parameters[parType_lastGood], 1 * sizeof(float),
                 cudaMemcpyDeviceToHost);
      xy_center[0][0] += dxy[0];

      break;

    default:
      assert(false);
      break;
    }

    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                         undeformed_xs[0].begin(), undeformed_ys[0].begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(
                         undeformed_xs[0].end(), undeformed_ys[0].end())),
                     translateFunctor(dxy[0], dxy[1]));
  } break;

  case def_strict_Lagrangian: {
    float *d_parameters = getParameters(parType_lastGood);
    int numberOfPoints = getNumberOfPoints(0);
    float *undX_ptr = getUndXPtr(0);
    float *undY_ptr = getUndYPtr(0);
    float *undCenter = getUndCenter(0);

    int blocksPerGrid =
        (numberOfPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    kModel_inPlace<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        d_parameters,

        fittingModel,

        numberOfPoints, undX_ptr, undY_ptr, undCenter);

    makeUndCenter0();
  } break;

  default:
    assert(false);
    break;
  }

  makeAllUndLevels();
  makeAllUndCenters();
}

void cudaPolygonAnnular::updatePolygon(
    deformationDescriptionEnum deformationDescription) {
#if DEBUG_CUDA_POLYGON
  printf("cudaPolygonAnnular::updatePolygon\n");
#endif
  switch (deformationDescription) {
  case def_Eulerian:
    return;

  case def_Lagrangian: {
    float dx, dy;

    switch (fittingModel) {
    case fm_U:

      xy_center[0][0] += parameters[parType_lastGood][0];

      dx = parameters[parType_lastGood][0];
      dy = 0.f;

      break;

    case fm_UV:
    case fm_UVQ:
    case fm_UVUxUyVxVy:

      xy_center[0][0] += parameters[parType_lastGood][0];
      xy_center[0][1] += parameters[parType_lastGood][1];

      dx = parameters[parType_lastGood][0];
      dy = parameters[parType_lastGood][1];

      break;

    default:
      assert(false);
      break;
    }

    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                         undeformed_xs[0].begin(), undeformed_ys[0].begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(
                         undeformed_xs[0].end(), undeformed_ys[0].end())),
                     translateFunctor(dx, dy));
  } break;

  case def_strict_Lagrangian: {
    float *d_parameters = getParameters(parType_lastGood);
    int numberOfPoints = getNumberOfPoints(0);
    float *undX_ptr = getUndXPtr(0);
    float *undY_ptr = getUndYPtr(0);
    float *undCenter = getUndCenter(0);

    int blocksPerGrid =
        (numberOfPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    kModel_inPlace<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        d_parameters,

        fittingModel,

        numberOfPoints, undX_ptr, undY_ptr, undCenter);

    makeUndCenter0();
  } break;

  default:
    assert(false);
    break;
  }

  makeAllUndLevels();
  makeAllUndCenters();
}

void cudaPolygon::transferParameters(parameterTypeEnum parSrc,
                                     parameterTypeEnum parDst) {
  if (parDst == parSrc)
    return;

  cudaMemcpy(parameters[parDst], parameters[parSrc],
             numberOfModelParameters * sizeof(float), cudaMemcpyDeviceToDevice);

#if DEBUG_CUDA_POLYGON
  printf("cudaPolygon::transferParameters() %d -> %d\n", parSrc, parDst);
#endif
}

void cudaPolygon::makeAllUndLevels() {
  int prevLevel = 0;
  int firstLevel = (start == 0 ? step : start);

  for (int ilevel = firstLevel; ilevel <= stop; ilevel += step) {
    undeformed_xs[ilevel].resize(undeformed_xs[prevLevel].size());
    undeformed_ys[ilevel].resize(undeformed_ys[prevLevel].size());

    ZipIt zipEnd = thrust::copy_if(
        thrust::make_zip_iterator(
            thrust::make_tuple(undeformed_xs[prevLevel].begin(),
                               undeformed_ys[prevLevel].begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            undeformed_xs[prevLevel].end(), undeformed_ys[prevLevel].end())),
        thrust::make_zip_iterator(thrust::make_tuple(
            undeformed_xs[ilevel].begin(), undeformed_ys[ilevel].begin())),
        copyFunctor(prevLevel, ilevel));

    TupleIt tupleEnd = zipEnd.get_iterator_tuple();

    VIt xsEnd = thrust::get<0>(tupleEnd);
    VIt ysEnd = thrust::get<1>(tupleEnd);

    undeformed_xs[ilevel].erase(xsEnd, undeformed_xs[ilevel].end());
    undeformed_ys[ilevel].erase(ysEnd, undeformed_ys[ilevel].end());

    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(
            undeformed_xs[ilevel].begin(), undeformed_ys[ilevel].begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            undeformed_xs[ilevel].end(), undeformed_ys[ilevel].end())),
        scale2DFunctor(prevLevel, ilevel));

#if DEBUG_CUDA_POLYGON
    printf("cudaPolygon::makeLevels ilevel = %d , sector = %d\n", ilevel,
           sector);
#endif

#if DEBUG_CUDA_POLYGON_POINTS

    if (ilevel == firstLevel) {
      printf(" cudaPolygon::makeLevels prevLevel = %d\n", prevLevel);

      thrust::host_vector<float> h_xs = undeformed_xs[prevLevel];
      thrust::host_vector<float> h_ys = undeformed_ys[prevLevel];

      for (int i = 0; i < h_xs.size(); ++i) {
        printf("x[ %d ] = %f , y[ %d ] = %f \n", i, h_xs[i], i, h_ys[i]);
      }
    }

    printf(" cudaPolygon::makeLevels ilevel = %d\n", ilevel);

    thrust::host_vector<float> h_xs = undeformed_xs[ilevel];
    thrust::host_vector<float> h_ys = undeformed_ys[ilevel];

    for (int i = 0; i < h_xs.size(); ++i) {
      printf("x[ %d ] = %f , y[ %d ] = %f \n", i, h_xs[i], i, h_ys[i]);
    }

#endif

    prevLevel = ilevel;
  }

  allocateGlobalABChi();
}

void cudaPolygon::allocateGlobalABChi() {
  /*
   * Allocate device global memory to perform global reduction. One per block.
   * Big enought for the first pyramid step (start)
   */
  int numberOfPoints = getNumberOfPoints(start);
  int numberOfBlocks =
      (numberOfPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int globalSize =
      sizeof(float) * numberOfBlocks *
      (1 + numberOfModelParameters * (1 + numberOfModelParameters));

  if (globalABChi)
    deallocateGlobalABChi();

  cudaError_t err = cudaMalloc((void **)&globalABChi, globalSize);
  if (err != cudaSuccess) {
    printf("Failed to allocate global globalABCHI (error code %s)!\n",
           cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void cudaPolygon::deallocateGlobalABChi() {
  if (!globalABChi)
    return;

  // Free the global globalABCHI, one per block, to perform global reduction
  cudaError_t err = cudaFree(globalABChi);
  if (err != cudaSuccess) {
    printf("Failed to free device globalABCHI (error code %s)!\n",
           cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void cudaPolygon::makeUndCenter0() {
  xy_center[0].resize(2);

  xy_center[0][0] =
      thrust::reduce(undeformed_xs[0].begin(), undeformed_xs[0].end()) /
      (float)undeformed_xs[0].size();
  xy_center[0][1] =
      thrust::reduce(undeformed_ys[0].begin(), undeformed_ys[0].end()) /
      (float)undeformed_ys[0].size();

#if DEBUG_CUDA_POLYGON
  printf("cudaPolygon::makeUndCenter0\n");
#endif
}

void cudaPolygon::makeAllUndCenters() {
  int prevLevel = 0;
  int firstLevel = (start == 0 ? step : start);

  for (int ilevel = firstLevel; ilevel <= stop; ilevel += step) {
    xy_center[ilevel] = xy_center[prevLevel];

    thrust::transform(xy_center[ilevel].begin(), xy_center[ilevel].end(),
                      xy_center[ilevel].begin(),
                      scale1DFunctor(prevLevel, ilevel));
#if DEBUG_CUDA_POLYGON
    printf("cudaPolygon::makeAllUndCenters ilevel = %d , sector = %d\n", ilevel,
           sector);
#endif

#if DEBUG_CUDA_POLYGON_POINTS

    if (ilevel == firstLevel) {
      printf(" cudaPolygon::makeAllUndCenters prevLevel = %d\n", prevLevel);

      thrust::host_vector<float> h_xs = xy_center[prevLevel];

      printf("x_center = %f , y_center = %f \n", h_xs[0], h_xs[1]);
    }

    printf(" cudaPolygon::makeAllUndCenters ilevel = %d\n", ilevel);

    thrust::host_vector<float> h_xs = xy_center[ilevel];

    printf("x_center = %f , y_center = %f \n", h_xs[0], h_xs[1]),

#endif

        prevLevel = ilevel;
  }
}

void cudaPolygonAnnular::cleanAnnularRectangle0(float r, float dr, float a,
                                                float da, float cx, float cy,
                                                int as) {
  ZipIt zipEnd = thrust::remove_if(
      thrust::cuda::par.on(domainSelectionStream),

      thrust::make_zip_iterator(thrust::make_tuple(undeformed_xs[0].begin(),
                                                   undeformed_ys[0].begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(undeformed_xs[0].end(), undeformed_ys[0].end())),
      removeAnnularFunctor(r, dr, a, da, cx, cy, as));

  TupleIt tupleEnd = zipEnd.get_iterator_tuple();

  VIt xsEnd = thrust::get<0>(tupleEnd);
  VIt ysEnd = thrust::get<1>(tupleEnd);

  undeformed_xs[0].erase(xsEnd, undeformed_xs[0].end());
  undeformed_ys[0].erase(ysEnd, undeformed_ys[0].end());
}

void cudaPolygonBlob::cleanBlobRectangle0(v_points blobContour) {
  LineEquationsDevice lineEquations = makeLineEquations(blobContour);
  v_pairs contours = blobContour;

  ZipIt zipEnd = thrust::remove_if(
      thrust::cuda::par.on(domainSelectionStream),

      thrust::make_zip_iterator(thrust::make_tuple(undeformed_xs[0].begin(),
                                                   undeformed_ys[0].begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(undeformed_xs[0].end(), undeformed_ys[0].end())),
      removeBlobFunctor(contours.data(), lineEquations.data(),
                        lineEquations.size()));

  TupleIt tupleEnd = zipEnd.get_iterator_tuple();

  VIt xsEnd = thrust::get<0>(tupleEnd);
  VIt ysEnd = thrust::get<1>(tupleEnd);

  undeformed_xs[0].erase(xsEnd, undeformed_xs[0].end());
  undeformed_ys[0].erase(ysEnd, undeformed_ys[0].end());
}

LineEquationsDevice cudaPolygonBlob::makeLineEquations(v_points blobContour) {
  LineEquationsHost lineEquationsH(blobContour.size());

  for (unsigned int i = 0; i < blobContour.size(); ++i) {
    float v1x1, v1y1, v1x2, v1y2;

    v1x1 = blobContour[i].first;
    v1y1 = blobContour[i].second;

    if (i != blobContour.size() - 1) {
      v1x2 = blobContour[i + 1].first;
      v1y2 = blobContour[i + 1].second;
    } else {
      v1x2 = blobContour[0].first;
      v1y2 = blobContour[0].second;
    }

    thrust::get<0>(lineEquationsH[i]) = v1y2 - v1y1;
    thrust::get<1>(lineEquationsH[i]) = v1x1 - v1x2;
    thrust::get<2>(lineEquationsH[i]) = (v1x2 * v1y1) - (v1x1 * v1y2);
  }

  LineEquationsDevice lineEquationsD = lineEquationsH;

  return lineEquationsD;
}
