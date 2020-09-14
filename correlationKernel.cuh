#ifndef CORRELATION_KERNELS_CUH
#define CORRELATION_KERNELS_CUH

#include <cuda_runtime.h>

#include <thrust/tuple.h>

#include "enums.hpp"

#include <stdio.h>

/**
  CUDA Kernel to perform parameter update one kernel launch ( nPar threads, 1
  block ).
  */
__global__ void kUpdateParameters(const float *parametersSrc,
                                  float *parametersDst,
                                  const float *parametersIncrement,
                                  const int numberOfModelParameters);

/**
  CUDA Kernel to perform pyramid scaling in one kernel launch ( 1 thread, 1
  block ).
  */
__global__ void kScale(float *d_parametersLastGood,
                       float *d_parametersTentative, float *d_parametersSaved,
                       const fittingModelEnum fittingModel,
                       const float multiplier);

/**
  CUDA Kernel to perform inplace fitting model in one kernel launch. Takes the
  undX,Y positions
  and computes their corresponding defX,Y based on the fitting model (U, UV,
  UVQ, UVUxUyVxVy).
  */
__global__ void kModel_inPlace(const float *d_parameters,

                               const fittingModelEnum fittingModel,

                               const int numberOfPoints, float *undX_ptr,
                               float *undY_ptr, const float *undCenter);

/**
  CUDA Kernel to perform fitting model and interpolation in one kernel launch.
  Takes the undX,Y positions
  and computes their corresponding defX,y and derivatives of the image with
  respect to the parameters
  based on the fitting model (U, UV, UVQ, UVUxUyVxVy). Then Queries the undImage
  and defImage to assempble
  the matix and vector contribution of this point. Finnaly reduces all info to a
  single matrix, vector and chi.
  */
__global__ void kCorrelation(const float *d_parameters,

                             const fittingModelEnum fittingModel,
                             const interpolationModelEnum interpolationModel,

                             const int numberOfColors,
                             const cudaTextureObject_t undTexture,
                             const cudaTextureObject_t defTexture,

                             const int numberOfPoints, const float *undX_ptr,
                             const float *undY_ptr, const float *undCenter,
                             float *global_mat_A_and_vec_B_and_CHI);

/**
 CUDA device function to perform the nearest interpolation and get the
 contribution to the matrix A, vector b and chi
 */
__device__ void dNearest(float undX, float undY, float defX, float defY,

                         float xMinusCenter, float yMinusCenter,

                         const fittingModelEnum fittingModel,

                         float *shared_mat_A_and_vec_B_and_CHI, int size_A,
                         int size_A_B, int size_A_B_chi,

                         int numberOfColors, cudaTextureObject_t undTexture,
                         cudaTextureObject_t defTexture);
/**
 CUDA device function to perform the bilinear interpolation and get the
 contribution to the matrix A, vector b and chi
 */
__device__ void dBilinear(float undX, float undY, float defX, float defY,

                          float xMinusCenter, float yMinusCenter,

                          const fittingModelEnum fittingModel,

                          float *shared_mat_A_and_vec_B_and_CHI, int size_A,
                          int size_A_B, int size_A_B_chi,

                          int numberOfColors, cudaTextureObject_t undTexture,
                          cudaTextureObject_t defTexture);

/**
 CUDA device function to perform the bicubic interpolation and get the
 contribution to the matrix A, vector b and chi
 */
__device__ void dBicubic(float undX, float undY, float defX, float defY,

                         float xMinusCenter, float yMinusCenter,

                         const fittingModelEnum fittingModel,

                         float *shared_mat_A_and_vec_B_and_CHI, int size_A,
                         int size_A_B, int size_A_B_chi,

                         int numberOfColors, cudaTextureObject_t undTexture,
                         cudaTextureObject_t defTexture);

/**
 CUDA device function to build the H vector for the U model
 */
__device__ void dBuildHU(float *H,

                         const float def_dwdx);

/**
 CUDA device function to build the H vector for the UV model
 */
__device__ void dBuildHUV(float *H,

                          const float def_dwdx, const float def_dwdy);

/**
 CUDA device function to build the H vector for the UVQ model
 */
__device__ void dBuildHUVQ(float *H,

                           const float def_dwdx, const float def_dwdy,

                           const float xMinusCenter, const float yMinusCenter);

/**
 CUDA device function to build the H vector for the UVUxUyVxVy model
 */
__device__ void dBuildHUVUxUyVxVy(float *H,

                                  const float def_dwdx, const float def_dwdy,

                                  const float xMinusCenter,
                                  const float yMinusCenter);

#endif // CORRELATION_KERNELS_CUH
