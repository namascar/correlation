#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

#include <thrust/tuple.h>

#include "enums.hpp"

#include <stdio.h>

//***************************************************************************
//
//	UTILITY Kernels

// CUDA kernel to build LS problem in GPU#0
__global__ void
k_build_LS_problem_in_GPU0(
    float *global_mat_A_and_vec_B_and_CHI,
    const int number_of_model_parameters, const float scaling, const float lambda);

// CUDA kernel to copy temp LS variables in GPU#0
__global__ void
k_aggregate_LS_problem_in_GPU0(
    float *global_mat_A_and_vec_B_and_CHI,
    const int number_of_model_parameters);

// CUDA kernel to perform the global reduction
__global__ void
k_global_reduction(
    const int number_of_blocks,
    //const int s,
    float *global_mat_A_and_vec_B_and_CHI,
    const int number_of_model_parameters);

//***************************************************************************
//
//  MODEL kernels

// CUDA Kernel device code to perform U model
// of a blob array of points.
//
__global__ void
k_model_U(
    const float     *d_und_xy_positions,
    const int        number_of_points,
    const float     *d_parameters,
          float     *d_def_xy_positions,
          float     *d_dTxydp,
          errorEnum *error_code
    );

// CUDA Kernel device code to perform UV model
// of a blob array of points.
//
//
__global__ void
k_model_UV(
    const float     *d_und_xy_positions,
    const int        number_of_points,
    const float     *d_parameters,
          float     *d_def_xy_positions,
          float     *d_dTxydp,
          errorEnum *error_code
    );

// CUDA Kernel device code to perform UVQ model
// of a blob array of points.
//
//
__global__ void
k_model_UVQ(
    const float    *d_und_xy_positions,
    const int       number_of_points,
    const float    *d_parameters,
    const float     und_x_center,
    const float     und_y_center,
          float     *d_def_xy_positions,
          float     *d_dTxydp,
          errorEnum *error_code
    );

// CUDA Kernel device code to perform UVUxUyVxVy model
// of a blob array of points.
//
//
__global__ void
k_model_UVUxUyVxVy(
    const float     *d_und_xy_positions,
    const int         number_of_points,
    const float     *d_parameters,
    const float      und_x_center,
    const float      und_y_center,
          float     *d_def_xy_positions,
          float     *d_dTxydp,
          errorEnum *error_code
    );

//***************************************************************************
//
//  INTERPOLATION kernels

// CUDA Kernel device code to perform "nearest" interpolation of a blob array of points.
//  Also performs block reduction.
__global__ void
k_interpolation_and_matrix_assembler_nearest(
    const float *d_def_xy_positions,
    const float *d_und_intensities, const int number_of_points,
    const float *d_dTxydp,
    const int    number_of_model_parameters,
    const int    number_of_colors,
    float       *global_mat_A_and_vec_B_and_CHI,
    cudaTextureObject_t def_tex,
    errorEnum   *error_code);

// CUDA Kernel device code to perform bilinear interpolation of a blob array of points.
//  Also performs block reduction.
__global__ void
k_interpolation_and_matrix_assembler_bilinear(
    const float *d_def_xy_positions,
    const float *d_und_intensities, const int number_of_points,
    const float *d_dTxydp,
    const int    number_of_model_parameters,
    const int    number_of_colors,
    float       *global_mat_A_and_vec_B_and_CHI,
    cudaTextureObject_t def_tex,
    errorEnum   *error_code);

// CUDA Kernel device code to perform bicubic interpolation of a blob array of points.
//  Also performs block reduction.
__global__ void
k_interpolation_and_matrix_assembler_bicubic(
    const float *d_def_xy_positions,
    const float *d_und_intensities, const int number_of_points,
    const float *d_dTxydp,
    const int    number_of_model_parameters,
    const int    number_of_colors,
    float       *global_mat_A_and_vec_B_and_CHI,
    cudaTextureObject_t def_tex,
    errorEnum   *error_code);

// CUDA Kernel to make a black and white pyramid level
__global__ void
k_pyramid_bw
(
    const cudaSurfaceObject_t src,
          cudaSurfaceObject_t dst,
    const int dstRows ,
    const int dstCols
);

// CUDA Kernel to make a color pyramid level
__global__ void
k_pyramid_color
(
    const cudaSurfaceObject_t src,
          cudaSurfaceObject_t dst,
    const int dstRows ,
    const int dstCols
);



#endif // KERNELS_CUH
