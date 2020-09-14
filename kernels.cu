#include "kernels.cuh"

//***************************************************************************
//***************************************************************************
//
//	CUDA Kernels
//
//***************************************************************************
//***************************************************************************

// CUDA kernel to build LS problem in GPU#0
__global__ void
k_build_LS_problem_in_GPU0(float *global_mat_A_and_vec_B_and_CHI,
                           const int number_of_model_parameters,
                           const float scaling, const float lambda) {
  int size_A_B = number_of_model_parameters * number_of_model_parameters +
                 number_of_model_parameters;

  for (int j = 0; j < size_A_B; ++j) {
    // Scale mat_A and vec_B
    global_mat_A_and_vec_B_and_CHI[j] *= scaling;
  }

  for (int j = 0; j < number_of_model_parameters; ++j) {
    // Apply lambda to diagonal elements
    global_mat_A_and_vec_B_and_CHI[j * number_of_model_parameters + j] *=
        (1.f + lambda);

    //  Copy upper part of the matrix to the lower part ( already done in
    //  Cholesky? )
    // for ( int i = 0; i < j; ++i )
    //{
    //    global_mat_A_and_vec_B_and_CHI[ j * number_of_model_parameters + i ] =
    //    global_mat_A_and_vec_B_and_CHI[ i * number_of_model_parameters + j ];
    //}
  }
}

// CUDA kernel to copy mat_A, vec_B and CHI from another GPU, stored on
//      global_mat_A_vec_B_and_CHI[ #*# + # + 1 -> 2 * ( #*# + # ) ]
//  to  global_mat_A_vec_B_and_CHI[ 0           ->       #*# + #   ] in GPU#0
__global__ void
k_aggregate_LS_problem_in_GPU0(float *global_mat_A_and_vec_B_and_CHI,
                               const int number_of_model_parameters) {
  int size_A_B_CHI = number_of_model_parameters * number_of_model_parameters +
                     number_of_model_parameters + 1;

  for (int j = 0; j < size_A_B_CHI; ++j) {
    // Copy vec_B
    global_mat_A_and_vec_B_and_CHI[j] +=
        global_mat_A_and_vec_B_and_CHI[j + size_A_B_CHI];
  }
}

// CUDA kernel to perform the global reduction
__global__ void k_global_reduction(const int number_of_blocks,
                                   // const int s,
                                   float *global_mat_A_and_vec_B_and_CHI,
                                   const int number_of_model_parameters) {
  // Allocate shared memory mat_A_and_vec_B_and_CHI
  extern __shared__ float shared_mat_A_and_vec_B_and_CHI[];

  // Contribution to global_mat_A_and_vec_B_and_CHI from block "bid" is already
  // in global memory.
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (bid * blockDim.x + tid < number_of_blocks) {
    int size_A_B_chi = number_of_model_parameters * number_of_model_parameters +
                       number_of_model_parameters + 1;

    memcpy(&shared_mat_A_and_vec_B_and_CHI[tid * size_A_B_chi],
           &global_mat_A_and_vec_B_and_CHI[(bid * blockDim.x + tid) *
                                           size_A_B_chi],
           size_A_B_chi * sizeof(float));

    __syncthreads();

    // Do a layer of reduction in shared memory. Up to 256 threads from this
    // block will reduce to 1
    for (int s = 1; s < blockDim.x; s *= 2) {

      if ((tid % (2 * s)) == 0 &&
          (tid + s) + blockDim.x * blockIdx.x <
              number_of_blocks) // don't include contribution from non-existing
                                // blocks
      {
        for (int i = 0; i < size_A_B_chi; ++i)
          shared_mat_A_and_vec_B_and_CHI[tid * size_A_B_chi + i] +=
              shared_mat_A_and_vec_B_and_CHI[(tid + s) * size_A_B_chi + i];
      }

      __syncthreads();
    }

    // Thread id = 0 includes the contribution of its block's shared ABchi into
    // the global space for another layer of global reduction
    if (tid == 0) {
      memcpy(&global_mat_A_and_vec_B_and_CHI[bid * size_A_B_chi],
             shared_mat_A_and_vec_B_and_CHI, size_A_B_chi * sizeof(float));
    }
  }
}

//#####################################################################################
//
//  MODEL kernels

// CUDA Kernel device code to perform U model
// of a blob array of points.
//
//
__global__ void k_model_U(const float *d_und_xy_positions,
                          const int number_of_points, const float *d_parameters,
                          float *d_def_xy_positions, float *d_dTxydp,
                          errorEnum *error_code) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < number_of_points) {
    // Compute deformed locations by applying the current model
    //
    //          U
    //

    float und_x = d_und_xy_positions[2 * i + 0];
    float und_y = d_und_xy_positions[2 * i + 1];

    d_def_xy_positions[2 * i + 0] = und_x + (float)d_parameters[0];
    d_def_xy_positions[2 * i + 1] = und_y;

    // Compute gradients of the Transformation with respect to the parameters

    int index = 2 * i;

    d_dTxydp[index + 0] = 1;

    d_dTxydp[index + 1] = 0;
  }

} // end of kernel "k_model_U"

// CUDA Kernel device code to perform UV model
// of a blob array of points.
//
//
__global__ void k_model_UV(const float *d_und_xy_positions,
                           const int number_of_points,
                           const float *d_parameters, float *d_def_xy_positions,
                           float *d_dTxydp, errorEnum *error_code) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < number_of_points) {
    // Compute deformed locations by applying the current model
    //
    //          U, V
    //
    const int number_of_model_parameters = 2;

    float und_x = d_und_xy_positions[2 * i + 0];
    float und_y = d_und_xy_positions[2 * i + 1];

    d_def_xy_positions[2 * i + 0] = und_x + (float)d_parameters[0];
    d_def_xy_positions[2 * i + 1] = und_y + (float)d_parameters[1];

    // Compute gradients of the Transformation with respect to the parameters

    int index = 2 * number_of_model_parameters * i;

    d_dTxydp[index + 0] = 1;
    d_dTxydp[index + 1] = 0;

    d_dTxydp[index + 2] = 0;
    d_dTxydp[index + 3] = 1;
  }

} // end of kernel "k_model_UV"

// CUDA Kernel device code to perform UVQ model
// of a blob array of points.
//
//
__global__ void k_model_UVQ(const float *d_und_xy_positions,
                            const int number_of_points,
                            const float *d_parameters, const float und_x_center,
                            const float und_y_center, float *d_def_xy_positions,
                            float *d_dTxydp, errorEnum *error_code) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < number_of_points) {
    // Compute deformed locations by applying the current model
    //
    //          U, V, Q
    //
    const int number_of_model_parameters = 3;

    float und_x = d_und_xy_positions[2 * i + 0];
    float und_y = d_und_xy_positions[2 * i + 1];

    float dx = und_x - und_x_center;
    float dy = und_y - und_y_center;

    d_def_xy_positions[2 * i + 0] =
        und_x + (float)d_parameters[0] - dy * (float)d_parameters[2];
    d_def_xy_positions[2 * i + 1] =
        und_y + (float)d_parameters[1] + dx * (float)d_parameters[2];

    // Compute gradients of the Transformation with respect to the parameters

    int index = 2 * number_of_model_parameters * i;

    d_dTxydp[index + 0] = 1;
    d_dTxydp[index + 1] = 0;
    d_dTxydp[index + 2] = -dy;

    d_dTxydp[index + 3] = 0;
    d_dTxydp[index + 4] = 1;
    d_dTxydp[index + 5] = dx;
  }

} // end of kernel "k_model_UVQ"

// CUDA Kernel device code to perform UVUxUyVxVy model
// of a blob array of points.
//
//
__global__ void
k_model_UVUxUyVxVy(const float *d_und_xy_positions, const int number_of_points,
                   const float *d_parameters, const float und_x_center,
                   const float und_y_center, float *d_def_xy_positions,
                   float *d_dTxydp, errorEnum *error_code) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < number_of_points) {
    // Compute deformed locations by applying the current model
    //
    //          U, V, Ux, Uy, Vx, Vy
    //
    const int number_of_model_parameters = 6;

    float und_x = d_und_xy_positions[2 * i + 0];
    float und_y = d_und_xy_positions[2 * i + 1];

    float dx = und_x - und_x_center;
    float dy = und_y - und_y_center;

    d_def_xy_positions[2 * i + 0] = und_x + (float)d_parameters[0] +
                                    dx * (float)d_parameters[2] +
                                    dy * (float)d_parameters[3];
    d_def_xy_positions[2 * i + 1] = und_y + (float)d_parameters[1] +
                                    dx * (float)d_parameters[4] +
                                    dy * (float)d_parameters[5];

    // Compute gradients of the Transformation with respect to the parameters

    int index = 2 * number_of_model_parameters * i;

    d_dTxydp[index + 0] = 1;
    d_dTxydp[index + 1] = 0;
    d_dTxydp[index + 2] = dx;
    d_dTxydp[index + 3] = dy;
    d_dTxydp[index + 4] = 0;
    d_dTxydp[index + 5] = 0;

    d_dTxydp[index + 6] = 0;
    d_dTxydp[index + 7] = 1;
    d_dTxydp[index + 8] = 0;
    d_dTxydp[index + 9] = 0;
    d_dTxydp[index + 10] = dx;
    d_dTxydp[index + 11] = dy;
  }

} // end of kernel "k_model_UVUxUyVxVy"

//#####################################################################################
//
//  Interpolation kernels

// CUDA Kernel device code to perform "nearest" interpolation of a blob array of
// points.
//  Also performs block reduction.
__global__ void k_interpolation_and_matrix_assembler_nearest(
    const float *d_def_xy_positions, const float *d_und_intensities,
    const int number_of_points, const float *d_dTxydp,
    const int number_of_model_parameters, const int number_of_colors,
    float *global_mat_A_and_vec_B_and_CHI, cudaTextureObject_t def_tex,
    errorEnum *error_code) {
  // dynamically allocate shared mat_A_and_vec_B_and_CHI
  extern __shared__ float shared_mat_A_and_vec_B_and_CHI[];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (i < number_of_points) {
    float def_x = d_def_xy_positions[2 * i + 0] + 0.5f;
    float def_y = d_def_xy_positions[2 * i + 1] + 0.5f;

    int ix0 = (int)def_x;
    int iy0 = (int)def_y;

    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    //  local memory space
    int size_A = number_of_model_parameters * number_of_model_parameters;
    int size_A_B = size_A + number_of_model_parameters;
    int size_A_B_chi = size_A_B + 1;

    float H[6]; // max number of parameters = 6 (for UVUxUyVxVy)

    // flush this thread's mat_A, vec_B and chi
    for (int index = 0; index < size_A_B_chi; ++index)
      shared_mat_A_and_vec_B_and_CHI[tid * size_A_B_chi + index] = 0.f;

    int index_dTdx = i * number_of_model_parameters * 2;
    int index_dTdy = index_dTdx + number_of_model_parameters;

    for (int c = 0; c < number_of_colors; ++c) {
      float und_w = d_und_intensities[i * number_of_colors + c];

      //  Interpolated deformed image
      float def_w = (float)tex2D<unsigned char>(
          def_tex, (number_of_colors * ix0 + c), iy0);

      //  gradient dw/dx of the Interpolated deformed image
      float def_dwdx = (float)tex2D<unsigned char>(
                           def_tex, (number_of_colors * ix1 + c), iy0) -
                       (float)tex2D<unsigned char>(
                           def_tex, (number_of_colors * ix0 + c), iy0);

      //  gradient dw/dy of the Interpolated deformed image
      float def_dwdy = (float)tex2D<unsigned char>(
                           def_tex, (number_of_colors * ix0 + c), iy1) -
                       (float)tex2D<unsigned char>(
                           def_tex, (number_of_colors * ix0 + c), iy0);

      float V = und_w - def_w;
      //  chi contribution included to the block-shared space
      shared_mat_A_and_vec_B_and_CHI[tid * size_A_B_chi + size_A_B] += V * V;

      for (int p = 0; p < number_of_model_parameters; ++p)
        H[p] = def_dwdx * d_dTxydp[index_dTdx + p] +
               def_dwdy * d_dTxydp[index_dTdy + p];

      // Include the contribution of this color/point to the block-shared vector
      // and
      //      symmetric matrix
      int index_A_B = tid * size_A_B_chi;
      int index_B = index_A_B + size_A;

      for (int p1 = 0; p1 < number_of_model_parameters; ++p1) {
        shared_mat_A_and_vec_B_and_CHI[index_B + p1] += H[p1] * V;

        int index_A = index_A_B + p1 * number_of_model_parameters;

        for (int p2 = 0; p2 < number_of_model_parameters; ++p2)
          shared_mat_A_and_vec_B_and_CHI[index_A + p2] += H[p1] * H[p2];
      }

    } // for c

    __syncthreads();

    // Do block reduction in shared memory since we are using 256 = 2^8 threads
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      if ((tid % (2 * s)) == 0 &&
          (tid + s) + blockDim.x * blockIdx.x <
              number_of_points) // don't include contribution from the last
                                // block
      {
        for (int index = 0; index < size_A_B_chi; ++index)
          shared_mat_A_and_vec_B_and_CHI[tid * size_A_B_chi + index] +=
              shared_mat_A_and_vec_B_and_CHI[(tid + s) * size_A_B_chi + index];
      }

      __syncthreads();
    }

    // Thread id = 0 includes the contribution of its block into the global
    // space for later global reduction
    if (tid == 0) {
      for (unsigned int index = 0; index < size_A_B_chi; ++index)
        global_mat_A_and_vec_B_and_CHI[bid * size_A_B_chi + index] =
            shared_mat_A_and_vec_B_and_CHI[index];
    }

  } // if (i < number_of_points)
} // "Nearest" interpolation kernel

// CUDA Kernel device code to perform bilinear interpolation of a blob array of
// points.
//  Also performs block reduction.
__global__ void k_interpolation_and_matrix_assembler_bilinear(
    const float *d_def_xy_positions, const float *d_und_intensities,
    const int number_of_points, const float *d_dTxydp,
    const int number_of_model_parameters, const int number_of_colors,
    float *global_mat_A_and_vec_B_and_CHI, cudaTextureObject_t def_tex,
    errorEnum *error_code) {
  // dynamically allocate shared mat_A_and_vec_B_and_CHI
  extern __shared__ float shared_mat_A_and_vec_B_and_CHI[];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (i < number_of_points) {
    float def_x = d_def_xy_positions[2 * i + 0];
    float def_y = d_def_xy_positions[2 * i + 1];

    int ix0 = (int)def_x;
    int iy0 = (int)def_y;

    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    float dx = def_x - ix0;
    float dy = def_y - iy0;

    //  local memory space
    int size_A = number_of_model_parameters * number_of_model_parameters;
    int size_A_B = size_A + number_of_model_parameters;
    int size_A_B_chi = size_A_B + 1;

    float H[6]; // max number of parameters = 6 (for UVUxUyVxVy)

    // flush this thread's mat_A, vec_B and chi
    for (int index = 0; index < size_A_B_chi; ++index)
      shared_mat_A_and_vec_B_and_CHI[tid * size_A_B_chi + index] = 0.f;

    int index_dTdx = i * number_of_model_parameters * 2;
    int index_dTdy = index_dTdx + number_of_model_parameters;

    for (int c = 0; c < number_of_colors; ++c) {
      float und_w = d_und_intensities[i * number_of_colors + c];

      // the value of the interpolant on the four middle points matches the data
      float all_parameters[4] = {
          (float)tex2D<unsigned char>(def_tex, (number_of_colors * ix0 + c),
                                      iy0),

          (float)tex2D<unsigned char>(def_tex, (number_of_colors * ix1 + c),
                                      iy0) -
              (float)tex2D<unsigned char>(def_tex, (number_of_colors * ix0 + c),
                                          iy0),

          (float)tex2D<unsigned char>(def_tex, (number_of_colors * ix0 + c),
                                      iy1) -
              (float)tex2D<unsigned char>(def_tex, (number_of_colors * ix0 + c),
                                          iy0),

          (float)tex2D<unsigned char>(def_tex, (number_of_colors * ix1 + c),
                                      iy1) -
              (float)tex2D<unsigned char>(def_tex, (number_of_colors * ix1 + c),
                                          iy0) -
              (float)tex2D<unsigned char>(def_tex, (number_of_colors * ix0 + c),
                                          iy1) +
              (float)tex2D<unsigned char>(def_tex, (number_of_colors * ix0 + c),
                                          iy0)};

      //  Interpolated deformed image
      float def_w = all_parameters[0] + all_parameters[1] * dx +
                    all_parameters[2] * dy + all_parameters[3] * dx * dy;

      //  gradient dw/dx of the Interpolated deformed image
      float def_dwdx = all_parameters[1] + all_parameters[3] * dy;

      //  gradient dw/dy of the Interpolated deformed image
      float def_dwdy = all_parameters[2] + all_parameters[3] * dx;

      float V = und_w - def_w;
      //  chi contribution included to the block-shared space
      shared_mat_A_and_vec_B_and_CHI[tid * size_A_B_chi + size_A_B] += V * V;

      for (int p = 0; p < number_of_model_parameters; ++p)
        H[p] = def_dwdx * d_dTxydp[index_dTdx + p] +
               def_dwdy * d_dTxydp[index_dTdy + p];

      // Include the contribution of this color/point to the block-shared vector
      // and
      //      symmetric matrix
      int index_A_B = tid * size_A_B_chi;
      int index_B = index_A_B + size_A;

      for (int p1 = 0; p1 < number_of_model_parameters; ++p1) {
        shared_mat_A_and_vec_B_and_CHI[index_B + p1] += H[p1] * V;

        int index_A = index_A_B + p1 * number_of_model_parameters;

        for (int p2 = 0; p2 < number_of_model_parameters; ++p2)
          shared_mat_A_and_vec_B_and_CHI[index_A + p2] += H[p1] * H[p2];
      }

    } // for c

    __syncthreads();

    // Do block reduction in shared memory since we are using 256 = 2^8 threads
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      if ((tid % (2 * s)) == 0 &&
          (tid + s) + blockDim.x * blockIdx.x <
              number_of_points) // don't include contribution from the last
                                // block
      {
        for (int index = 0; index < size_A_B_chi; ++index)
          shared_mat_A_and_vec_B_and_CHI[tid * size_A_B_chi + index] +=
              shared_mat_A_and_vec_B_and_CHI[(tid + s) * size_A_B_chi + index];
      }

      __syncthreads();
    }

    // Thread id = 0 includes the contribution of its block into the global
    // space for later global reduction
    if (tid == 0) {
      for (unsigned int index = 0; index < size_A_B_chi; ++index)
        global_mat_A_and_vec_B_and_CHI[bid * size_A_B_chi + index] =
            shared_mat_A_and_vec_B_and_CHI[index];
    }

  } // if (i < number_of_points)
} // bilinear interpolation kernel

// CUDA Kernel device code to perform bicubic interpolation of a blob array of
// points.
//  Also performs block reduction.
__global__ void k_interpolation_and_matrix_assembler_bicubic(
    const float *d_def_xy_positions, const float *d_und_intensities,
    const int number_of_points, const float *d_dTxydp,
    const int number_of_model_parameters, const int number_of_colors,
    float *global_mat_A_and_vec_B_and_CHI, cudaTextureObject_t def_tex,
    errorEnum *error_code) {
  // dynamically allocate shared mat_A_and_vec_B_and_CHI
  extern __shared__ float shared_mat_A_and_vec_B_and_CHI[];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (i < number_of_points) {
    const float d_bicubic_interpolation_matrix[256] = {
        16,  -20,  -20, 25,   16,   8,    -20, -10, 16,   -20, 8,   -10, 16,
        8,   8,    4,   -48,  48,   60,   -60, -32, -20,  40,  25,  -48, 48,
        -24, 24,   -32, -20,  -16,  -10,  36,  -36, -45,  45,  20,  16,  -25,
        -20, 36,   -36, 18,   -18,  20,   16,  10,  8,    -8,  8,   10,  -10,
        -4,  -4,   5,   5,    -8,   8,    -4,  4,   -4,   -4,  -2,  -2,  -48,
        60,  48,   -60, -48,  -24,  48,   24,  -32, 40,   -20, 25,  -32, -16,
        -20, -10,  144, -144, -144, 144,  96,  60,  -96,  -60, 96,  -96, 60,
        -60, 64,   40,  40,   25,   -108, 108, 108, -108, -60, -48, 60,  48,
        -72, 72,   -45, 45,   -40,  -32,  -25, -20, 24,   -24, -24, 24,  12,
        12,  -12,  -12, 16,   -16,  10,   -10, 8,   8,    5,   5,   36,  -45,
        -36, 45,   36,  18,   -36,  -18,  20,  -25, 16,   -20, 20,  10,  16,
        8,   -108, 108, 108,  -108, -72,  -45, 72,  45,   -60, 60,  -48, 48,
        -40, -25,  -32, -20,  81,   -81,  -81, 81,  45,   36,  -45, -36, 45,
        -45, 36,   -36, 25,   20,   20,   16,  -18, 18,   18,  -18, -9,  -9,
        9,   9,    -10, 10,   -8,   8,    -5,  -5,  -4,   -4,  -8,  10,  8,
        -10, -8,   -4,  8,    4,    -4,   5,   -4,  5,    -4,  -2,  -4,  -2,
        24,  -24,  -24, 24,   16,   10,   -16, -10, 12,   -12, 12,  -12, 8,
        5,   8,    5,   -18,  18,   18,   -18, -10, -8,   10,  8,   -9,  9,
        -9,  9,    -5,  -4,   -5,   -4,   4,   -4,  -4,   4,   2,   2,   -2,
        -2,  2,    -2,  2,    -2,   1,    1,   1,   1};

    float def_x = d_def_xy_positions[2 * i + 0];
    float def_y = d_def_xy_positions[2 * i + 1];

    int ix0 = (int)def_x - 1;
    int iy0 = (int)def_y - 1;

    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    int ix2 = ix0 + 2;
    int iy2 = iy0 + 2;
    int ix3 = ix0 + 3;
    int iy3 = iy0 + 3;

    float dx = def_x - ix0;
    float dy = def_y - iy0;

    //  local memory space
    int size_A = number_of_model_parameters * number_of_model_parameters;
    int size_A_B = size_A + number_of_model_parameters;
    int size_A_B_chi = size_A_B + 1;

    float H[6]; // max number of parameters = 6 (for UVUxUyVxVy)

    // flush this thread's mat_A, vec_B and chi
    for (int index = 0; index < size_A_B_chi; ++index) {
      shared_mat_A_and_vec_B_and_CHI[tid * size_A_B_chi + index] = 0.f;
    }

    int index_dTdx = i * number_of_model_parameters * 2;
    int index_dTdy = index_dTdx + number_of_model_parameters;

    float px[5] = {0.f, 1.f, dx, dx * dx, dx * dx * dx};
    float py[5] = {0.f, 1.f, dy, dy * dy, dy * dy * dy};

    for (int c = 0; c < number_of_colors; ++c) {
      float und_w = d_und_intensities[i * number_of_colors + c];

      //*error_code = error_correlation_max_iters_reached;

      //  Query deformed image at the interpolation grid points
      int index_x0 = number_of_colors * ix0 + c;
      int index_x1 = number_of_colors * ix1 + c;
      int index_x2 = number_of_colors * ix2 + c;
      int index_x3 = number_of_colors * ix3 + c;

      float w00 = (float)tex2D<unsigned char>(def_tex, index_x0, iy0);
      float w01 = (float)tex2D<unsigned char>(def_tex, index_x0, iy1);
      float w02 = (float)tex2D<unsigned char>(def_tex, index_x0, iy2);
      float w03 = (float)tex2D<unsigned char>(def_tex, index_x0, iy3);

      float w10 = (float)tex2D<unsigned char>(def_tex, index_x1, iy0);
      float w11 = (float)tex2D<unsigned char>(def_tex, index_x1, iy1);
      float w12 = (float)tex2D<unsigned char>(def_tex, index_x1, iy2);
      float w13 = (float)tex2D<unsigned char>(def_tex, index_x1, iy3);

      float w20 = (float)tex2D<unsigned char>(def_tex, index_x2, iy0);
      float w21 = (float)tex2D<unsigned char>(def_tex, index_x2, iy1);
      float w22 = (float)tex2D<unsigned char>(def_tex, index_x2, iy2);
      float w23 = (float)tex2D<unsigned char>(def_tex, index_x2, iy3);

      float w30 = (float)tex2D<unsigned char>(def_tex, index_x3, iy0);
      float w31 = (float)tex2D<unsigned char>(def_tex, index_x3, iy1);
      float w32 = (float)tex2D<unsigned char>(def_tex, index_x3, iy2);
      float w33 = (float)tex2D<unsigned char>(def_tex, index_x3, iy3);

      //  Constructing the interpolation matrix problem
      float interpolation_vector[16]; // number_of_interpolation_parameters = 16

      interpolation_vector[0] = w11; // this is the anchor point of the
                                     // intepolation. i.e. if dx=dy=0,
                                     // W(dx,dy)=w11
      interpolation_vector[1] = w21;
      interpolation_vector[2] = w12;
      interpolation_vector[3] = w22;

      //  the derivative in the x-dir is a middle finite diference dw/dx(x,y) =
      //  (w[x+1,y]-w[x-1,y])/2
      interpolation_vector[4] = (w21 - w01) / 2.f;
      interpolation_vector[5] = (w31 - w11) / 2.f;
      interpolation_vector[6] = (w22 - w02) / 2.f;
      interpolation_vector[7] = (w32 - w12) / 2.f;

      //  the derivative in the y-dir is a middle finite diference dw/dy(x,y) =
      //  (w[x,y+1]-w[x,y-1])/2
      interpolation_vector[8] = (w12 - w10) / 2.f;
      interpolation_vector[9] = (w22 - w20) / 2.f;
      interpolation_vector[10] = (w13 - w11) / 2.f;
      interpolation_vector[11] = (w23 - w21) / 2.f;

      //  the derivative in the x-y-dir is a middle finite diference dw^2/dx dy
      //  (x,y) = (w[x+1,y+1]+w[x-1,y-1]-w[x-1,y+1]-w[x+1,y-1])/4
      interpolation_vector[12] = (w22 + w00 - w20 - w02) / 4.f;
      interpolation_vector[13] = (w32 + w10 - w30 - w12) / 4.f;
      interpolation_vector[14] = (w23 + w01 - w21 - w03) / 4.f;
      interpolation_vector[15] = (w33 + w11 - w31 - w13) / 4.f;

      //  Solve the system - interpolation_matrix is already inverted
      float interpolation_parameters[16]; // number_of_interpolation_parameters
                                          // = 16

      for (int ik = 0; ik < 16; ++ik) {
        interpolation_parameters[ik] = 0;

        for (int jk = 0; jk < 16; ++jk) {
          interpolation_parameters[ik] +=
              d_bicubic_interpolation_matrix[ik * 16 + jk] *
              interpolation_vector[jk];
        }
      }

      // Initialize results to 0
      float def_w = 0.f;
      float def_dwdx = 0.f;
      float def_dwdy = 0.f;

      for (int jk = 0; jk < 4; jk++) {
        int index_jk = jk * 4;

        for (int ik = 0; ik < 4; ik++) {

          int index_jk_ik = index_jk + ik;

          def_w +=
              interpolation_parameters[index_jk_ik] * py[jk + 1] * px[ik + 1];

          def_dwdx +=
              ik * interpolation_parameters[index_jk_ik] * py[jk + 1] * px[ik];

          def_dwdy +=
              jk * interpolation_parameters[index_jk_ik] * py[jk] * px[ik + 1];

        } // ik bracket
      }   // jk bracket

      float V = und_w - def_w;

      //  Add chi contribution included to the block-shared space
      shared_mat_A_and_vec_B_and_CHI[tid * size_A_B_chi + size_A_B] += V * V;

      for (int p = 0; p < number_of_model_parameters; ++p) {
        H[p] = def_dwdx * d_dTxydp[index_dTdx + p] +
               def_dwdy * d_dTxydp[index_dTdy + p];
      }

      // Include the contribution of this color/point to the block-shared vector
      // and
      //      symmetric matrix
      int index_A_B = tid * size_A_B_chi;
      int index_B = index_A_B + size_A;

      for (int p1 = 0; p1 < number_of_model_parameters; ++p1) {
        shared_mat_A_and_vec_B_and_CHI[index_B + p1] += H[p1] * V;

        int index_A = index_A_B + p1 * number_of_model_parameters;

        for (int p2 = 0; p2 < number_of_model_parameters; ++p2) {
          shared_mat_A_and_vec_B_and_CHI[index_A + p2] += H[p1] * H[p2];
        }
      }

    } // for c

    __syncthreads();

    // Do block reduction in shared memory since we are using 256 = 2^8 threads
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      if ((tid % (2 * s)) == 0 &&
          (tid + s) + blockDim.x * blockIdx.x <
              number_of_points) // don't include contribution from the last
                                // block
      {
        for (int index = 0; index < size_A_B_chi; ++index) {
          shared_mat_A_and_vec_B_and_CHI[tid * size_A_B_chi + index] +=
              shared_mat_A_and_vec_B_and_CHI[(tid + s) * size_A_B_chi + index];
        }
      }
      __syncthreads();
    }

    // Thread id = 0 includes the contribution of its block into the global
    // space for later global reduction
    if (tid == 0) {
      for (unsigned int index = 0; index < size_A_B_chi; ++index) {
        global_mat_A_and_vec_B_and_CHI[bid * size_A_B_chi + index] =
            shared_mat_A_and_vec_B_and_CHI[index];
      }
    }

  } // if (i < number_of_points)
} // bicubic interpolation kernel

//#####################################################################################
//
//  Pyramid kernels

// CUDA Kernel to make a black and white pyramid level
__global__ void k_pyramid_bw(const cudaTextureObject_t src,
                             cudaSurfaceObject_t dst, const int dstRows,
                             const int dstCols) {
  int iDst = blockDim.x * blockIdx.x + threadIdx.x;
  int jDst = blockDim.y * blockIdx.y + threadIdx.y;

  if (iDst < dstCols && jDst < dstRows) {
    int iSrc2 = iDst * 2;
    int iSrc0 = iSrc2 - 2;
    int iSrc1 = iSrc2 - 1;
    int iSrc3 = iSrc2 + 1;
    int iSrc4 = iSrc2 + 2;

    int jSrc2 = jDst * 2;
    int jSrc0 = jSrc2 - 2;
    int jSrc1 = jSrc2 - 1;
    int jSrc3 = jSrc2 + 1;
    int jSrc4 = jSrc2 + 2;

    unsigned char pyramid =
        (unsigned char)(0.0025f *
                            (float)tex2D<unsigned char>(src, iSrc0, jSrc0) +
                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc1, jSrc0) +
                        0.0200f *
                            (float)tex2D<unsigned char>(src, iSrc2, jSrc0) +
                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc3, jSrc0) +
                        0.0025f *
                            (float)tex2D<unsigned char>(src, iSrc4, jSrc0) +

                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc0, jSrc1) +
                        0.0625f *
                            (float)tex2D<unsigned char>(src, iSrc1, jSrc1) +
                        0.1000f *
                            (float)tex2D<unsigned char>(src, iSrc2, jSrc1) +
                        0.0625f *
                            (float)tex2D<unsigned char>(src, iSrc3, jSrc1) +
                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc4, jSrc1) +

                        0.0200f *
                            (float)tex2D<unsigned char>(src, iSrc0, jSrc2) +
                        0.1000f *
                            (float)tex2D<unsigned char>(src, iSrc1, jSrc2) +
                        0.1600f *
                            (float)tex2D<unsigned char>(src, iSrc2, jSrc2) +
                        0.1000f *
                            (float)tex2D<unsigned char>(src, iSrc3, jSrc2) +
                        0.0200f *
                            (float)tex2D<unsigned char>(src, iSrc4, jSrc2) +

                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc0, jSrc3) +
                        0.0625f *
                            (float)tex2D<unsigned char>(src, iSrc1, jSrc3) +
                        0.1000f *
                            (float)tex2D<unsigned char>(src, iSrc2, jSrc3) +
                        0.0625f *
                            (float)tex2D<unsigned char>(src, iSrc3, jSrc3) +
                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc4, jSrc3) +

                        0.0025f *
                            (float)tex2D<unsigned char>(src, iSrc0, jSrc4) +
                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc1, jSrc4) +
                        0.0200f *
                            (float)tex2D<unsigned char>(src, iSrc2, jSrc4) +
                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc3, jSrc4) +
                        0.0025f *
                            (float)tex2D<unsigned char>(src, iSrc4, jSrc4));

    surf2Dwrite(pyramid, dst, iDst, jDst);
  }
} // end of kernel "k_pyramid_bw"

// CUDA Kernel to make a color pyramid level
__global__ void k_pyramid_color(const cudaTextureObject_t src,
                                cudaSurfaceObject_t dst, const int dstRows,
                                const int dstCols) {
  int iDst = blockDim.x * blockIdx.x + threadIdx.x;
  int jDst = blockDim.y * blockIdx.y + threadIdx.y;

  if (iDst < dstCols * 3 && jDst < dstRows) {
    int iSrc2 = iDst / 3 * 6 + iDst % 3;
    int iSrc0 = iSrc2 - 6;
    int iSrc1 = iSrc2 - 3;
    int iSrc3 = iSrc2 + 3;
    int iSrc4 = iSrc2 + 6;

    int jSrc2 = jDst * 2;
    int jSrc0 = jSrc2 - 2;
    int jSrc1 = jSrc2 - 1;
    int jSrc3 = jSrc2 + 1;
    int jSrc4 = jSrc2 + 2;

    unsigned char pyramid =
        (unsigned char)(0.0025f *
                            (float)tex2D<unsigned char>(src, iSrc0, jSrc0) +
                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc1, jSrc0) +
                        0.0200f *
                            (float)tex2D<unsigned char>(src, iSrc2, jSrc0) +
                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc3, jSrc0) +
                        0.0025f *
                            (float)tex2D<unsigned char>(src, iSrc4, jSrc0) +

                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc0, jSrc1) +
                        0.0625f *
                            (float)tex2D<unsigned char>(src, iSrc1, jSrc1) +
                        0.1000f *
                            (float)tex2D<unsigned char>(src, iSrc2, jSrc1) +
                        0.0625f *
                            (float)tex2D<unsigned char>(src, iSrc3, jSrc1) +
                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc4, jSrc1) +

                        0.0200f *
                            (float)tex2D<unsigned char>(src, iSrc0, jSrc2) +
                        0.1000f *
                            (float)tex2D<unsigned char>(src, iSrc1, jSrc2) +
                        0.1600f *
                            (float)tex2D<unsigned char>(src, iSrc2, jSrc2) +
                        0.1000f *
                            (float)tex2D<unsigned char>(src, iSrc3, jSrc2) +
                        0.0200f *
                            (float)tex2D<unsigned char>(src, iSrc4, jSrc2) +

                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc0, jSrc3) +
                        0.0625f *
                            (float)tex2D<unsigned char>(src, iSrc1, jSrc3) +
                        0.1000f *
                            (float)tex2D<unsigned char>(src, iSrc2, jSrc3) +
                        0.0625f *
                            (float)tex2D<unsigned char>(src, iSrc3, jSrc3) +
                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc4, jSrc3) +

                        0.0025f *
                            (float)tex2D<unsigned char>(src, iSrc0, jSrc4) +
                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc1, jSrc4) +
                        0.0200f *
                            (float)tex2D<unsigned char>(src, iSrc2, jSrc4) +
                        0.0125f *
                            (float)tex2D<unsigned char>(src, iSrc3, jSrc4) +
                        0.0025f *
                            (float)tex2D<unsigned char>(src, iSrc4, jSrc4));

    surf2Dwrite(pyramid, dst, iDst, jDst);
  }
} // end of kernel "k_pyramid_color"
