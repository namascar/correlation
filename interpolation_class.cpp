#include "interpolation_class.hpp"

float modelU_distort_x(float x, [[maybe_unused]] float y, [[maybe_unused]] float cx, [[maybe_unused]] float cy, [[maybe_unused]] float ro,
                       float *model_parameters) {
  return x + model_parameters[0];
}

float modelU_distort_y([[maybe_unused]] float x, float y, [[maybe_unused]] float cx, [[maybe_unused]] float cy, [[maybe_unused]] float ro,
                       [[maybe_unused]] float *model_parameters) {
  return y;
}

float modelUV_distort_x(float x, [[maybe_unused]] float y, [[maybe_unused]] float cx, [[maybe_unused]] float cy, [[maybe_unused]] float ro,
                        float *model_parameters) {
  return x + model_parameters[0];
}

float modelUV_distort_y([[maybe_unused]] float x, float y, [[maybe_unused]] float cx, [[maybe_unused]] float cy, [[maybe_unused]] float ro,
                        float *model_parameters) {
  return y + model_parameters[1];
}

float modelUVQ_distort_x(float x, float y, [[maybe_unused]] float cx, float cy, [[maybe_unused]] float ro,
                         float *model_parameters) {
  return x + model_parameters[0] - (y - cy) * model_parameters[2];
}

float modelUVQ_distort_y(float x, float y, float cx, [[maybe_unused]] float cy, [[maybe_unused]] float ro,
                         float *model_parameters) {
  return y + model_parameters[1] + (x - cx) * model_parameters[2];
}

float modelUVUxUyVxVy_distort_x(float x, float y, float cx, float cy, [[maybe_unused]] float ro,
                                float *model_parameters) {
  return x + model_parameters[0] + (x - cx) * model_parameters[2] +
         (y - cy) * model_parameters[3];
}

float modelUVUxUyVxVy_distort_y(float x, float y, float cx, float cy, [[maybe_unused]] float ro,
                                float *model_parameters) {
  return y + model_parameters[1] + (x - cx) * model_parameters[4] +
         (y - cy) * model_parameters[5];
}

void InterpolationClass::set_und_image(unsigned char *und_image_ptr_in,
                                       int und_image_step_in) {
  und_image_ptr =
      und_image_ptr_in; // consider checking both images are consistent
  und_image_step = und_image_step_in;
}

void InterpolationClass::set_def_image(unsigned char *def_image_ptr_in,
                                       int def_image_rows_in,
                                       int def_image_cols_in,
                                       int def_image_step_in,
                                       float *all_parameters_in) {
  def_image_ptr = def_image_ptr_in;
  def_image_rows = def_image_rows_in;
  def_image_cols = def_image_cols_in;
  def_image_step = def_image_step_in;
  all_parameters = all_parameters_in;
}

InterpolationClass_bicubic::~InterpolationClass_bicubic() {
  delete[] interpolation_vector;
  delete[] interpolation_matrix;
}

InterpolationClass_bilinear::~InterpolationClass_bilinear() {}

InterpolationClass_nearest::~InterpolationClass_nearest() {}

InterpolationClass::~InterpolationClass() {
  delete[] vec_B;
  delete[] mat_A;
  delete[] local_all_parameters;
}

void InterpolationClass_bicubic::get_interpolation(float *part_of_w_results,
                                                   float *part_of_dwdxy_results,
                                                   float xdef, float ydef) {
  if (xdef > 1.f && ydef > 1.f && xdef < def_image_cols - 2.f &&
      ydef < def_image_rows - 2.f) {
    int ix = (int)xdef;
    int iy = (int)ydef;

    get_interpolation_parameters(ix, iy);

    int index0 =
        (ix + iy * def_image_cols) *
            (number_of_colors * number_of_interpolation_parameters + 1) +
        1;

    float dx = xdef - ix + 1.f;
    float dy = ydef - iy + 1.f;

    float px[4] = {1.f, dx, dx * dx, dx * dx * dx};
    float py[4] = {1.f, dy, dy * dy, dy * dy * dy};

    // Initialize results to 0
    for (int c = 0; c < number_of_colors; c++) {
      part_of_w_results[c] = 0.f;
      part_of_dwdxy_results[c] = 0.f;
      part_of_dwdxy_results[number_of_colors + c] = 0.f;

      int index_c = index0 + c * number_of_interpolation_parameters;

      for (int jk = 0; jk < 4; jk++) {
        int index_c_jk = index_c + jk * 4;

        for (int ik = 0; ik < 4; ik++) {
          int index_c_jk_ik = index_c_jk + ik;

          //   w_result
          part_of_w_results[c] +=
              all_parameters[index_c_jk_ik] * py[jk] * px[ik];

          if (ik > 0) // dwdx_result
            part_of_dwdxy_results[c] +=
                ik * all_parameters[index_c_jk_ik] * py[jk] * px[ik - 1];

          if (jk > 0) // dwdy_result
            part_of_dwdxy_results[number_of_colors + c] +=
                jk * all_parameters[index_c_jk_ik] * py[jk - 1] * px[ik];
        }
      }
    }
  } // if bracket
  else {
    for (int c = 0; c < number_of_colors; ++c) {
      part_of_w_results[c] = 0.f;
      part_of_dwdxy_results[c] = 0.f;
      part_of_dwdxy_results[number_of_colors + c] = 0.f;
    }
    error_status = true;
    error_code = error_interpolation_out_of_image;
  }
}

void InterpolationClass_bilinear::get_interpolation(
    float *part_of_w_results, float *part_of_dwdxy_results, float xdef,
    float ydef) {
  if (xdef > 0 && ydef > 0 && xdef < def_image_cols - 1 &&
      ydef < def_image_rows - 1) {
    int ix = (int)xdef;
    int iy = (int)ydef;

    get_interpolation_parameters(ix, iy);

    int index0 =
        (ix + iy * def_image_cols) *
            (number_of_colors * number_of_interpolation_parameters + 1) +
        1;

    float dx = xdef - ix;
    float dy = ydef - iy;

    float px[2] = {1.f, dx};
    float py[2] = {1.f, dy};

    for (int c = 0; c < number_of_colors; ++c) {
      part_of_w_results[c] = 0.f;
      part_of_dwdxy_results[c] = 0.f;
      part_of_dwdxy_results[number_of_colors + c] = 0.f;

      int index_c = index0 + c * number_of_interpolation_parameters;

      for (int jk = 0; jk < 2; ++jk) {
        int index_c_jk = index_c + jk * 2;

        for (int ik = 0; ik < 2; ++ik) {
          int index_c_jk_ik = index_c_jk + ik;

          part_of_w_results[c] +=
              all_parameters[index_c_jk_ik] * py[jk] * px[ik];

          if (ik > 0)
            part_of_dwdxy_results[c] += all_parameters[index_c_jk_ik] * py[jk];

          if (jk > 0)
            part_of_dwdxy_results[number_of_colors + c] +=
                all_parameters[index_c_jk_ik] * px[ik];
        }
      }
    }
  } else {
    for (int c = 0; c < number_of_colors; ++c) {
      part_of_w_results[c] = 0.f;
      part_of_dwdxy_results[c] = 0.f;
      part_of_dwdxy_results[number_of_colors + c] = 0.f;
    }
    error_status = true;
    error_code = error_interpolation_out_of_image;
  }
}

void InterpolationClass_nearest::get_interpolation(float *part_of_w_results,
                                                   float *part_of_dwdxy_results,
                                                   float xdef, float ydef) {
  if (xdef > 0 && ydef > 0 && xdef < def_image_cols - 1 &&
      ydef < def_image_rows - 1) {
    int ix = (int)(xdef + 0.5f);
    int iy = (int)(ydef + 0.5f);

    get_interpolation_parameters(ix, iy);

    int index0 = (ix + iy * def_image_cols) *
                 (number_of_colors * number_of_interpolation_parameters + 1);

    for (int c = 0; c < number_of_colors; c++) {
      int index_c = index0 + c * number_of_interpolation_parameters;

      part_of_w_results[c] = all_parameters[index_c + 1];
      part_of_dwdxy_results[c] = all_parameters[index_c + 2];
      part_of_dwdxy_results[number_of_colors + c] = all_parameters[index_c + 3];
    }
  } else {
    for (int c = 0; c < number_of_colors; ++c) {
      part_of_w_results[c] = 0.f;
      part_of_dwdxy_results[c] = 0.f;
      part_of_dwdxy_results[number_of_colors + c] = 0.f;
    }
    error_status = true;
    error_code = error_interpolation_out_of_image;
  }
}

void InterpolationClass::get_interpolation_parameters(int ix, int iy) {
  int index0 = (ix + iy * def_image_cols) *
               (number_of_colors * number_of_interpolation_parameters + 1);

  if (all_parameters[index0] < 0.1f) {
    for (int c = 0; c < number_of_colors; ++c) {
      int index_c = index0 + 1 + c * number_of_interpolation_parameters;

      get_new_interpolation_parameters(&all_parameters[index_c], ix, iy, c);
    }

    all_parameters[index0] = 1.f;
  }
}

void InterpolationClass_bicubic::get_new_interpolation_parameters(
    float *part_of_all_parameters, int xdef, int ydef, int color_in) {
  if (error_status) {
    for (int i = 0; i < number_of_interpolation_parameters + 1; ++i)
      part_of_all_parameters[i] = 0.f;

    return;
  }

  int ix0 = xdef - 1;
  int iy0 = ydef - 1;

  int ix1 = xdef;
  int iy1 = ydef;

  int ix2 = ix0 + 2;
  int iy2 = iy0 + 2;
  int ix3 = ix0 + 3;
  int iy3 = iy0 + 3;

  int index_iy0 = def_image_step * iy0;
  int index_iy1 = def_image_step * iy1;
  int index_iy2 = def_image_step * iy2;
  int index_iy3 = def_image_step * iy3;

  int color = number_of_colors + color_in;

  int index_ix0 = ix0 * color;
  int index_ix1 = ix1 * color;
  int index_ix2 = ix2 * color;
  int index_ix3 = ix3 * color;

  float w00 = (float)def_image_ptr[index_iy0 + index_ix0];
  float w01 = (float)def_image_ptr[index_iy1 + index_ix0];
  float w02 = (float)def_image_ptr[index_iy2 + index_ix0];
  float w03 = (float)def_image_ptr[index_iy3 + index_ix0];

  float w10 = (float)def_image_ptr[index_iy0 + index_ix1];
  float w11 = (float)def_image_ptr[index_iy1 + index_ix1];
  float w12 = (float)def_image_ptr[index_iy2 + index_ix1];
  float w13 = (float)def_image_ptr[index_iy3 + index_ix1];

  float w20 = (float)def_image_ptr[index_iy0 + index_ix2];
  float w21 = (float)def_image_ptr[index_iy1 + index_ix2];
  float w22 = (float)def_image_ptr[index_iy2 + index_ix2];
  float w23 = (float)def_image_ptr[index_iy3 + index_ix2];

  float w30 = (float)def_image_ptr[index_iy0 + index_ix3];
  float w31 = (float)def_image_ptr[index_iy1 + index_ix3];
  float w32 = (float)def_image_ptr[index_iy2 + index_ix3];
  float w33 = (float)def_image_ptr[index_iy3 + index_ix3];

  // the value of the interpolat on the four middle points matches the data
  interpolation_vector[0] = w11; // this is the anchor point of the
                                 // intepolation. i.e. if dx=dy=0, W(dx,dy)=w11
  interpolation_vector[1] = w21;
  interpolation_vector[2] = w12;
  interpolation_vector[3] = w22;

  // the derivative in the x-dir is a middle finite diference dw/dx(x,y) =
  // (w[x+1,y]-w[x-1,y])/2
  interpolation_vector[4] = (w21 - w01) / 2.f;
  interpolation_vector[5] = (w31 - w11) / 2.f;
  interpolation_vector[6] = (w22 - w02) / 2.f;
  interpolation_vector[7] = (w32 - w12) / 2.f;

  // the derivative in the y-dir is a middle finite diference dw/dy(x,y) =
  // (w[x,y+1]-w[x,y-1])/2
  interpolation_vector[8] = (w12 - w10) / 2.f;
  interpolation_vector[9] = (w22 - w20) / 2.f;
  interpolation_vector[10] = (w13 - w11) / 2.f;
  interpolation_vector[11] = (w23 - w21) / 2.f;

  // the derivative in the x-y-dir is a middle finite diference dw^2/dx dy (x,y)
  // = (w[x+1,y+1]+w[x-1,y-1]-w[x-1,y+1]-w[x+1,y-1])/4
  interpolation_vector[12] = (w22 + w00 - w20 - w02) / 4.f;
  interpolation_vector[13] = (w32 + w10 - w30 - w12) / 4.f;
  interpolation_vector[14] = (w23 + w01 - w21 - w03) / 4.f;
  interpolation_vector[15] = (w33 + w11 - w31 - w13) / 4.f;

  // Solve the system - interpolation_matrix is already inverted
  for (int i = 0; i < number_of_interpolation_parameters; ++i) {
    float temp = 0.f;

    int index_i = i * number_of_interpolation_parameters;

    for (int j = 0; j < number_of_interpolation_parameters; ++j) {
      temp += interpolation_matrix[index_i + j] * interpolation_vector[j];
    }
    part_of_all_parameters[i] = temp;
  }

  return;
}

void InterpolationClass_bilinear::get_new_interpolation_parameters(
    float *part_of_all_parameters, int xdef, int ydef, int color_in) {
  if (error_status) {
    for (int i = 0; i < number_of_interpolation_parameters + 1; ++i)
      part_of_all_parameters[i] = 0.f;

    return;
  }

  int ix0 = xdef;
  int iy0 = ydef;

  int ix1 = ix0 + 1;
  int iy1 = iy0 + 1;

  int index_iy0 = def_image_step * iy0;
  int index_iy1 = def_image_step * iy1;

  int color = number_of_colors + color_in;

  int index_ix0 = ix0 * color;
  int index_ix1 = ix1 * color;

  float w00 = (float)def_image_ptr[index_iy0 + index_ix0];
  float w01 = (float)def_image_ptr[index_iy1 + index_ix0];

  float w10 = (float)def_image_ptr[index_iy0 + index_ix1];
  float w11 = (float)def_image_ptr[index_iy1 + index_ix1];

  // the value of the interpolant on the four middle points matches the data
  part_of_all_parameters[0] = w00;
  part_of_all_parameters[1] = w10 - w00;
  part_of_all_parameters[2] = w01 - w00;
  part_of_all_parameters[3] = w11 - w10 - w01 + w00;

  return;
}

void InterpolationClass_nearest::get_new_interpolation_parameters(
    float *part_of_all_parameters, int xdef, int ydef, int color_in) {
  if (error_status) {
    for (int i = 0; i < number_of_interpolation_parameters + 1; ++i)
      part_of_all_parameters[i] = 0.f;

    return;
  }

  int ix0 = xdef;
  int iy0 = ydef;

  int ix1 = ix0 + 1;
  int iy1 = iy0 + 1;

  float w00 = (float)
      def_image_ptr[def_image_step * iy0 + ix0 * number_of_colors + color_in];

  float w01 = (float)
      def_image_ptr[def_image_step * iy1 + ix0 * number_of_colors + color_in];

  float w10 = (float)
      def_image_ptr[def_image_step * iy0 + ix1 * number_of_colors + color_in];

  // the value of the interpolat on the four middle points matches the data
  part_of_all_parameters[0] = w00;
  part_of_all_parameters[1] = w10 - w00;
  part_of_all_parameters[2] = w01 - w00;

  return;
}

void InterpolationClass_bicubic::make_interpolation_matrix() {
  // float x1 = 1.f;
  // float x2 = 2.f;

  // float y1 = 1.f;
  // float y2 = 2.f;

  // //position at four grid points
  // for ( int j = 0; j < 4; j++)
  //     for ( int i = 0; i < 4; i++)
  //     {
  //         interpolation_matrix[j * 4 + i +    0 *
  //         number_of_interpolation_parameters] =  pow(y1 , j) * pow(x1 , i);
  //         interpolation_matrix[j * 4 + i +    1 *
  //         number_of_interpolation_parameters] =  pow(y1 , j) * pow(x2 , i);
  //         interpolation_matrix[j * 4 + i +    2 *
  //         number_of_interpolation_parameters] =  pow(y2 , j) * pow(x1 , i);
  //         interpolation_matrix[j * 4 + i +    3 *
  //         number_of_interpolation_parameters] =  pow(y2 , j) * pow(x2 , i);
  //     }

  // //x-derivatives at four grid points (x^0 coefficiens don't have derivative
  // contribution)
  // for ( int j = 0; j < 4; j++)
  // {
  //     interpolation_matrix[j * 4 +    4 * number_of_interpolation_parameters]
  //     =  0.f;
  //     interpolation_matrix[j * 4 +    5 * number_of_interpolation_parameters]
  //     =  0.f;
  //     interpolation_matrix[j * 4 +    6 * number_of_interpolation_parameters]
  //     =  0.f;
  //     interpolation_matrix[j * 4 +    7 * number_of_interpolation_parameters]
  //     =  0.f;
  // }

  // //x-derivatives at four grid points
  // for ( int j = 0; j < 4; j++)
  //     for ( int i = 1; i < 4; i++)
  //     {
  //         interpolation_matrix[j * 4 + i +     4 *
  //         number_of_interpolation_parameters] =  (float)i * pow(y1 , j) *
  //         pow(x1 , i - 1);
  //         interpolation_matrix[j * 4 + i +     5 *
  //         number_of_interpolation_parameters] =  (float)i * pow(y1 , j) *
  //         pow(x2 , i - 1);
  //         interpolation_matrix[j * 4 + i +     6 *
  //         number_of_interpolation_parameters] =  (float)i * pow(y2 , j) *
  //         pow(x1 , i - 1);
  //         interpolation_matrix[j * 4 + i +     7 *
  //         number_of_interpolation_parameters] =  (float)i * pow(y2 , j) *
  //         pow(x2 , i - 1);
  //     }

  // //y-derivatives at four grid points (y^0 coefficiens don't have derivative
  // contribution)
  // for ( int i = 0; i < 4; i++)
  // {
  //     interpolation_matrix[i +      8 * number_of_interpolation_parameters] =
  //     0.f;
  //     interpolation_matrix[i +      9 * number_of_interpolation_parameters] =
  //     0.f;
  //     interpolation_matrix[i +     10 * number_of_interpolation_parameters] =
  //     0.f;
  //     interpolation_matrix[i +     11 * number_of_interpolation_parameters] =
  //     0.f;
  // }

  // //y-derivatives at four grid points
  // for ( int j = 1; j < 4; j++)
  //     for ( int i = 0; i < 4; i++)
  //     {
  //         //position at four grid points
  //         interpolation_matrix[j * 4 + i +      8 *
  //         number_of_interpolation_parameters] =  (float)j * pow(y1 , j - 1) *
  //         pow(x1 , i);
  //         interpolation_matrix[j * 4 + i +      9 *
  //         number_of_interpolation_parameters] =  (float)j * pow(y1 , j - 1) *
  //         pow(x2 , i);
  //         interpolation_matrix[j * 4 + i +     10 *
  //         number_of_interpolation_parameters] =  (float)j * pow(y2 , j - 1) *
  //         pow(x1 , i);
  //         interpolation_matrix[j * 4 + i +     11 *
  //         number_of_interpolation_parameters] =  (float)j * pow(y2 , j - 1) *
  //         pow(x2 , i);
  //     }
  // //xy-derivatives at four grid points (y^0 coefficiens don't have derivative
  // contribution)
  // for ( int i = 0; i < 4; i++)
  // {
  //     interpolation_matrix[i +     12 * number_of_interpolation_parameters] =
  //     0.f;
  //     interpolation_matrix[i +     13 * number_of_interpolation_parameters] =
  //     0.f;
  //     interpolation_matrix[i +     14 * number_of_interpolation_parameters] =
  //     0.f;
  //     interpolation_matrix[i +     15 * number_of_interpolation_parameters] =
  //     0.f;
  // }

  // //xy-derivatives at four grid points (x^0 coefficiens don't have derivative
  // contribution)
  // for ( int j = 0; j < 4; j++)
  // {
  //     interpolation_matrix[j * 4 +     12 *
  //     number_of_interpolation_parameters] =  0.f;
  //     interpolation_matrix[j * 4 +     13 *
  //     number_of_interpolation_parameters] =  0.f;
  //     interpolation_matrix[j * 4 +     14 *
  //     number_of_interpolation_parameters] =  0.f;
  //     interpolation_matrix[j * 4 +     15 *
  //     number_of_interpolation_parameters] =  0.f;
  // }
  // //xy-derivatives at four grid points
  // for ( int j = 1; j < 4; j++)
  //     for ( int i = 1; i < 4; i++)
  //     {
  //         interpolation_matrix[j * 4 + i +     12 *
  //         number_of_interpolation_parameters] =  (float)i * (float)j * pow(y1
  //         , j - 1) * pow(x1 , i - 1);
  //         interpolation_matrix[j * 4 + i +     13 *
  //         number_of_interpolation_parameters] =  (float)i * (float)j * pow(y1
  //         , j - 1) * pow(x2 , i - 1);
  //         interpolation_matrix[j * 4 + i +     14 *
  //         number_of_interpolation_parameters] =  (float)i * (float)j * pow(y2
  //         , j - 1) * pow(x1 , i - 1);
  //         interpolation_matrix[j * 4 + i +     15 *
  //         number_of_interpolation_parameters] =  (float)i * (float)j * pow(y2
  //         , j - 1) * pow(x2 , i - 1);
  //     }

  // Get the exact solution instead.
  float temp1[256] = {
      16,  -20,  -20, 25,  16,   8,   -20, -10,  16,  -20, 8,   -10,  16,   8,
      8,   4,    -48, 48,  60,   -60, -32, -20,  40,  25,  -48, 48,   -24,  24,
      -32, -20,  -16, -10, 36,   -36, -45, 45,   20,  16,  -25, -20,  36,   -36,
      18,  -18,  20,  16,  10,   8,   -8,  8,    10,  -10, -4,  -4,   5,    5,
      -8,  8,    -4,  4,   -4,   -4,  -2,  -2,   -48, 60,  48,  -60,  -48,  -24,
      48,  24,   -32, 40,  -20,  25,  -32, -16,  -20, -10, 144, -144, -144, 144,
      96,  60,   -96, -60, 96,   -96, 60,  -60,  64,  40,  40,  25,   -108, 108,
      108, -108, -60, -48, 60,   48,  -72, 72,   -45, 45,  -40, -32,  -25,  -20,
      24,  -24,  -24, 24,  12,   12,  -12, -12,  16,  -16, 10,  -10,  8,    8,
      5,   5,    36,  -45, -36,  45,  36,  18,   -36, -18, 20,  -25,  16,   -20,
      20,  10,   16,  8,   -108, 108, 108, -108, -72, -45, 72,  45,   -60,  60,
      -48, 48,   -40, -25, -32,  -20, 81,  -81,  -81, 81,  45,  36,   -45,  -36,
      45,  -45,  36,  -36, 25,   20,  20,  16,   -18, 18,  18,  -18,  -9,   -9,
      9,   9,    -10, 10,  -8,   8,   -5,  -5,   -4,  -4,  -8,  10,   8,    -10,
      -8,  -4,   8,   4,   -4,   5,   -4,  5,    -4,  -2,  -4,  -2,   24,   -24,
      -24, 24,   16,  10,  -16,  -10, 12,  -12,  12,  -12, 8,   5,    8,    5,
      -18, 18,   18,  -18, -10,  -8,  10,  8,    -9,  9,   -9,  9,    -5,   -4,
      -5,  -4,   4,   -4,  -4,   4,   2,   2,    -2,  -2,  2,   -2,   2,    -2,
      1,   1,    1,   1};

  for (int i = 0; i < 256; ++i) {
    interpolation_matrix[i] = temp1[i];
  }

  // //invert the matrix
  // Eigen::Map<Eigen::MatrixXf>
  // eigen_map_interpolation_matrix(interpolation_matrix, 16,16);

  // #if DEBUG_INTERPOLATION_MAT
  //     std::cout << "eigen_map_interpolation_matrix " << std::endl;
  //     std::cout << eigen_map_interpolation_matrix << std::endl;
  // #endif

  // eigen_map_interpolation_matrix = eigen_map_interpolation_matrix.inverse();

  // #if DEBUG_INTERPOLATION_MAT
  //     std::cout << "inverted eigen_map_interpolation_matrix " << std::endl;
  //     std::cout << eigen_map_interpolation_matrix << std::endl;
  // #endif

  // #if DEBUG_INTERPOLATION_MAT
  //     eigen_map_interpolation_matrix =
  //     eigen_map_interpolation_matrix.inverse();

  //     std::cout << "inverted of inverted eigen_map_interpolation_matrix " <<
  //     std::endl;
  //     std::cout << eigen_map_interpolation_matrix << std::endl;
  // #endif
}

void InterpolationClass_bilinear::make_interpolation_matrix() {}

void InterpolationClass_nearest::make_interpolation_matrix() {}

bool InterpolationClass::get_error_status() { return error_status; }

void InterpolationClass::set_error_status(bool error_status_in) {
  error_status = error_status_in;
}

errorEnum InterpolationClass::get_error_code() { return error_code; }

void InterpolationClass::set_error_code(errorEnum error_code_in) {
  error_code = error_code_in;
}

int InterpolationClass::get_number_of_colors() { return number_of_colors; }

float *InterpolationClass::get_mat_A() { return mat_A; }

float *InterpolationClass::get_vec_B() { return vec_B; }

float InterpolationClass::get_chi() { return chi; }

int InterpolationClass::get_number_of_interpolation_parameters(
    interpolationModelEnum interpolationModel_in) {
  switch (interpolationModel_in) {
  case im_nearest:
    return 3;
  case im_bilinear:
    return 4;
  case im_bicubic:
    return 16;
  default:
    assert(false);
  }

  return -1;
}

int InterpolationClass::get_number_of_model_parameters(
    fittingModelEnum fittingModel_in) {
  switch (fittingModel_in) {
  case fm_U:
    return 1;
  case fm_UV:
    return 2;
  case fm_UVQ:
    return 3;
  case fm_UVUxUyVxVy:
    return 6;
  default:
    assert(false);
  }
  return -1;
}

void InterpolationClass::set_multiple_interpolations(
    int number_of_points_in, float *und_intensities_in,
    float *und_xy_positions_in, float *def_xy_positions_in, float *w_results_in,
    float *dwdxy_results_in, float *model_parameters_in, float und_x_center_in,
    float und_y_center_in, float *dTxydp_in) {
  error_status = false;
  error_code = error_none;

  number_of_points = number_of_points_in;

  und_intensities = und_intensities_in;
  und_xy_positions = und_xy_positions_in;
  def_xy_positions = def_xy_positions_in;
  w_results = w_results_in;
  dwdxy_results = dwdxy_results_in;

  model_parameters = model_parameters_in;

  und_x_center = und_x_center_in;
  und_y_center = und_y_center_in;

  dTxydp = dTxydp_in;
}

void InterpolationClass::get_multiple_interpolations() {
  auto start_time = std::chrono::system_clock::now();

  float *H = new float[number_of_model_parameters];

  // flush chi, mat_A and vec_B
  chi = 0.f;
  for (int p1 = 0; p1 < number_of_model_parameters; ++p1) {
    vec_B[p1] = 0.f;

    for (int p2 = 0; p2 < number_of_model_parameters; ++p2)
      mat_A[p1 * number_of_model_parameters + p2] = 0.f;
  }

  int twoTimesColors = 2 * number_of_colors;
  int twoTimesParameters = 2 * number_of_model_parameters;

  // Loop through all points assigned to this thread

  for (int i = 0; i < number_of_points; ++i) {
    // if ( error_status ) break;

    int iTimesTwoTimesColors = i * twoTimesColors;
    int iTimesTwoTimesParameters = i * twoTimesParameters;
    int iTimesTwo = i * 2;
    int iTimesTwoPlusOne = iTimesTwo + 1;
    int iTimesColors = i * number_of_colors;

    // Compute the deformed pixel locations and their gradients with respect to
    //      current parameters
    int und_ix = (int)(und_xy_positions[iTimesTwo] + 0.5f);
    int und_iy = (int)(und_xy_positions[iTimesTwoPlusOne] + 0.5f);

    //  Compute intensity values in the deformed image and its gradients with
    //  respect to
    //      def_x and def_y. i.e, puts together the w_results and dwdxy_results
    get_interpolation(&w_results[iTimesColors],
                      &dwdxy_results[iTimesTwoTimesColors],
                      def_xy_positions[iTimesTwo], // def_x , def_y );
                      def_xy_positions[iTimesTwoPlusOne]);

    for (int c = 0; c < number_of_colors; ++c) {
      float und_w = (float)und_image_ptr[und_image_step * und_iy +
                                         und_ix * number_of_colors + c];
#if DEBUG_NEWTON_RAPHSON_FAIL_DUMP
      und_intensities[iTimesColors + c] = und_w;
#endif
      float def_w = w_results[iTimesColors + c];

      float V = und_w - def_w;

      chi += V * V;

      int iTimesTwoTimesColorsPlusC = iTimesTwoTimesColors + c;
      int iTimesTwoTimesColorsPlusCPlusColors =
          iTimesTwoTimesColorsPlusC + number_of_colors;

      for (int p = 0; p < number_of_model_parameters; ++p) {
        int iTimesTwoTimesParametersPlusParameters =
            iTimesTwoTimesParameters + p;

        H[p] =
            dwdxy_results[iTimesTwoTimesColorsPlusC] * // x component
                dTxydp[iTimesTwoTimesParametersPlusParameters] +

            dwdxy_results[iTimesTwoTimesColorsPlusCPlusColors] * // y component
                dTxydp[iTimesTwoTimesParametersPlusParameters +
                       number_of_model_parameters];
      } // for p

      for (int p1 = 0; p1 < number_of_model_parameters; ++p1) {
        int p1TimesParameters = p1 * number_of_model_parameters;

        vec_B[p1] += H[p1] * V;

        for (int p2 = p1; p2 < number_of_model_parameters; ++p2) {
          mat_A[p1TimesParameters + p2] += H[p1] * H[p2];
        }
      }
    } // color loop
  }   // ipoints loop

  delete[] H;
  H = nullptr;

  auto end_time = std::chrono::system_clock::now();
  time_duration_interpolation =
      (float)std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                   start_time)
          .count() /
      1000000.f;

  debug_interpolation();
}

std::vector<InterpolationClass *> InterpolationClass::new_InterpolationClass(
    const interpolationModelEnum &interpolationModel_in,
    const fittingModelEnum &fittingModel_in, const int number_of_colors_in,
    const int number_of_threads) {
  std::vector<InterpolationClass *> n(number_of_threads);

  switch (interpolationModel_in) {
  case im_nearest: {

#if DEBUG_INTERPOLATION
    std::cout << "+++++++++++++++++++++" << std::endl;
    std::cout << "Nearest node" << std::endl;
    std::cout << "+++++++++++++++++++++" << std::endl;
#endif

    for (int i = 0; i < number_of_threads; ++i) {
      n[i] =
          new InterpolationClass_nearest{number_of_colors_in, fittingModel_in};
    }
    break;
  }
  case im_bilinear: {
#if DEBUG_INTERPOLATION
    std::cout << "+++++++++++++++++++++" << std::endl;
    std::cout << "Bilinear interpolation" << std::endl;
    std::cout << "+++++++++++++++++++++" << std::endl;
#endif

    for (int i = 0; i < number_of_threads; ++i) {
      n[i] =
          new InterpolationClass_bilinear{number_of_colors_in, fittingModel_in};
    }
    break;
  }
  case im_bicubic: {
#if DEBUG_INTERPOLATION
    std::cout << "+++++++++++++++++++++" << std::endl;
    std::cout << "Bicubic interpolation" << std::endl;
    std::cout << "+++++++++++++++++++++" << std::endl;
#endif

    for (int i = 0; i < number_of_threads; ++i) {
      n[i] =
          new InterpolationClass_bicubic{number_of_colors_in, fittingModel_in};
    }
    break;
  }
  default:
    assert(false);
  }
  return n;
}
int InterpolationClass::get_thread_id() { return thread_id; }

float InterpolationClass::get_time() { return time_duration_interpolation; }

void InterpolationClass::set_thread_id(int thread_id_in) {
  thread_id = thread_id_in;
}

void InterpolationClass::debug_interpolation() {
#if DEBUG_INTERPOLATION

  std::cout << "Interpolation results" << std::endl;
  for (int ipoint = 0; ipoint < number_of_points; ++ipoint) {
    float def_x = def_xy_positions[ipoint * 2 + 0];
    float def_y = def_xy_positions[ipoint * 2 + 1];

    std::cout << "point " << ipoint << " def W (" << def_x << "," << def_y
              << ") = ";

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

    std::cout << std::endl;
  }
  std::cout << std::endl;

#endif
}
