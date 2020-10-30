#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "defines.hpp"
#include "enums.hpp"

// General includes
#include "domains.hpp"
#include <algorithm>
#include <cmath>
#include <utility> // std::pair
#include <vector>

#include <QMetaType>
//----------------------------------------------------------------------
//
//   Constants
//
//----------------------------------------------------------------------

Q_DECLARE_METATYPE(v_points)

const float PI = 3.14159265359f;
const float min_blob_segment_squared = 100.f; // in pixels^2
const float min_angular_step_deg = 10.f;

//----------------------------------------------------------------------
//
//   utility functions
//
//----------------------------------------------------------------------
float makeRectangularXYc(float left, float right);
float makeAnnularRo(float right_x, float right_y, float left_x, float left_y);
float makeAnnularRi(float right_x, float right_y, float left_x, float left_y,
                    float ri_by_ro);
float makeAnnularXc(float right_x, float right_y, float left_x, float left_y);
float makeAnnularYc(float right_x, float right_y, float left_x, float left_y);
float makeblobXc(v_points contour_in);
float makeblobYc(v_points contour_in);
float best_rotation_UVUxUyVxVy(float *model_parameters); // transforms model
                                                         // parameters into a
                                                         // rotation angle in
                                                         // rad.
void copy_model_parameters(float *source, float *destination, int n_p);

#endif
