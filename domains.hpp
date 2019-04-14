#ifndef DOMAINS_HPP
#define DOMAINS_HPP

// General includes
#include <utility>      // std::pair
#include <vector>

#include "enums.hpp"

typedef std::vector< std::pair< float , float > > v_points;

//----------------------------------------------------------------------
//
//   structs
//
//----------------------------------------------------------------------

struct rectangularDomainStruct
{
    bool ready;

    float x_center;
    float y_center;

    float x_begin;
    float y_begin;
    float x_end;
    float y_end;

    int horizontal_subdivisions;
    int vertical_subdivisions;
};

struct annularDomainStruct
{
    bool ready;

    float x_center;
    float y_center;

    float r_inside;
    float r_outside;
    float ri_by_ro;

    int angular_subdivisions;
    int radial_subdivisions;
};

struct blobDomainStruct
{
    bool ready;

    float x_center;
    float y_center;

    float x_scale;
    float y_scale;

    v_points xy_contour;
};

struct frame_results
{
    bool  empty;

    float und_center_x;
    float und_center_y;
    float und_angle;
    float und_e;

    float und_global_ro;
    float und_global_ri;
    float und_global_angle;
    float und_global_center_x;
    float und_global_center_y;
    float und_global_e;

    v_points und_contour;
    v_points und_inside_points;

    float def_center_x;
    float def_center_y;
    float def_angle;
    float def_e;

    float def_global_ro;
    float def_global_ri;
    float def_global_angle;
    float def_global_center_x;
    float def_global_center_y;
    float def_global_e;

    v_points def_contour;
    v_points def_inside_points;

    float *resulting_parameters;
    float *previous_resulting_parameters; // to compute d_initial_guess for Eulerian deformations
    float *initial_guess;

    int number_of_points;
    float chi;
    int iterations;
    bool error_status;
    errorEnum error_code;

    float past_und_center_x;
    float past_und_center_y;

    float center_bias_x;
    float center_bias_y;
};

struct CorrelationResult
{
    float       resultingParameters[ 6 ];
    float       chi;
    int         numberOfPoints;
    int         iterations;
    errorEnum   errorCode   { error_none };
    float       undCenterX;
    float       undCenterY;
};
#endif // DOMAINS_HPP
