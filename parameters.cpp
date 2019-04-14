#include <parameters.hpp>

//----------------------------------------------------------------------
//
//   utility functions
//
//----------------------------------------------------------------------
float makeRectangularXYc(float left, float right)
{
    return  ( left  + right ) * 0.5f;
}

float makeAnnularRo( float right_x, float right_y, float left_x, float left_y )
{
    return  0.5f * std::min( abs( right_x - left_x ), abs( right_y - left_y ) );
}

float makeAnnularRi( float right_x, float right_y, float left_x, float left_y, float ri_by_ro )
{
    return  ri_by_ro * makeAnnularRo( right_x, right_y, left_x, left_y );
}

float makeAnnularXc( float right_x, float right_y, float left_x, float left_y )
{
    return  ( right_x > left_x ? (float) left_x + makeAnnularRo( right_x, right_y, left_x, left_y ) : (float) left_x - makeAnnularRo( right_x, right_y, left_x, left_y ) );
}

float makeAnnularYc( float right_x, float right_y, float left_x, float left_y )
{
    return  ( right_y > left_y ? (float) left_y + makeAnnularRo( right_x, right_y, left_x, left_y ) : (float) left_y - makeAnnularRo( right_x, right_y, left_x, left_y ) );
}

float makeblobXc(v_points contour_in)
{
    float xc = 0;

    for ( int i = 0; i < (int) contour_in.size(); ++i )
    {
        xc += contour_in[i].first;
    }

    return  xc / (float) contour_in.size();
}

float makeblobYc( v_points contour_in )
{
    float yc = 0;

    for ( int i = 0; i < (int)contour_in.size(); ++i )
    {
        yc += contour_in[i].second;
    }

    return  yc / (float) contour_in.size();
}

float best_rotation_UVUxUyVxVy(float *model_parameters)
{
    return atan2( (model_parameters[4] - model_parameters[3] ) , ( model_parameters[2] + model_parameters[5] + 2.f ) );
}

void copy_model_parameters( float *source, float *destination, int n_p )
{
    for ( int p = 0; p < n_p; ++p )
    {
        destination[p] = source[p];
    }
}
