#include "pyramid_class.h"
#include "defines.hpp"

void Pyramid_class::clear_und_images()
{
    for ( unsigned int ilevel = 1 ; ilevel < und_images.size() ; ++ilevel )                     // Only delete levels 1 and above. Level 0 is cleared by openCV
    {
        delete[] und_images[ ilevel ];
    }
    und_images.clear();
}

void Pyramid_class::clear_def_images()
{
    for ( unsigned int ilevel = 1 ; ilevel < def_images.size() ; ++ilevel )                     // Only delete levels 1 and above. Level 0 is cleared by openCV
    {
        delete[] def_images[ ilevel ];
    }
    def_images.clear();
}

void Pyramid_class::clear_nxt_images()
{
    for ( unsigned int ilevel = 1 ; ilevel < nxt_images.size() ; ++ilevel )                     // Only delete levels 1 and above. Level 0 is cleared by openCV
    {
        delete[] nxt_images[ ilevel ];
    }
    nxt_images.clear();
}

void Pyramid_class::clear_xy_positions()
{
    int firstLevel = ( start == 0 ? step : start ) ;

    for ( unsigned int ilevel = firstLevel ; ilevel < xy_positions.size() ; ilevel += step )     // Only delete levels 1 and above. Level 0 is cleared by manager
    {
        delete[] xy_positions[ ilevel ];
    }
    xy_positions.clear();
    number_of_points.clear();
}

void Pyramid_class::clear_all_interpolation_parameters()
{
    for ( unsigned int ilevel = start ; ilevel < all_interpolation_parameters.size() ; ilevel += step )
    {
        delete[] all_interpolation_parameters[ ilevel ];
    }
    all_interpolation_parameters.clear();
}

std::vector< unsigned char* > Pyramid_class::make_pyramid( std::vector< unsigned char* > pyramid_in , ImageType imageType )
{
#if DEBUG_PYRAMID
    printf( "Pyramid::make_pyramid %d\n" , imageType );
    fflush( stdout );
#endif

    int sourceCols, sourceRows ;

    switch( imageType )
    {
    case imageType_und:
        sourceCols              = und_image.cols;
        sourceRows              = und_image.rows;
        break;

    case imageType_def:
        sourceCols              = def_image.cols;
        sourceRows              = def_image.rows;
        break;

    case imageType_nxt:
        sourceCols              = nxt_image.cols;
        sourceRows              = nxt_image.rows;
        break;

    default:
        assert ( false );
        break;
    }

    float kernelMaker[ 5 ] = { 0.05f , 0.25f , 0.4f , 0.25f , 0.05f };
    float kernel[ 25 ];

    for ( int i = 0 ; i < 5 ; ++i )
    {
        for ( int j = 0 ; j < 5 ; ++j )
        {
            kernel[ 5 * j + i ] = kernelMaker[ i ] * kernelMaker[ j ];
        }
    }

    for ( int ilevel = 1 ; ilevel <= stop ; ++ilevel )
    {
        int sourceStep = sourceCols  * number_of_colors;
        int targetStep = sourceStep / 2;
        int targetCols = sourceCols / 2;
        int targetRows = sourceRows / 2;

        pyramid_in[ ilevel ] = new unsigned char[ targetCols * targetRows * number_of_colors ]();

        for ( int tj = 1 ; tj < targetRows - 1 ; ++tj )
        {
            for ( int ti = 1 ; ti < targetCols - 1 ; ++ti )
            {
                for ( int c = 0 ; c < number_of_colors ; ++c )
                {
                    int si = ti * 2;
                    int sj = tj * 2;

                    float addition = 0.f;

                    for ( int dj = -2 ; dj <= 2 ; ++dj )
                    {
                        for ( int di = -2 ; di <= 2 ; ++di )
                        {
                            unsigned char src = pyramid_in[ ilevel - 1 ][ sourceStep * ( sj + dj ) + ( si + di ) * number_of_colors + c ];
                            float ker = kernel[ ( 2 + dj ) * 5 + ( 2 + di ) ];
                            addition += (float) src * ker ;
                        }
                    }
                    pyramid_in[ ilevel ][ targetStep * tj + ti * number_of_colors + c ] = (unsigned char) addition;
                }
            }
        }

        sourceRows = targetRows;
        sourceCols = targetCols;
    }

#if DEBUG_PYRAMID
    printf( "Pyramid::make_pyramid Done making %d type pyramid\n" , imageType );
    fflush( stdout );
#endif

    return pyramid_in;
}

//public methods
void Pyramid_class::set_und_image( const cv::Mat &und_image_in )
{
#if DEBUG_PYRAMID
    printf( "Pyramid::set_und_image\n" );
    fflush( stdout );
#endif

    und_image = und_image_in;
    assert ( und_image.isContinuous() );

    clear_und_images();

    und_images  = std::vector< unsigned char* > ( stop + 1 );

    und_images[ 0 ] = und_image_in.data;
    und_rows        = und_image.rows;
    und_cols        = und_image.cols;
    und_step        = und_image.step1();
    und_images      = make_pyramid( und_images , imageType_und );

#if DEBUG_PYRAMID_IMAGES
    cv::imshow( "und image" , und_image );
#endif
}

void Pyramid_class::set_def_image( const cv::Mat &def_image_in )
{
#if DEBUG_PYRAMID
    printf( "Pyramid::set_def_image\n" );
    fflush( stdout );
#endif

    def_image = def_image_in;
    assert ( def_image.isContinuous() );

    clear_def_images();

    def_images  = std::vector< unsigned char* > ( stop + 1 );

    def_images[ 0 ] = def_image_in.data;
    def_rows        = def_image.rows;
    def_cols        = def_image.cols;
    def_step        = def_image.step1();
    def_images      = make_pyramid( def_images , imageType_def );

    if ( def_cols * def_rows * ( number_of_colors * number_of_interpolation_parameters + 1 ) > allocated_all_interpolation_parameters )
    {
         allocated_all_interpolation_parameters = def_cols * def_rows * ( number_of_colors * number_of_interpolation_parameters + 1 );
         set_all_interpolation_parameters();
    }
    reset_all_interpolation_parameters();
}

void Pyramid_class::set_nxt_image( const cv::Mat &nxt_image_in )
{
#if DEBUG_PYRAMID
    printf( "Pyramid::set_nxt_image\n" );
    fflush( stdout );
#endif

    nxt_image = nxt_image_in;
    assert ( nxt_image.isContinuous() );

    clear_nxt_images();

    nxt_images  = std::vector< unsigned char* > ( stop + 1 );

    nxt_images[ 0 ] = nxt_image.data;
    nxt_rows        = nxt_image.rows;
    nxt_cols        = nxt_image.cols;
    nxt_step        = nxt_image.step1();
    nxt_images      = make_pyramid( nxt_images , imageType_nxt );
}

void Pyramid_class::und_from_def( )
{
#if DEBUG_PYRAMID
    printf( "Pyramid::und_from_def\n" );
    fflush( stdout );
#endif

    und_image   = def_image;
    und_images  = def_images;
    def_images.clear();
    und_rows    = def_rows;
    und_cols    = def_cols;
    und_step    = def_step;
    def_rows    = 0;
    def_cols    = 0;
    def_step    = 0;
}

void Pyramid_class::def_from_nxt( )
{
#if DEBUG_PYRAMID
    printf( "Pyramid::def_from_nxt\n" );
    fflush( stdout );
#endif

    def_image   = nxt_image;
    def_images  = nxt_images;
    nxt_images.clear();
    def_rows    = nxt_rows;
    def_cols    = nxt_cols;
    def_step    = nxt_step;
    nxt_rows    = 0;
    nxt_cols    = 0;
    nxt_step    = 0;

    if ( def_cols * def_rows * ( number_of_colors * number_of_interpolation_parameters + 1 ) > allocated_all_interpolation_parameters )
    {
         allocated_all_interpolation_parameters = def_cols * def_rows * ( number_of_colors * number_of_interpolation_parameters + 1 );
         set_all_interpolation_parameters();
    }
    reset_all_interpolation_parameters();

#if DEBUG_PYRAMID_IMAGES
    cv::imshow( "new def image" , def_image );
    fflush( stdout );
#endif
}

void Pyramid_class::translate_model_parameters( float *model_parameters ,
                                                  int pyramid_level_src,
                                                  int pyramid_level_dst )
{
    float magnification;
    if ( pyramid_level_dst - pyramid_level_src > 0 )
    {
        magnification = 1.f / (float) ( 1 << (   pyramid_level_dst - pyramid_level_src ) );
    }
    else
    {
        magnification =       (float) ( 1 << ( - pyramid_level_dst + pyramid_level_src ) );
    }

    switch ( fittingModel )
    {
    case fm_U:
    case fm_UV:
    case fm_UVQ:
    case fm_UVUxUyVxVy:

        for ( int ipar = 0 ; ipar < std::min( number_of_model_parameters , 2 ) ; ++ipar )
        {
            model_parameters[ ipar ] *= magnification;
        }

        break;

    default:
        bool unknownFittingModel = false;
        assert ( unknownFittingModel );
        break;
    }
}

void Pyramid_class::set_xy_positions( float *xy_positions_in , int number_of_points_in )
{
    clear_xy_positions();

    xy_positions            = std::vector< float* >( stop + 1 );
    number_of_points        = std::vector< int >( stop + 1 );
    xy_positions[ 0 ]       = xy_positions_in;
    number_of_points[ 0 ]   = number_of_points_in;

    int prevLevel = 0;
    int firstLevel = ( start == 0 ? step : start ) ;

    for ( int ilevel = firstLevel; ilevel <= stop ; ilevel += step )
    {
      int       magnification   = 1 << ( ilevel - prevLevel ) ;
      float     magnificationInv = 1.f / (float) magnification;

      std::vector< float > v;
      for ( int ipoint = 0 ; ipoint < number_of_points[ prevLevel ] ; ++ipoint )
      {
          int index = ipoint * 2;
          int ix    = (int) ( xy_positions[ prevLevel ][ index     ] + 0.5f );
          int iy    = (int) ( xy_positions[ prevLevel ][ index + 1 ] + 0.5f );

          if ( ix % magnification == 0 && iy % magnification == 0 )
          {
              v.push_back( xy_positions[ prevLevel ][ index     ] * magnificationInv );
              v.push_back( xy_positions[ prevLevel ][ index + 1 ] * magnificationInv );
          }
      }

      number_of_points[ ilevel ]    = v.size() / 2;
      xy_positions[ ilevel ]        = new float[ v.size() ];
      memcpy( xy_positions[ ilevel ] , v.data() , v.size() * sizeof( float ) );

      prevLevel = ilevel;
    }
}

void Pyramid_class::set_und_center()
{
    float und_x_center      = 0.f;
    float und_y_center      = 0.f;

    int n                   = number_of_points[ 0 ];
    float *und_xy_positions = xy_positions[ 0 ];

    for ( int ipoints = 0 ; ipoints < n ; ++ipoints )
    {
        und_x_center += und_xy_positions[ ipoints * 2     ];
        und_y_center += und_xy_positions[ ipoints * 2 + 1 ];
    }

    und_x_center /= (float) n;
    und_y_center /= (float) n;

    set_und_center( und_x_center , und_y_center );
#if DEBUG_CENTER
    printf ("pyramid: set_und_center: und_x_center = %10.4f und_y_center = %10.4f\n",
            und_x_center,
            und_y_center );
    fflush( stdout );
#endif
}

void Pyramid_class::set_und_center( float und_x_center_in ,
                                    float und_y_center_in )
{
    xy_center.clear();
    xy_center = v_points ( stop + 1 );
    xy_center[ 0 ] = std::make_pair( und_x_center_in , und_y_center_in );

    int thisLevel = ( start == 0 ? step : start ) ;

    for ( int ilevel = thisLevel ; ilevel <= stop ; ilevel += step )
    {
        float magnificationInv = 1.f / (float ) ( 1 << ilevel );
        xy_center[ ilevel ] = std::make_pair( und_x_center_in * magnificationInv ,
                                              und_y_center_in * magnificationInv );
    }
}

void Pyramid_class::set_all_interpolation_parameters( )
{
#if DEBUG_TIME_INTERPOLATION_PARAMETERS_ALLOCATION
    auto start_all_interpolation_parameter_allocation = std::clock();
#endif

    clear_all_interpolation_parameters();

    all_interpolation_parameters = std::vector< float* >( stop + 1 );

    for ( int ilevel = start ; ilevel <= stop ; ilevel += step )
    {
        int allocate_this_level = allocated_all_interpolation_parameters /  ( ( 1 << ilevel ) * ( 1 << ilevel ) ) ;
        all_interpolation_parameters[ ilevel ] = new float[ allocate_this_level ];
    }

#if DEBUG_TIME_INTERPOLATION_PARAMETERS_ALLOCATION
    auto duration_all_interpolation_parameter_allocation = ( std::clock() - start_all_interpolation_parameter_allocation ) / (float) CLOCKS_PER_SEC;
    std::cout<<"pyramid() :Interpolation Parameters Allocation time(s): "<< duration_all_interpolation_parameter_allocation <<'\n';
#endif
}

void Pyramid_class::reset_all_interpolation_parameters( )
{
#if DEBUG_TIME_INTERPOLATION_PARAMETERS_RESET
    auto start_all_interpolation_parameter_reset = std::clock();
#endif

    for ( int ilevel = start ; ilevel <= stop ; ilevel += step )
    {
        int def_image_cols_level = def_cols / ( 1 << ilevel );
        int def_image_rows_level = def_rows / ( 1 << ilevel );

        //set the "all parameter's" flag to 0 = not written
        for ( int ix = 0 ; ix < def_image_cols_level ; ix++ )
        {
           for ( int iy = 0 ; iy < def_image_rows_level ; iy++ )
           {
                all_interpolation_parameters[ ilevel ][ ( ix + iy * def_image_cols_level ) * ( number_of_colors * number_of_interpolation_parameters + 1 ) ] = 0.f;
           }
        }
    }
#if DEBUG_TIME_INTERPOLATION_PARAMETERS_RESET
    auto duration_all_interpolation_parameter_reset = ( std::clock() - start_all_interpolation_parameter_reset ) / (float) CLOCKS_PER_SEC;
    std::cout<<"pyramid() :Interpolation Parameters Reset time(s): "<< duration_all_interpolation_parameter_reset <<'\n';
#endif
}

void Pyramid_class::set_number_of_model_parameters( int number_of_model_parameters_in )
{
    number_of_model_parameters = number_of_model_parameters_in;
}

unsigned char* Pyramid_class::get_und_ptr( int level )
{
    return und_images[ level ];
}

unsigned char* Pyramid_class::get_def_ptr( int level )
{
    return def_images[ level ];
}

float* Pyramid_class::get_all_param( int level )
{
    return all_interpolation_parameters[ level ];
}

float* Pyramid_class::get_xy_positions( int level )
{
    return xy_positions[ level ];
}

int Pyramid_class::get_number_of_points( int level )
{
    return number_of_points[ level ];
}

void Pyramid_class::get_und_center( float &und_x_center_in, float &und_y_center_in, int level )
{
    und_x_center_in =  xy_center[ level ].first;
    und_y_center_in =  xy_center[ level ].second;
}

int Pyramid_class::get_rows( int level , ImageType type )
{
    int localRows;
    switch ( type )
    {
    case imageType_und:
        localRows = und_rows / ( 1 << level );
        break;
    case imageType_def:
        localRows = def_rows / ( 1 << level );
        break;
    default:
        assert( false );
    }

    return localRows;
}

int Pyramid_class::get_cols( int level , ImageType type )
{
    int localCols;
    switch ( type )
    {
    case imageType_und:
        localCols = und_cols / ( 1 << level );
        break;
    case imageType_def:
        localCols = def_cols / ( 1 << level );
        break;
    default:
        assert( false );
    }

    return localCols;
}

int Pyramid_class::get_step( int level , ImageType type )
{
    int localStep;
    switch ( type )
    {
    case imageType_und:
        localStep = und_step / ( 1 << level );
        break;
    case imageType_def:
        localStep = def_step / ( 1 << level );
        break;
    default:
        assert( false );
    }

    return localStep;
}








