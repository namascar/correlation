#ifndef CUDA_POLYGON_CUH
#define CUDA_POLYGON_CUH

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/system/cuda/execution_policy.h>

#include "domains.hpp"
#include "defines.hpp"
#include "enums.hpp"
#include "model_class.hpp"

#include "kernels.cuh"
#include "correlationKernel.cuh"

#include <stdio.h>
#include <vector>
#include <algorithm>

#include <nvToolsExt.h>                     // Marker for the nvvp profiler

typedef thrust::host_vector  < thrust::tuple< float , float , float > > LineEquationsHost;
typedef thrust::device_vector< thrust::tuple< float , float , float > > LineEquationsDevice;

typedef std::vector < thrust::device_vector< float > >                  pyramid_vector;

typedef thrust::device_vector< float >::iterator                        VIt;
typedef thrust::tuple< VIt, VIt >                                       TupleIt;
typedef thrust::zip_iterator< TupleIt >                                 ZipIt;

struct BoundingRectangle
{
    int x0 , y0 , x1 , y1;

    BoundingRectangle( int x0_ , int y0_ , int x1_ , int y1_ )
        : x0( x0_ ),
          y0( y0_ ),
          x1( x1_ ),
          y1( y1_ ){}

    BoundingRectangle( float r , float dr , float a , float da,
                       float cx , float cy , int as )
    {
        switch( as )
        {
            case 0 :

                assert( false );

                break;

            case 1:

                x0 = cx - ( r + dr );
                x1 = cx + ( r + dr );
                y0 = cy - ( r + dr );
                y1 = cy + ( r + dr );

                break;

            default:
            {
                float sin0 =  (float) sin( a            );
                float cos0 =  (float) cos( a            );
                float sin1 =  (float) sin( a + da       );
                float cos1 =  (float) cos( a + da       );
                float sin2 =  (float) sin( a + da / 2.f );
                float cos2 =  (float) cos( a + da / 2.f );

                float corner_00_x = cx + ( r      ) * cos0;
                float corner_01_x = cx + ( r      ) * cos1;
                float corner_10_x = cx + ( r + dr ) * cos0 * 1.2f; //Cheap sag
                float corner_11_x = cx + ( r + dr ) * cos1 * 1.2f;

                float corner_00_y = cy + ( r      ) * sin0;
                float corner_01_y = cy + ( r      ) * sin1;
                float corner_10_y = cy + ( r + dr ) * sin0 * 1.2f;
                float corner_11_y = cy + ( r + dr ) * sin1 * 1.2f;

                float arc_x       = cx + ( r + dr ) * cos2;
                float arc_y       = cy + ( r + dr ) * sin2;

                x0 = std::min( arc_x, std::min( std::min( corner_00_x, corner_01_x ), std::min( corner_10_x, corner_11_x ) ) );
                x1 = std::max( arc_x, std::max( std::max( corner_00_x, corner_01_x ), std::max( corner_10_x, corner_11_x ) ) );

                y0 = std::min( arc_y, std::min( std::min( corner_00_y, corner_01_y ), std::min( corner_10_y, corner_11_y ) ) );
                y1 = std::max( arc_y, std::max( std::max( corner_00_y, corner_01_y ), std::max( corner_10_y, corner_11_y ) ) );

                break;
            }
        }
    }

    BoundingRectangle( v_points blobContour )
    {
        float x0float = blobContour[ 0 ].first;
        float x1float = blobContour[ 0 ].first;
        float y0float = blobContour[ 0 ].second;
        float y1float = blobContour[ 0 ].second;

        for ( unsigned int i = 1 ; i < blobContour.size() ; ++i )
        {
			if ( blobContour[ i ].first < x0float ) x0float = blobContour[ i ].first; 
			if ( blobContour[ i ].first > x1float ) x1float = blobContour[ i ].first;
			if ( blobContour[ i ].second < y0float ) y0float = blobContour[ i ].second;
			if ( blobContour[ i ].second > y1float ) y1float = blobContour[ i ].second;
        }

        x0 = (int) x0float;
        y0 = (int) y0float;
        x1 = (int) x1float;
        y1 = (int) y1float;
    }
};

struct RectFunctor
{
    int  x0 , x1 , y0 , y1 , x_pitch;

    RectFunctor( int x0_ , int y0_ , int x1_ , int y1_ )
        : x0 ( x0_ ) , x1 ( x1_ ) , y0 ( y0_ ) , y1 ( y1_ ) , x_pitch( x1 - x0 + 1 )
            {}

    template <typename Tuple>
    __host__ __device__
    void operator() ( Tuple xy )
    {
        // Only x is indexed, compute y first so as to don't overwrite x
        thrust::get< 1 >( xy ) = (float) ( (int) thrust::get< 0 >( xy ) / x_pitch + y0 );
        thrust::get< 0 >( xy ) = (float) ( (int) thrust::get< 0 >( xy ) % x_pitch + x0 );
    }
};

struct copyFunctor
{

    int       magnification;

    copyFunctor( int prevLevel , int thisLevel )
        : magnification( 1 << ( thisLevel - prevLevel ) )
            {}

    template <typename Tuple>
    __host__ __device__
    bool operator() ( Tuple xy )
    {
        // Find if the xy point needs to be copied
        int ix    = (int) ( thrust::get< 0 >( xy ) + 0.5f );
        int iy    = (int) ( thrust::get< 1 >( xy ) + 0.5f );

       return ( ix % magnification == 0 && iy % magnification == 0 );
    }
};

struct scale1DFunctor
{
    int       magnification;
    float     magnificationInv;

    scale1DFunctor( int prevLevel , int thisLevel )
        : magnification( 1 << ( thisLevel - prevLevel ) ),
          magnificationInv( 1.f / (float) magnification )
            {}

    template <typename T>
    __host__ __device__
    T operator() ( T x )
    {
        // Scale the xy point
        return x * magnificationInv;
    }
};

struct scale2DFunctor
{
    int       magnification;
    float     magnificationInv;

    scale2DFunctor( int prevLevel , int thisLevel )
        : magnification( 1 << ( thisLevel - prevLevel ) ),
          magnificationInv( 1.f / (float) magnification )
            {}

    template <typename Tuple>
    __host__ __device__
    void operator() ( Tuple xy )
    {
        // Scale the xy point
        thrust::get< 0 >( xy ) = thrust::get< 0 >( xy ) * magnificationInv;
        thrust::get< 1 >( xy ) = thrust::get< 1 >( xy ) * magnificationInv;
    }
};

struct removeAnnularFunctor
{
    float ri2 , ro2 , a , da, cx , cy , as;

    removeAnnularFunctor( float r_ , float dr_ , float a_ , float da_ ,
                          float cx_ , float cy_ , int as_ )
        : ri2( r_ * r_ ) , ro2( ( r_ + dr_ ) * ( r_ + dr_ ) ) , a( a_ ) , da( da_ ), cx( cx_ ) , cy( cy_ ) , as( as_ )
            {}

    template <typename Tuple>
    __host__ __device__
    bool operator() ( Tuple xy )
    {
        // Find if the xy point needs to be scaled-cop
        float dx    = thrust::get< 0 >( xy ) - cx;
        float dy    = thrust::get< 1 >( xy ) - cy;

        float R2 = dx * dx + dy * dy;
        float angle = atan2( dy , dx );
        angle = ( angle < 0.f ) ? angle + 2.f * 3.14159265359f : angle;

        if ( R2 < ri2 || R2 > ro2 )         return true;
        if ( as == 1 )                      return false;
        if ( angle < a || angle > a + da )  return true;

        return false;
    }
};

struct removeBlobFunctor
{
    thrust::device_ptr< thrust::pair    < float , float > >         contourPtr;
    thrust::device_ptr< thrust::tuple   < float , float , float > > lineEquationsPtr;
    int size;

    removeBlobFunctor
       (
            thrust::device_ptr< thrust::pair  < float , float > >           contourPtr_ ,
            thrust::device_ptr< thrust::tuple < float , float , float > >   lineEquationsPtr_,
            int size_
        )
        : contourPtr( contourPtr_ ),
          lineEquationsPtr( lineEquationsPtr_ ),
          size( size_ )
            {}

    __host__ __device__
    intersectionEnum crosses( thrust::tuple< float , float > point ,
                              thrust::pair<  float , float > edge1 ,
                              thrust::pair<  float , float > edge2 ,
                              thrust::tuple< float , float , float > lineEquation )
    {
        float v1x2    = thrust::get< 0 >( point );  // define a horizontal segment from outside the image to the point
        float v1y2    = thrust::get< 1 >( point );

        float v2y1    = edge1.second;
        float v2y2    = edge2.second;

        // From
        // https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
        if ( v2y1 > v1y2 && v2y2 > v1y2 ) return intersection_no;
        if ( v2y1 < v1y2 && v2y2 < v1y2 ) return intersection_no;

        // The fact that vector 2 intersected the infinite line 1 above doesn't
        // mean it also intersects the vector 1. Vector 1 is only a subset of that
        // infinite line 1, so it may have intersected that line before the vector
        // started or after it ended. To know for sure, we have to repeat the
        // the same test the other way round. We start by calculating the
        // infinite line 2 in linear equation standard form.
        float a = thrust::get< 0 >( lineEquation );
        float b = thrust::get< 1 >( lineEquation );
        float c = thrust::get< 2 >( lineEquation );

        // Calculate d1 and d2 again, this time using points of vector 1.
        //d1 = ( a2 * v1x1 ) + ( b2 * v1y1 ) + c2;
        //d2 = ( a2 * v1x2 ) + ( b2 * v1y2 ) + c2;
        float temp = b * v1y2 + c;
        float d1 = - a          + temp; //v1x1 = -1 , v1y1 = v1y2
        float d2 = ( a * v1x2 ) + temp;

        // Again, if both have the same sign (and neither one is 0),
        // no intersection is possible.
        if ( d1 > 0 && d2 > 0 ) return intersection_no;
        if ( d1 < 0 && d2 < 0 ) return intersection_no;

        // If we get here, only two possibilities are left. Either the two
        // vectors intersect in exactly one point or they are collinear, which
        // means they intersect in any number of points from zero to infinite.
        //if ( ( a1 * b2 ) - ( a2 * b1 ) == 0.0f ) return intersection_colliear;
        if ( d1 == 0.f && d2 == 0.f ) return intersection_colliear;

        // If they are not collinear, they must intersect in exactly one point.
        return intersection_yes;
    }

    template <typename Tuple>
    __host__ __device__
    bool operator() ( Tuple xy )
    {
        // Find if the xy is inside the polygon

        int count = 1;

        for ( int iEdge = 0 ; iEdge < size - 1 ; ++iEdge )
        {
            if ( crosses( xy , contourPtr[ iEdge ] , contourPtr[ iEdge + 1 ],
                     lineEquationsPtr[ iEdge ] ) == intersection_yes )
            {
                count++;
            }
        }

        if ( crosses( xy , contourPtr[ size - 1 ] , contourPtr[ 0 ],
                 lineEquationsPtr[ size - 1 ] ) == intersection_yes )
        {
            count++;
        }

        return count % 2;
    }
};

struct translateFunctor
{
    float dx;
    float dy;

    translateFunctor( float dx_ , float dy_ )
        : dx ( dx_ ) , dy ( dy_ ){}

    template <typename Tuple>
    __host__ __device__
    void operator() ( Tuple xy )
    {
        // Translate the xy point
        thrust::get< 0 >( xy ) = thrust::get< 0 >( xy ) + dx;
        thrust::get< 1 >( xy ) = thrust::get< 1 >( xy ) + dy;
    }
};

class cudaPolygon
{

protected:

    int x0;
    int y0;
    int x1;
    int y1;

    int start, step, stop;

	int size;
    int sector;

    fittingModelEnum fittingModel;
    int              numberOfModelParameters;

    cudaStream_t domainSelectionStream;

    float          *globalABChi   = nullptr;

    pyramid_vector  undeformed_xs;
    pyramid_vector  undeformed_ys;
    pyramid_vector  xyr_center;

    thrust::device_vector< float > deformed_xs0;
    thrust::device_vector< float > deformed_ys0;
    bool deformed0IsReady = false;

    // Derivative of the dilation term ( 1 + e ( L / R - 1 ) ) wrt e
    pyramid_vector  DdilDe;

    std::vector< float* > parameters;
    int                   currentPyramidLevel = 0;

    // Pair of transfer memory spaces - host memory is pinned
    CorrelationResult *gpuCorrelationResults;
    CorrelationResult *cpuCorrelationResults;

    void            fillRectangle                   ();
    void            makeUndCenter0                  ();
    void            allocateGlobalABChi             ();
    void            deallocateGlobalABChi           ();
    void            makeAllUndCenters               ();
    void            makeAllUndLevels                ();

public:

    cudaPolygon( int iSector , BoundingRectangle boundingRectangle ,
                 int start_ , int step_ , int stop_ ,
                 fittingModelEnum fittingModel_ )
        :x0 ( boundingRectangle.x0 ),
         y0 ( boundingRectangle.y0 ),
         x1 ( boundingRectangle.x1 ),
         y1 ( boundingRectangle.y1 ),
         start( start_ ) , step( step_ ) , stop ( stop_ ) ,
         size ( ( x1 - x0 + 1 ) * ( y1 - y0 + 1 ) ),
         sector ( iSector ),
         fittingModel ( fittingModel_ )
    {
        numberOfModelParameters = ModelClass::get_number_of_model_parameters( fittingModel );

        parameters.resize( parType_NUMBER_OF_ITEMS );
        for ( unsigned int iPar = 0 ; iPar < parameters.size() ; ++iPar )
        {
            cudaMalloc( (void**) &parameters[ iPar ] , numberOfModelParameters * sizeof( float ) );
        }

        cudaStreamCreate( &domainSelectionStream );

        fillRectangle();

        // Pair of transfer memory spaces - host memory is pinned
        cudaMalloc    ( (void**) &gpuCorrelationResults , sizeof( CorrelationResult ) );
        cudaMallocHost( (void**) &cpuCorrelationResults , sizeof( CorrelationResult ) );
    }

    v_points             getUndXY0ToCPU             ();
    v_points             getDefXY0ToCPU             ();
    int                  getNumberOfPoints          ( int level );
    float               *getUndXPtr                 ( int level );
    float               *getUndYPtr                 ( int level );
    float               *getUndCenter               ( int level );
    float               *getDdilDePtr               ( int level );
    float               *getGlobalABChi             ();
    float               *getParameters              ( parameterTypeEnum parSrc );
    CorrelationResult   *getCorrelationResultsToCPU ();

    void                 updateParameters(            int numberOfModelParameters,
                                                      parameterTypeEnum parSrc,
                                                      parameterTypeEnum parDst,
                                                      cudaStream_t stream );

    void                 scaleParametersForLevel    ( int level );

    void                 initializeParametersLevel0 ( float *initialGuess_ );
    void                 transferParameters         ( parameterTypeEnum parSrc , parameterTypeEnum parDst );
    virtual void         updatePolygon              ( deformationDescriptionEnum deformationDescription );

    ~cudaPolygon()
    {
    #if DEBUG_CUDA_POLYGON
        printf( "Deleting cudaPolygon Base\n" );
    #endif
        deallocateGlobalABChi();

        cudaStreamDestroy( domainSelectionStream );

        cudaFree    ( gpuCorrelationResults );
        cudaFreeHost( cpuCorrelationResults );

        for ( unsigned int iPar = 0 ; iPar < parameters.size() ; ++iPar )
        {
            cudaFree( parameters[ iPar ] );
        }
    }
};

class cudaPolygonRectangular : public cudaPolygon
{

private:

public:
    cudaPolygonRectangular( int iSector , int x0 , int y0 , int x1 , int y1 ,
                            int start , int step , int stop ,
                            fittingModelEnum fittingModel_ )
        : cudaPolygon( iSector , BoundingRectangle( x0 , y0 , x1 , y1 ) , start , step , stop , fittingModel_ )
    {
        makeUndCenter0();
        makeAllUndLevels();
        makeAllUndCenters();
    }

    ~cudaPolygonRectangular()
    {
#if DEBUG_CUDA_POLYGON
    printf( "Deleting cudaPolygonRectangular\n" );
#endif
    }
};

class cudaPolygonAnnular : public cudaPolygon
{

private:
    void            makeAllDdilDe                   ();
    void            cleanAnnularRectangle0          ( float r , float dr , float a , float da,
                                                      float cx , float cy , int as );

public:
    cudaPolygonAnnular( int iSector , float r , float dr , float a , float da,
                        float cx , float cy , int as,
                        int start , int step , int stop,
                        fittingModelEnum fittingModel_ )
        : cudaPolygon( iSector , BoundingRectangle( r , dr , a , da , cx , cy , as ) , start , step , stop , fittingModel_ )
    {
        // Put a marker on the nvvp CUDA profiler
        nvtxRangePushA ( "cudaPolygonAnnular::cudaPolygonAnnular" );

        cleanAnnularRectangle0( r , dr , a , da , cx , cy , as );
        makeUndCenter0();
        makeAllUndLevels();
        makeAllUndCenters();

        #if DEBUG_CUDA_POLYGON
        printf( "cudaPolygonAnnular::cudaPolygonAnnular\n" );
        #endif

        nvtxRangePop();
    }

    void                 updatePolygon               ( deformationDescriptionEnum deformationDescription ) override;

    ~cudaPolygonAnnular()
    {
#if DEBUG_CUDA_POLYGON
    printf( "Deleting cudaPolygonAnnular\n" );
#endif
    }
};

class cudaPolygonBlob : public cudaPolygon
{
    typedef thrust::device_vector< thrust::pair < float , float > > v_pairs;

private:
    void                cleanBlobRectangle0          ( v_points blob );
    LineEquationsDevice makeLineEquations            ( v_points blobContour );
public:
    cudaPolygonBlob( v_points blobContour,
                     int start , int step , int stop,
                     fittingModelEnum fittingModel_ )
        : cudaPolygon( 0 , BoundingRectangle( blobContour ) , start , step , stop , fittingModel_ )
    {
        cleanBlobRectangle0( blobContour );
        makeUndCenter0();
        makeAllUndLevels();
        makeAllUndCenters();
    }

    ~cudaPolygonBlob()
    {
#if DEBUG_CUDA_POLYGON
    printf( "Deleting cudaPolygonBlob\n" );
#endif
    }
};

#endif // CUDA_POLYGON_CUH
