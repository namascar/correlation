#include "correlationKernel.cuh"

/**
  CUDA Kernel to perform parameter update one kernel launch ( nPar threads, 1 block ).
  */
__global__ void
kUpdateParameters
(
        const float                  *parametersSrc,
              float                  *parametersDst,
        const float                  *parametersIncrement,
        const int                     numberOfModelParameters
)
{
    int i = threadIdx.x;
    if ( i < numberOfModelParameters )
    {
        parametersDst[ i ] = parametersSrc[ i ] + parametersIncrement[ i ];
    }
}

/**
  CUDA Kernel to perform pyramid scaling in one kernel launch ( 1 thread, 1 block ).
  */
__global__ void
kScale
(
              float                  *d_parametersLastGood,
              float                  *d_parametersTentative,
              float                  *d_parametersSaved,
        const fittingModelEnum        fittingModel,
        const float                   multiplier
)
{
    switch( fittingModel )
    {
        case fm_U:

            d_parametersLastGood [ 0 ] = d_parametersLastGood[ 0 ] * multiplier;
            d_parametersTentative[ 0 ] = d_parametersLastGood[ 0 ];
            d_parametersSaved    [ 0 ] = d_parametersLastGood[ 0 ];

            break;

        case fm_UV:
        case fm_UVQ:
        case fm_UVUxUyVxVy:

            d_parametersLastGood [ 0 ] = d_parametersLastGood[ 0 ] * multiplier;
            d_parametersTentative[ 0 ] = d_parametersLastGood[ 0 ];
            d_parametersSaved    [ 0 ] = d_parametersLastGood[ 0 ];

            d_parametersLastGood [ 1 ] = d_parametersLastGood[ 1 ] * multiplier;
            d_parametersTentative[ 1 ] = d_parametersLastGood[ 1 ];
            d_parametersSaved    [ 1 ] = d_parametersLastGood[ 1 ];

            break;
    }
}

/**
  CUDA Kernel to perform inplace fitting model in one kernel launch. Takes the undX,Y positions
  and computes their corresponding defX,Y based on the fitting model ( U, UV, UVQ, UVUxUyVxVy ).
  */
__global__ void
kModel_inPlace
(
        const float                  *d_parameters,

        const fittingModelEnum        fittingModel,

        const int                     numberOfPoints,
              float                  *undX_ptr,
              float                  *undY_ptr,
        const float                  *undCenter
)
{
    int iPoint = blockDim.x * blockIdx.x + threadIdx.x;

    if ( iPoint < numberOfPoints )
    {
        float undX = undX_ptr[ iPoint ];
        float undY = undY_ptr[ iPoint ];

        switch( fittingModel )
        {
            case fm_U:

                undX_ptr[ iPoint ] = undX + d_parameters[ 0 ];
                undY_ptr[ iPoint ] = undY;

                break;

            case fm_UV:

                undX_ptr[ iPoint ] = undX + d_parameters[ 0 ];
                undY_ptr[ iPoint ] = undY + d_parameters[ 1 ];

                break;

            case fm_UVQ:
                {
                    float xMinusCenter = undX - undCenter[ 0 ];
                    float yMinusCenter = undY - undCenter[ 1 ];

                    undX_ptr[ iPoint ] = undX + d_parameters[ 0 ] - yMinusCenter * d_parameters[ 2 ];
                    undY_ptr[ iPoint ] = undY + d_parameters[ 1 ] + xMinusCenter * d_parameters[ 2 ];
                }
                break;

            case fm_UVUxUyVxVy:
                {
                    float xMinusCenter = undX - undCenter[ 0 ];
                    float yMinusCenter = undY - undCenter[ 1 ];

                    undX_ptr[ iPoint ] = undX + d_parameters[ 0 ] +
                                 xMinusCenter * d_parameters[ 2 ] +
                                 yMinusCenter * d_parameters[ 3 ];

                    undY_ptr[ iPoint ] = undY + d_parameters[ 1 ] +
                                 xMinusCenter * d_parameters[ 4 ] +
                                 yMinusCenter * d_parameters[ 5 ];
                }

                break;

            default:
                break;
        }
    }
}

/**
  CUDA Kernel to perform fitting model and interpolation in one kernel launch. Takes the undX,Y positions
  and computes their corresponding defX,y and derivatives of the image with respect to the parameters
  based on the fitting model ( U, UV, UVQ, UVUxUyVxVy ). Then Queries the undImage and defImage to assempble
  the matix and vector contribution of this point. Finnaly reduces all info to a single matrix, vector and chi.
  */
__global__ void
kCorrelation
(
        const float                  *d_parameters,

        const fittingModelEnum        fittingModel,
        const interpolationModelEnum  interpolationModel,

        const int                     numberOfColors,
        const cudaTextureObject_t     undTexture,
        const cudaTextureObject_t     defTexture,

        const int                     numberOfPoints,
        const float                  *undX_ptr,
        const float                  *undY_ptr,
        const float                  *undCenter,

              float                  *global_mat_A_and_vec_B_and_CHI
)
{
    // Allocate shared memory mat_A_and_vec_B_and_CHI
    extern __shared__ float shared_mat_A_and_vec_B_and_CHI[];

    int iPoint = blockDim.x * blockIdx.x + threadIdx.x;

    if ( iPoint < numberOfPoints )
    {
        float undX = undX_ptr[ iPoint ];
        float undY = undY_ptr[ iPoint ];

        float xMinusCenter = undX - undCenter[ 0 ];
        float yMinusCenter = undY - undCenter[ 1 ];

        float defX, defY;
        int size_A , size_A_B , size_A_B_chi;

        switch( fittingModel )
        {
            case fm_U:

                defX = undX + d_parameters[ 0 ];
                defY = undY;

                size_A                       = 1;
                size_A_B                     = 2;
                size_A_B_chi                 = 3;

                break;

            case fm_UV:

                defX = undX + d_parameters[ 0 ];
                defY = undY + d_parameters[ 1 ];

                size_A                       = 4;
                size_A_B                     = 6;
                size_A_B_chi                 = 7;

                break;

            case fm_UVQ:

                defX = undX + d_parameters[ 0 ] - yMinusCenter * d_parameters[ 2 ];
                defY = undY + d_parameters[ 1 ] + xMinusCenter * d_parameters[ 2 ];

                size_A                       = 9;
                size_A_B                     = 12;
                size_A_B_chi                 = 13;

                break;
            case fm_UVUxUyVxVy:

                defX = undX + d_parameters[ 0 ] +
                        xMinusCenter * d_parameters[ 2 ] +
                        yMinusCenter * d_parameters[ 3 ];

                defY = undY + d_parameters[ 1 ] +
                        xMinusCenter * d_parameters[ 4 ] +
                        yMinusCenter * d_parameters[ 5 ];

                size_A                       = 36;
                size_A_B                     = 42;
                size_A_B_chi                 = 43;

                break;

            default:
                break;
        }

        switch( interpolationModel )
        {
            case im_nearest:
                dNearest(
                              undX,
                              undY,
                              defX,
                              defY,

                              xMinusCenter,
                              yMinusCenter,

                              fittingModel,

                             &shared_mat_A_and_vec_B_and_CHI[ threadIdx.x * size_A_B_chi ],
                              size_A,
                              size_A_B,
                              size_A_B_chi,

                              numberOfColors,
                              undTexture,
                              defTexture
                         );
                break;
            case im_bilinear:
                dBilinear(
                              undX,
                              undY,
                              defX,
                              defY,

                              xMinusCenter,
                              yMinusCenter,

                              fittingModel,

                             &shared_mat_A_and_vec_B_and_CHI[ threadIdx.x * size_A_B_chi ],
                              size_A,
                              size_A_B,
                              size_A_B_chi,

                              numberOfColors,
                              undTexture,
                              defTexture
                         );
                break;
            case im_bicubic:
                dBicubic(
                              undX,
                              undY,
                              defX,
                              defY,

                              xMinusCenter,
                              yMinusCenter,

                              fittingModel,

                             &shared_mat_A_and_vec_B_and_CHI[ threadIdx.x * size_A_B_chi ],
                              size_A,
                              size_A_B,
                              size_A_B_chi,

                              numberOfColors,
                              undTexture,
                              defTexture
                         );
                break;
            default:
                break;
        }

        __syncthreads();

        // Do block reduction in shared memory since we are using 256 = 2^8 threads
        for ( unsigned int s = 1 ; s < blockDim.x ; s *= 2 )
        {
            if ( ( threadIdx.x % ( 2 * s ) ) == 0 && ( threadIdx.x + s ) + blockDim.x * blockIdx.x < numberOfPoints ) //don't include contributions into the last thread of last block
            {
                for ( int index = 0 ; index < size_A_B_chi ; ++index )
                {
                    shared_mat_A_and_vec_B_and_CHI[ threadIdx.x * size_A_B_chi + index ] +=
                            shared_mat_A_and_vec_B_and_CHI[ ( threadIdx.x + s ) * size_A_B_chi + index ];
                }
            }

            __syncthreads();
        }

        // Thread id = 0 includes the contribution of its block into the global space for global reduction
        if ( threadIdx.x == 0 )
        {
            memcpy(
                        &global_mat_A_and_vec_B_and_CHI[ blockIdx.x * size_A_B_chi ],
                         shared_mat_A_and_vec_B_and_CHI,
                         size_A_B_chi * sizeof( float )
                  );
        }
    }
}

/**
 CUDA device function to perform the nearest interpolation and get the contribution to the matrix A, vector b and chi
 */
__device__ __inline__  void
dNearest(
              float                     undX,
              float                     undY,
              float                     defX,
              float                     defY,

              float                     xMinusCenter,
              float                     yMinusCenter,

              const fittingModelEnum    fittingModel,

              float                    *shared_mat_A_and_vec_B_and_CHI,
              int                       size_A,
              int                       size_A_B,
              int                       size_A_B_chi,

              int                       numberOfColors,
              cudaTextureObject_t       undTexture,
              cudaTextureObject_t       defTexture
         )
{
    int undX0 = (int) ( undX + 0.5f );
    int undY0 = (int) ( undY + 0.5f );

    int ix0             = (int) ( defX + 0.5f );
    int iy0             = (int) ( defY + 0.5f );

    int ix1             = ix0 + 1;
    int iy1             = iy0 + 1;

    //float dx            = defX - ix0;
    //float dy            = defY - iy0;

    //flush this thread's mat_A, vec_B and chi
    memset( shared_mat_A_and_vec_B_and_CHI , 0.f , size_A_B_chi * sizeof( float ) );

    for ( int c = 0 ; c < numberOfColors ; ++c )
    {
        float und_w = (float) tex2D<unsigned char>( undTexture, numberOfColors * undX0 + c , undY0 );

        //  Query deformed image at the interpolation grid points
        int index_x0 = numberOfColors * ix0 + c;
        int index_x1 = numberOfColors * ix1 + c;

        //  Interpolated deformed image
        float def_w    = (float) tex2D<unsigned char>( defTexture , ( index_x0 ) , iy0 );

        //  gradient dw/dx of the Interpolated deformed image
        float def_dwdx = (float) tex2D<unsigned char>( defTexture , ( index_x1 ) , iy0 ) -
                         (float) tex2D<unsigned char>( defTexture , ( index_x0 ) , iy0 );

        //  gradient dw/dy of the Interpolated deformed image
        float def_dwdy = (float) tex2D<unsigned char>( defTexture , ( index_x0 ) , iy1 ) -
                         (float) tex2D<unsigned char>( defTexture , ( index_x0 ) , iy0 );

        float V = und_w - def_w;

        //  Add chi contribution included to the block-shared space
        shared_mat_A_and_vec_B_and_CHI[ size_A_B ] += V * V;

        float H[ 6 ];
        int numberOfModelParameters;

       switch( fittingModel )
       {
           case fm_U:

               numberOfModelParameters = 1;
               dBuildHU( H , def_dwdx );

               break;

           case fm_UV:

               numberOfModelParameters = 2;
               dBuildHUV( H , def_dwdx , def_dwdy );

               break;

           case fm_UVQ:

               numberOfModelParameters = 3;
               dBuildHUVQ( H , def_dwdx , def_dwdy , xMinusCenter , yMinusCenter );

               break;

           case fm_UVUxUyVxVy:

                numberOfModelParameters = 6;
                dBuildHUVUxUyVxVy( H , def_dwdx , def_dwdy , xMinusCenter , yMinusCenter );

                break;

            default:
                break;
        }

        //for ( int p = 0 ; p < numberOfModelParameters ; ++p )
        //{
        //    H [ p ] =   def_dwdx * DtxyDp [                           p ] +
        //                def_dwdy * DtxyDp [ numberOfModelParameters + p ];
        //}


        // Include the contribution of this color/point to the block-shared vector and
        //      symmetric matrix
        for ( int p1 = 0 ; p1 < numberOfModelParameters ; ++p1 )
        {
            shared_mat_A_and_vec_B_and_CHI [ size_A + p1 ] += H [ p1 ] * V;

            int index_A = p1 * numberOfModelParameters;

            for ( int p2 = 0 ; p2 < numberOfModelParameters ; ++p2 )
            {
                shared_mat_A_and_vec_B_and_CHI [ index_A + p2 ] += H [ p1 ] * H [ p2 ];
            }
        }

    }// for c
}

/**
 CUDA device function to perform the bilinear interpolation and get the contribution to the matrix A, vector b and chi
 */
__device__ __inline__  void
dBilinear(
              float                     undX,
              float                     undY,
              float                     defX,
              float                     defY,

              float                     xMinusCenter,
              float                     yMinusCenter,

              const fittingModelEnum    fittingModel,

              float                    *shared_mat_A_and_vec_B_and_CHI,
              int                       size_A,
              int                       size_A_B,
              int                       size_A_B_chi,

              int                       numberOfColors,
              cudaTextureObject_t       undTexture,
              cudaTextureObject_t       defTexture
         )
{
    int undX0 = (int) ( undX + 0.5f );
    int undY0 = (int) ( undY + 0.5f );

    int ix0             = (int) defX;
    int iy0             = (int) defY;

    int ix1             = ix0 + 1;
    int iy1             = iy0 + 1;

    float dx            = defX - ix0;
    float dy            = defY - iy0;

    //flush this thread's mat_A, vec_B and chi
    memset( shared_mat_A_and_vec_B_and_CHI , 0.f , size_A_B_chi * sizeof( float ) );

    for ( int c = 0 ; c < numberOfColors ; ++c )
    {
        float und_w = (float) tex2D<unsigned char>( undTexture, numberOfColors * undX0 + c , undY0 );

        //  Query deformed image at the interpolation grid points
        int index_x0 = numberOfColors * ix0 + c;
        int index_x1 = numberOfColors * ix1 + c;

        float w00 = (float) tex2D<unsigned char>( defTexture, index_x0 , iy0 );
        float w01 = (float) tex2D<unsigned char>( defTexture, index_x0 , iy1 );


        float w10 = (float) tex2D<unsigned char>( defTexture, index_x1 , iy0 );
        float w11 = (float) tex2D<unsigned char>( defTexture, index_x1 , iy1 );


        //the value of the interpolant on the four middle points matches the data
        float all_parameters[ 4 ] = {
                                    (float)tex2D <unsigned char>( defTexture , ( index_x0 ) , iy0 ),

                                    (float)tex2D <unsigned char>( defTexture , ( index_x1 ) , iy0 ) -
                                    (float)tex2D <unsigned char>( defTexture , ( index_x0 ) , iy0 ),

                                    (float)tex2D <unsigned char>( defTexture , ( index_x0 ) , iy1 ) -
                                    (float)tex2D <unsigned char>( defTexture , ( index_x0 ) , iy0 ),

                                    (float)tex2D <unsigned char>( defTexture , ( index_x1 ) , iy1 ) -
                                    (float)tex2D <unsigned char>( defTexture , ( index_x1 ) , iy0 ) -
                                    (float)tex2D <unsigned char>( defTexture , ( index_x0 ) , iy1 ) +
                                    (float)tex2D <unsigned char>( defTexture , ( index_x0 ) , iy0 )
                                                                 };

        //  Interpolated deformed image
        float def_w    = all_parameters[ 0 ]           +
                         all_parameters[ 1 ] * dx      +
                         all_parameters[ 2 ] *      dy +
                         all_parameters[ 3 ] * dx * dy;

        //  gradient dw/dx of the Interpolated deformed image
        float def_dwdx = all_parameters[ 1 ] +
                         all_parameters[ 3 ] *      dy;

        //  gradient dw/dy of the Interpolated deformed image
        float def_dwdy = all_parameters[ 2 ] +
                         all_parameters[ 3 ] * dx     ;

        float V = und_w - def_w;

        //  Add chi contribution included to the block-shared space
        shared_mat_A_and_vec_B_and_CHI[ size_A_B ] += V * V;

        float H[ 6 ];
        int numberOfModelParameters;

       switch( fittingModel )
       {
           case fm_U:

               numberOfModelParameters = 1;
               dBuildHU( H , def_dwdx );

               break;

           case fm_UV:

               numberOfModelParameters = 2;
               dBuildHUV( H , def_dwdx , def_dwdy );

               break;

           case fm_UVQ:

               numberOfModelParameters = 3;
               dBuildHUVQ( H , def_dwdx , def_dwdy , xMinusCenter , yMinusCenter );

               break;

           case fm_UVUxUyVxVy:

                numberOfModelParameters = 6;
                dBuildHUVUxUyVxVy( H , def_dwdx , def_dwdy , xMinusCenter , yMinusCenter );

                break;

            default:
                break;
        }

        //for ( int p = 0 ; p < numberOfModelParameters ; ++p )
        //{
        //    H [ p ] =   def_dwdx * DtxyDp [                           p ] +
        //                def_dwdy * DtxyDp [ numberOfModelParameters + p ];
        //}


        // Include the contribution of this color/point to the block-shared vector and
        //      symmetric matrix
        for ( int p1 = 0 ; p1 < numberOfModelParameters ; ++p1 )
        {
            shared_mat_A_and_vec_B_and_CHI [ size_A + p1 ] += H [ p1 ] * V;

            int index_A = p1 * numberOfModelParameters;

            for ( int p2 = 0 ; p2 < numberOfModelParameters ; ++p2 )
            {
                shared_mat_A_and_vec_B_and_CHI [ index_A + p2 ] += H [ p1 ] * H [ p2 ];
            }
        }

    }// for c
}

/**
 CUDA device function to perform the bicubic interpolation and get the contribution to the matrix A, vector b and chi
 */
__device__ __inline__  void
dBicubic(
              float                     undX,
              float                     undY,
              float                     defX,
              float                     defY,

              float                     xMinusCenter,
              float                     yMinusCenter,

              const fittingModelEnum    fittingModel,

              float                    *shared_mat_A_and_vec_B_and_CHI,
              int                       size_A,
              int                       size_A_B,
              int                       size_A_B_chi,

              int                       numberOfColors,
              cudaTextureObject_t       undTexture,
              cudaTextureObject_t       defTexture
         )
{
    int undX0 = (int) ( undX + 0.5f );
    int undY0 = (int) ( undY + 0.5f );

    int ix0             = (int) defX - 1;
    int iy0             = (int) defY - 1;

    int ix1             = ix0 + 1;
    int iy1             = iy0 + 1;
    int ix2             = ix0 + 2;
    int iy2             = iy0 + 2;
    int ix3             = ix0 + 3;
    int iy3             = iy0 + 3;

    float dx            = defX - ix0;
    float dy            = defY - iy0;

    //const float d_bicubic_interpolation_matrix [ 256 ] = {
    //    16.f,  -20.f,  -20.f,   25.f,   16.f,    8.f,  -20.f,  -10.f,   16.f,  -20.f,    8.f,  -10.f,   16.f,    8.f,    8.f,    4.f,
    //   -48.f,   48.f,   60.f,  -60.f,  -32.f,  -20.f,   40.f,   25.f,  -48.f,   48.f,  -24.f,   24.f,  -32.f,  -20.f,  -16.f,  -10.f,
    //    36.f,  -36.f,  -45.f,   45.f,   20.f,   16.f,  -25.f,  -20.f,   36.f,  -36.f,   18.f,  -18.f,   20.f,   16.f,   10.f,    8.f,
    //    -8.f,    8.f,   10.f,  -10.f,   -4.f,   -4.f,    5.f,    5.f,   -8.f,    8.f,   -4.f,    4.f,   -4.f,   -4.f,   -2.f,   -2.f,
    //   -48.f,   60.f,   48.f,  -60.f,  -48.f,  -24.f,   48.f,   24.f,  -32.f,   40.f,  -20.f,   25.f,  -32.f,  -16.f,  -20.f,  -10.f,
    //   144.f, -144.f, -144.f,  144.f,   96.f,   60.f,  -96.f,  -60.f,   96.f,  -96.f,   60.f,  -60.f,   64.f,   40.f,   40.f,   25.f,
    //  -108.f,  108.f,  108.f, -108.f,  -60.f,  -48.f,   60.f,   48.f,  -72.f,   72.f,  -45.f,   45.f,  -40.f,  -32.f,  -25.f,  -20.f,
    //    24.f,  -24.f,  -24.f,   24.f,   12.f,   12.f,  -12.f,  -12.f,   16.f,  -16.f,   10.f,  -10.f,    8.f,    8.f,    5.f,    5.f,
    //    36.f,  -45.f,  -36.f,   45.f,   36.f,   18.f,  -36.f,  -18.f,   20.f,  -25.f,   16.f,  -20.f,   20.f,   10.f,   16.f,    8.f,
    //  -108.f,  108.f,  108.f, -108.f,  -72.f,  -45.f,   72.f,   45.f,  -60.f,   60.f,  -48.f,   48.f,  -40.f,  -25.f,  -32.f,  -20.f,
    //    81.f,  -81.f,  -81.f,   81.f,   45.f,   36.f,  -45.f,  -36.f,   45.f,  -45.f,   36.f,  -36.f,   25.f,   20.f,   20.f,   16.f,
    //   -18.f,   18.f,   18.f,  -18.f,   -9.f,   -9.f,    9.f,    9.f,  -10.f,   10.f,   -8.f,    8.f,   -5.f,   -5.f,   -4.f,   -4.f,
    //    -8.f,   10.f,    8.f,  -10.f,   -8.f,   -4.f,    8.f,    4.f,   -4.f,    5.f,   -4.f,    5.f,   -4.f,   -2.f,   -4.f,   -2.f,
    //    24.f,  -24.f,  -24.f,   24.f,   16.f,   10.f,  -16.f,  -10.f,   12.f,  -12.f,   12.f,  -12.f,    8.f,    5.f,    8.f,    5.f,
    //   -18.f,   18.f,   18.f,  -18.f,  -10.f,   -8.f,   10.f,    8.f,   -9.f,    9.f,   -9.f,    9.f,   -5.f,   -4.f,   -5.f,   -4.f,
    //     4.f,   -4.f,   -4.f,    4.f,    2.f,    2.f,   -2.f,   -2.f,    2.f,   -2.f,    2.f,   -2.f,    1.f,    1.f,    1.f,    1.f };


    //flush this thread's mat_A, vec_B and chi
    memset( shared_mat_A_and_vec_B_and_CHI , 0.f , size_A_B_chi * sizeof( float ) );

    float px[ 5 ]    = { 0.f,   1.f,   dx,   dx * dx,   dx * dx * dx };
    float py[ 5 ]    = { 0.f,   1.f,   dy,   dy * dy,   dy * dy * dy };

    for ( int c = 0 ; c < numberOfColors ; ++c )
    {
        float und_w = (float) tex2D<unsigned char>( undTexture, numberOfColors * undX0 + c , undY0 );

        //  Query deformed image at the interpolation grid points
        int index_x0 = numberOfColors * ix0 + c;
        int index_x1 = numberOfColors * ix1 + c;
        int index_x2 = numberOfColors * ix2 + c;
        int index_x3 = numberOfColors * ix3 + c;

        float w00 = (float) tex2D<unsigned char>( defTexture, index_x0 , iy0 );
        float w01 = (float) tex2D<unsigned char>( defTexture, index_x0 , iy1 );
        float w02 = (float) tex2D<unsigned char>( defTexture, index_x0 , iy2 );
        float w03 = (float) tex2D<unsigned char>( defTexture, index_x0 , iy3 );

        float w10 = (float) tex2D<unsigned char>( defTexture, index_x1 , iy0 );
        float w11 = (float) tex2D<unsigned char>( defTexture, index_x1 , iy1 );
        float w12 = (float) tex2D<unsigned char>( defTexture, index_x1 , iy2 );
        float w13 = (float) tex2D<unsigned char>( defTexture, index_x1 , iy3 );

        float w20 = (float) tex2D<unsigned char>( defTexture, index_x2 , iy0 );
        float w21 = (float) tex2D<unsigned char>( defTexture, index_x2 , iy1 );
        float w22 = (float) tex2D<unsigned char>( defTexture, index_x2 , iy2 );
        float w23 = (float) tex2D<unsigned char>( defTexture, index_x2 , iy3 );

        float w30 = (float) tex2D<unsigned char>( defTexture, index_x3 , iy0 );
        float w31 = (float) tex2D<unsigned char>( defTexture, index_x3 , iy1 );
        float w32 = (float) tex2D<unsigned char>( defTexture, index_x3 , iy2 );
        float w33 = (float) tex2D<unsigned char>( defTexture, index_x3 , iy3 );

        //  Constructing the interpolation matrix problem
        float interpolation_vector [ 16 ];// number_of_interpolation_parameters = 16

        interpolation_vector[  0 ] = w11; //this is the anchor point of the intepolation. i.e. if dx=dy=0, W(dx,dy)=w11
        interpolation_vector[  1 ] = w21;
        interpolation_vector[  2 ] = w12;
        interpolation_vector[  3 ] = w22;

        //  the derivative in the x-dir is a middle finite diference dw/dx(x,y) = (w[x+1,y]-w[x-1,y])/2
        interpolation_vector[  4 ] = (w21 - w01) / 2.f;
        interpolation_vector[  5 ] = (w31 - w11) / 2.f;
        interpolation_vector[  6 ] = (w22 - w02) / 2.f;
        interpolation_vector[  7 ] = (w32 - w12) / 2.f;

        //  the derivative in the y-dir is a middle finite diference dw/dy(x,y) = (w[x,y+1]-w[x,y-1])/2
        interpolation_vector[  8 ] = (w12 - w10) / 2.f;
        interpolation_vector[  9 ] = (w22 - w20) / 2.f;
        interpolation_vector[ 10 ] = (w13 - w11) / 2.f;
        interpolation_vector[ 11 ] = (w23 - w21) / 2.f;

        //  the derivative in the x-y-dir is a middle finite diference dw^2/dx dy (x,y) = (w[x+1,y+1]+w[x-1,y-1]-w[x-1,y+1]-w[x+1,y-1])/4
        interpolation_vector[ 12 ] = (w22 + w00 - w20 - w02) / 4.f;
        interpolation_vector[ 13 ] = (w32 + w10 - w30 - w12) / 4.f;
        interpolation_vector[ 14 ] = (w23 + w01 - w21 - w03) / 4.f;
        interpolation_vector[ 15 ] = (w33 + w11 - w31 - w13) / 4.f;

        //  Solve the system - interpolation_matrix is already inverted
        float interpolation_parameters [ 16 ];  // number_of_interpolation_parameters = 16

        //for     ( int ik = 0; ik < 16; ++ik )
        //{
        //    interpolation_parameters [ ik ] = 0.f;
        //
        //    for ( int jk = 0; jk < 16; ++jk )
        //    {
        //        interpolation_parameters [ ik ] +=
        //             d_bicubic_interpolation_matrix [ ik * 16 + jk ] * interpolation_vector[ jk ];
        //    }
        //}

        interpolation_parameters [  0 ] =    16 * interpolation_vector[  0 ] +  -20 * interpolation_vector[  1 ] +  -20 * interpolation_vector[  2 ] +   25 * interpolation_vector[  3 ] +   16 * interpolation_vector[  4 ] +   8 * interpolation_vector[  5 ] + -20 * interpolation_vector[  6 ] + -10 * interpolation_vector[  7 ] +  16 * interpolation_vector[  8 ] + -20 * interpolation_vector[  9 ] +   8 * interpolation_vector[ 10 ] + -10 * interpolation_vector[ 11 ] +  16 * interpolation_vector[ 12 ] +   8 * interpolation_vector[ 13 ] +   8 * interpolation_vector[ 14 ] +   4 * interpolation_vector[ 15 ];
        interpolation_parameters [  1 ] =   -48 * interpolation_vector[  0 ] +   48 * interpolation_vector[  1 ] +   60 * interpolation_vector[  2 ] +  -60 * interpolation_vector[  3 ] +  -32 * interpolation_vector[  4 ] + -20 * interpolation_vector[  5 ] +  40 * interpolation_vector[  6 ] +  25 * interpolation_vector[  7 ] + -48 * interpolation_vector[  8 ] +  48 * interpolation_vector[  9 ] + -24 * interpolation_vector[ 10 ] +  24 * interpolation_vector[ 11 ] + -32 * interpolation_vector[ 12 ] + -20 * interpolation_vector[ 13 ] + -16 * interpolation_vector[ 14 ] + -10 * interpolation_vector[ 15 ];
        interpolation_parameters [  2 ] =	 36 * interpolation_vector[  0 ] +  -36 * interpolation_vector[  1 ] +  -45 * interpolation_vector[  2 ] +   45 * interpolation_vector[  3 ] +   20 * interpolation_vector[  4 ] +  16 * interpolation_vector[  5 ] + -25 * interpolation_vector[  6 ] + -20 * interpolation_vector[  7 ] +  36 * interpolation_vector[  8 ] + -36 * interpolation_vector[  9 ] +  18 * interpolation_vector[ 10 ] + -18 * interpolation_vector[ 11 ] +  20 * interpolation_vector[ 12 ] +  16 * interpolation_vector[ 13 ] +  10 * interpolation_vector[ 14 ] +   8 * interpolation_vector[ 15 ];
        interpolation_parameters [  3 ] =	 -8 * interpolation_vector[  0 ] +    8 * interpolation_vector[  1 ] +   10 * interpolation_vector[  2 ] +  -10 * interpolation_vector[  3 ] +   -4 * interpolation_vector[  4 ] +  -4 * interpolation_vector[  5 ] +   5 * interpolation_vector[  6 ] +   5 * interpolation_vector[  7 ] +  -8 * interpolation_vector[  8 ] +   8 * interpolation_vector[  9 ] +  -4 * interpolation_vector[ 10 ] +   4 * interpolation_vector[ 11 ] +  -4 * interpolation_vector[ 12 ] +  -4 * interpolation_vector[ 13 ] +  -2 * interpolation_vector[ 14 ] +  -2 * interpolation_vector[ 15 ];
        interpolation_parameters [  4 ] =	-48 * interpolation_vector[  0 ] +   60 * interpolation_vector[  1 ] +   48 * interpolation_vector[  2 ] +  -60 * interpolation_vector[  3 ] +  -48 * interpolation_vector[  4 ] + -24 * interpolation_vector[  5 ] +  48 * interpolation_vector[  6 ] +  24 * interpolation_vector[  7 ] + -32 * interpolation_vector[  8 ] +  40 * interpolation_vector[  9 ] + -20 * interpolation_vector[ 10 ] +  25 * interpolation_vector[ 11 ] + -32 * interpolation_vector[ 12 ] + -16 * interpolation_vector[ 13 ] + -20 * interpolation_vector[ 14 ] + -10 * interpolation_vector[ 15 ];
        interpolation_parameters [  5 ] =	144 * interpolation_vector[  0 ] + -144 * interpolation_vector[  1 ] + -144 * interpolation_vector[  2 ] +  144 * interpolation_vector[  3 ] +   96 * interpolation_vector[  4 ] +  60 * interpolation_vector[  5 ] + -96 * interpolation_vector[  6 ] + -60 * interpolation_vector[  7 ] +  96 * interpolation_vector[  8 ] + -96 * interpolation_vector[  9 ] +  60 * interpolation_vector[ 10 ] + -60 * interpolation_vector[ 11 ] +  64 * interpolation_vector[ 12 ] +  40 * interpolation_vector[ 13 ] +  40 * interpolation_vector[ 14 ] +  25 * interpolation_vector[ 15 ];
        interpolation_parameters [  6 ] =  -108 * interpolation_vector[  0 ] +  108 * interpolation_vector[  1 ] +  108 * interpolation_vector[  2 ] + -108 * interpolation_vector[  3 ] +  -60 * interpolation_vector[  4 ] + -48 * interpolation_vector[  5 ] +  60 * interpolation_vector[  6 ] +  48 * interpolation_vector[  7 ] + -72 * interpolation_vector[  8 ] +  72 * interpolation_vector[  9 ] + -45 * interpolation_vector[ 10 ] +  45 * interpolation_vector[ 11 ] + -40 * interpolation_vector[ 12 ] + -32 * interpolation_vector[ 13 ] + -25 * interpolation_vector[ 14 ] + -20 * interpolation_vector[ 15 ];
        interpolation_parameters [  7 ] =	 24 * interpolation_vector[  0 ] +  -24 * interpolation_vector[  1 ] +  -24 * interpolation_vector[  2 ] +   24 * interpolation_vector[  3 ] +   12 * interpolation_vector[  4 ] +  12 * interpolation_vector[  5 ] + -12 * interpolation_vector[  6 ] + -12 * interpolation_vector[  7 ] +  16 * interpolation_vector[  8 ] + -16 * interpolation_vector[  9 ] +  10 * interpolation_vector[ 10 ] + -10 * interpolation_vector[ 11 ] +   8 * interpolation_vector[ 12 ] +   8 * interpolation_vector[ 13 ] +   5 * interpolation_vector[ 14 ] +   5 * interpolation_vector[ 15 ];
        interpolation_parameters [  8 ] =	 36 * interpolation_vector[  0 ] +  -45 * interpolation_vector[  1 ] +  -36 * interpolation_vector[  2 ] +   45 * interpolation_vector[  3 ] +   36 * interpolation_vector[  4 ] +  18 * interpolation_vector[  5 ] + -36 * interpolation_vector[  6 ] + -18 * interpolation_vector[  7 ] +  20 * interpolation_vector[  8 ] + -25 * interpolation_vector[  9 ] +  16 * interpolation_vector[ 10 ] + -20 * interpolation_vector[ 11 ] +  20 * interpolation_vector[ 12 ] +  10 * interpolation_vector[ 13 ] +  16 * interpolation_vector[ 14 ] +   8 * interpolation_vector[ 15 ];
        interpolation_parameters [  9 ] =  -108 * interpolation_vector[  0 ] +  108 * interpolation_vector[  1 ] +  108 * interpolation_vector[  2 ] + -108 * interpolation_vector[  3 ] +  -72 * interpolation_vector[  4 ] + -45 * interpolation_vector[  5 ] +  72 * interpolation_vector[  6 ] +  45 * interpolation_vector[  7 ] + -60 * interpolation_vector[  8 ] +  60 * interpolation_vector[  9 ] + -48 * interpolation_vector[ 10 ] +  48 * interpolation_vector[ 11 ] + -40 * interpolation_vector[ 12 ] + -25 * interpolation_vector[ 13 ] + -32 * interpolation_vector[ 14 ] + -20 * interpolation_vector[ 15 ];
        interpolation_parameters [ 10 ] =	 81 * interpolation_vector[  0 ] +  -81 * interpolation_vector[  1 ] +  -81 * interpolation_vector[  2 ] +   81 * interpolation_vector[  3 ] +   45 * interpolation_vector[  4 ] +  36 * interpolation_vector[  5 ] + -45 * interpolation_vector[  6 ] + -36 * interpolation_vector[  7 ] +  45 * interpolation_vector[  8 ] + -45 * interpolation_vector[  9 ] +  36 * interpolation_vector[ 10 ] + -36 * interpolation_vector[ 11 ] +  25 * interpolation_vector[ 12 ] +  20 * interpolation_vector[ 13 ] +  20 * interpolation_vector[ 14 ] +  16 * interpolation_vector[ 15 ];
        interpolation_parameters [ 11 ] =	-18 * interpolation_vector[  0 ] +   18 * interpolation_vector[  1 ] +   18 * interpolation_vector[  2 ] +  -18 * interpolation_vector[  3 ] +   -9 * interpolation_vector[  4 ] +  -9 * interpolation_vector[  5 ] +   9 * interpolation_vector[  6 ] +   9 * interpolation_vector[  7 ] + -10 * interpolation_vector[  8 ] +  10 * interpolation_vector[  9 ] +  -8 * interpolation_vector[ 10 ] +   8 * interpolation_vector[ 11 ] +  -5 * interpolation_vector[ 12 ] +  -5 * interpolation_vector[ 13 ] +  -4 * interpolation_vector[ 14 ] +  -4 * interpolation_vector[ 15 ];
        interpolation_parameters [ 12 ] =	 -8 * interpolation_vector[  0 ] +   10 * interpolation_vector[  1 ] +    8 * interpolation_vector[  2 ] +  -10 * interpolation_vector[  3 ] +   -8 * interpolation_vector[  4 ] +  -4 * interpolation_vector[  5 ] +   8 * interpolation_vector[  6 ] +   4 * interpolation_vector[  7 ] +  -4 * interpolation_vector[  8 ] +   5 * interpolation_vector[  9 ] +  -4 * interpolation_vector[ 10 ] +   5 * interpolation_vector[ 11 ] +  -4 * interpolation_vector[ 12 ] +  -2 * interpolation_vector[ 13 ] +  -4 * interpolation_vector[ 14 ] +  -2 * interpolation_vector[ 15 ];
        interpolation_parameters [ 13 ] =	 24 * interpolation_vector[  0 ] +  -24 * interpolation_vector[  1 ] +  -24 * interpolation_vector[  2 ] +   24 * interpolation_vector[  3 ] +   16 * interpolation_vector[  4 ] +  10 * interpolation_vector[  5 ] + -16 * interpolation_vector[  6 ] + -10 * interpolation_vector[  7 ] +  12 * interpolation_vector[  8 ] + -12 * interpolation_vector[  9 ] +  12 * interpolation_vector[ 10 ] + -12 * interpolation_vector[ 11 ] +   8 * interpolation_vector[ 12 ] +   5 * interpolation_vector[ 13 ] +   8 * interpolation_vector[ 14 ] +   5 * interpolation_vector[ 15 ];
        interpolation_parameters [ 14 ] =	-18 * interpolation_vector[  0 ] +   18 * interpolation_vector[  1 ] +   18 * interpolation_vector[  2 ] +  -18 * interpolation_vector[  3 ] +  -10 * interpolation_vector[  4 ] +  -8 * interpolation_vector[  5 ] +  10 * interpolation_vector[  6 ] +   8 * interpolation_vector[  7 ] +  -9 * interpolation_vector[  8 ] +   9 * interpolation_vector[  9 ] +  -9 * interpolation_vector[ 10 ] +   9 * interpolation_vector[ 11 ] +  -5 * interpolation_vector[ 12 ] +  -4 * interpolation_vector[ 13 ] +  -5 * interpolation_vector[ 14 ] +  -4 * interpolation_vector[ 15 ];
        interpolation_parameters [ 15 ] =	  4 * interpolation_vector[  0 ] +   -4 * interpolation_vector[  1 ] +   -4 * interpolation_vector[  2 ] +    4 * interpolation_vector[  3 ] +    2 * interpolation_vector[  4 ] +   2 * interpolation_vector[  5 ] +  -2 * interpolation_vector[  6 ] +  -2 * interpolation_vector[  7 ] +   2 * interpolation_vector[  8 ] +  -2 * interpolation_vector[  9 ] +   2 * interpolation_vector[ 10 ] +  -2 * interpolation_vector[ 11 ] +   1 * interpolation_vector[ 12 ] +   1 * interpolation_vector[ 13 ] +   1 * interpolation_vector[ 14 ] +   1 * interpolation_vector[ 15 ];

        // Initialize results to 0
        float def_w    = 0.f;
        float def_dwdx = 0.f;
        float def_dwdy = 0.f;

        for ( int jk = 0; jk < 4; jk++)
        {
            int index_jk =  jk * 4;

            for ( int ik = 0; ik < 4; ik++)
            {
                int index_jk_ik = index_jk + ik;

                def_w    +=      interpolation_parameters [ index_jk_ik ]  *  py[ jk + 1 ]  *  px[ ik + 1 ];

                def_dwdx += ik * interpolation_parameters [ index_jk_ik ]  *  py[ jk + 1 ]  *  px[ ik     ];

                def_dwdy += jk * interpolation_parameters [ index_jk_ik ]  *  py[ jk     ]  *  px[ ik + 1 ];
            }
        }

        float V = und_w - def_w;

        //  Add chi contribution included to the block-shared space
        shared_mat_A_and_vec_B_and_CHI[ size_A_B ] += V * V;

        float H[ 6 ];
        int numberOfModelParameters;

       switch( fittingModel )
       {
           case fm_U:

               numberOfModelParameters = 1;
               dBuildHU( H , def_dwdx );

               break;

           case fm_UV:

               numberOfModelParameters = 2;
               dBuildHUV( H , def_dwdx , def_dwdy );

               break;

           case fm_UVQ:

               numberOfModelParameters = 3;
               dBuildHUVQ( H , def_dwdx , def_dwdy , xMinusCenter , yMinusCenter );

               break;

           case fm_UVUxUyVxVy:

                numberOfModelParameters = 6;
                dBuildHUVUxUyVxVy( H , def_dwdx , def_dwdy , xMinusCenter , yMinusCenter );

                break;

            default:
                break;
        }

        //for ( int p = 0 ; p < numberOfModelParameters ; ++p )
        //{
        //    H [ p ] =   def_dwdx * DtxyDp [                           p ] +
        //                def_dwdy * DtxyDp [ numberOfModelParameters + p ];
        //}


        // Include the contribution of this color/point to the block-shared vector and
        //      symmetric matrix
        for ( int p1 = 0 ; p1 < numberOfModelParameters ; ++p1 )
        {
            shared_mat_A_and_vec_B_and_CHI [ size_A + p1 ] += H [ p1 ] * V;

            int index_A = p1 * numberOfModelParameters;

            for ( int p2 = 0 ; p2 < numberOfModelParameters ; ++p2 )
            {
                shared_mat_A_and_vec_B_and_CHI [ index_A + p2 ] += H [ p1 ] * H [ p2 ];
            }
        }

    }// for c
}

/**
 CUDA device function to build the H vector for the U model
 */
__device__ __inline__  void
dBuildHU(
        float *H,

        const float def_dwdx
        )
{
    H [ 0 ] = def_dwdx;
}

/**
 CUDA device function to build the H vector for the UV model
 */
__device__ __inline__  void
dBuildHUV(
        float *H,

        const float def_dwdx,
        const float def_dwdy
        )
{
    H [ 0 ] = def_dwdx;
    H [ 1 ] = def_dwdy;
}

/**
 CUDA device function to build the H vector for the UVQ model
 */
__device__ __inline__  void
dBuildHUVQ(
        float *H,

        const float def_dwdx,
        const float def_dwdy,

        const float xMinusCenter,
        const float yMinusCenter
        )
{
    H [ 0 ] = def_dwdx;
    H [ 1 ] = def_dwdy;
    H [ 2 ] = - def_dwdx * yMinusCenter + def_dwdy * xMinusCenter;
}

/**
 CUDA device function to build the H vector for the UVUxUyVxVy model
 */
__device__ __inline__  void
dBuildHUVUxUyVxVy(
        float *H,

        const float def_dwdx,
        const float def_dwdy,

        const float xMinusCenter,
        const float yMinusCenter
        )
{
    H [ 0 ] = def_dwdx;
    H [ 1 ] = def_dwdy;
    H [ 2 ] = def_dwdx * xMinusCenter;
    H [ 3 ] = def_dwdx * yMinusCenter;
    H [ 4 ] = def_dwdy * xMinusCenter;
    H [ 5 ] = def_dwdy * yMinusCenter;
}
