/****************************************************************************
**
**  This software was developed by Javier Gonzalez on Feb 2018
**
**  This class controls a set of GPUs to perform digital image correlation
**
****************************************************************************/

#include "cuda_class.cuh"

CudaClass::CudaClass()
{
    #if DEBUG_CUDA
    printf("CudaClass::CudaClass\n" );
    #endif

    cudaMallocHost ( (void**) &pinnedChi   , sizeof( float ) );
    cudaMallocHost ( (void**) &lastGoodChi , sizeof( float ) );
}

CudaClass::~CudaClass()
{
    #if DEBUG_CUDA
    printf("CudaClass::~CudaClass\n" );
    #endif

    if ( correlationStream )
    {
        cudaError_t err = cudaSuccess;
        err = cudaStreamDestroy( correlationStream );
        if ( err != cudaSuccess )
        {
            printf( "Failed to Destroy correlationStream (error code %s)!\n", cudaGetErrorString( err ) );
            exit( EXIT_FAILURE );
        }
    }

    cudaFreeHost( pinnedChi   );
    cudaFreeHost( lastGoodChi );
}

int CudaClass::initialize()
{
    //  This method returns the number of devices to MainApp, and MainApp
    //  will disable the GPU mode if the number of devices is 0. If the GPU
    //  mode is disabled, there will be no more calls to this class.
    //  I decided not to include an if statement in every method of this
    //  class to check on this.

    if ( initialize_devices )
    {
        //  Initialize devices only once
        initialize_devices = false;

        // Calls every GPU to reduce initial latency ( control when it happens )
        cudaError_t err = cudaGetDeviceCount( &deviceCount );
        if ( err != cudaSuccess )
        {
            printf( "Failed to count GPUs (error code %s)!\n", cudaGetErrorString( err ) );
            exit( EXIT_FAILURE );
        }

        #if _WINDOWS
		deviceCount = 1;
        printf( "CudaClass::initialize Using only %d GPU on WINDOWS compile\n", deviceCount );
        #endif

        devicesAvailable = deviceCount;

        err = cudaStreamCreate( &correlationStream );
        if ( err != cudaSuccess )
        {
            printf( "Failed to create correlationStream (error code %s)!\n", cudaGetErrorString( err ) );
            exit( EXIT_FAILURE );
        }
    }

    return devicesAvailable;
}

void CudaClass::set_deviceCount( int deviceCount_in )
{
    deviceCount = deviceCount_in;

    #if DEBUG_CUDA
         printf("CUDA: set_deviceCount: Using %d devices\n", deviceCount);
    #endif
}

void CudaClass::set_fitting_model( fittingModelEnum fittingModel_in )
{
    fittingModel = fittingModel_in;
    number_of_model_parameters = ModelClass::get_number_of_model_parameters( fittingModel );
}

void CudaClass::set_interpolation_model( interpolationModelEnum interpolationModel_in)
{
    interpolationModel = interpolationModel_in;
}

void CudaClass::set_max_iters( int maximum_iterations_ )
{
    maximum_iterations = maximum_iterations_;
}

void CudaClass::set_precision( float required_precision_ )
{
    required_precision = required_precision_;
}

CorrelationResult *CudaClass::correlate( int iSector , float *initial_guess_ , frame_results &results )
{
    #if DEBUG_NEWTON_RAPHSON_CUDA
         printf( "CudaClass::correlate\n" );

         printf( "CudaClass::correlate: Initial_guess:\n" );
         int p = ModelClass::get_number_of_model_parameters( fittingModel );
         for ( int p = 0 ; p < number_of_model_parameters ; ++p )
         {
             printf( "%14.5e\t" , initial_guess_[ p ] );
         }
          printf( "\n" );
    #endif

    errorEnum errorCode = error_none;
    float min_lambda = 1e-9f;
    float max_lambda = 1e9f;
    int totalIterations = 0;

    // Upload initialGuess for level0 to the GPU via the pyramid/polygon object
    // initialGuess -> lastGoodParameters , tentativeParameters , savedParameters
    cudaPyramidManager.initializeParametersLevel0( iSector , initial_guess_ );

    int pyramid_start = cudaPyramidManager.getPyramidStart();
    int pyramid_step  = cudaPyramidManager.getPyramidStep();
    int pyramid_stop  = cudaPyramidManager.getPyramidStop();

    for ( int pyramidLevel = pyramid_stop ; pyramidLevel >= pyramid_start ; pyramidLevel -= pyramid_step )
    {
        // Put a marker on the nvvp CUDA profiler
        nvtxRangePushA ( "CudaClass::correlate level" );

        float lambda = 0.0001f;

        // Scale lastGood for this pyramid level. tentative and savedParameters are copied from lastGood
        cudaPyramidManager.scaleParametersForLevel( iSector , pyramidLevel );

        //###########################################################################################
        //
        // Find initial chi from the initial guess. Make that the lastGoodChi and start the iteration
        //

        // lastGoodParameters -> NR -> tentativeParameters AND lastGoodChi = chi( lastGoodParameters )
        findNewParameters( iSector , pyramidLevel , parType_lastGood , parType_saved , lastGoodChi , lambda );
        #if DEBUG_NEWTON_RAPHSON_CUDA
        printf( "CudaClass::correlate Initial chi is %f\n",
                *lastGoodChi );
        #endif

        bool useSavedParameters = true;

        //###########################################################################################
        //
        // Iteration to find correlation coefficients
        //
        for ( int iteration = 1 ; iteration <= maximum_iterations + 1 ; ++iteration )
        {
            #if DEBUG_NEWTON_RAPHSON_CUDA
                 printf( "\n\nCudaClass::correlate Starting iteration %d, pyramid level %d, sector = %d\n",
                        iteration,
                        pyramidLevel,
                        iSector );
            #endif
            if ( iteration > maximum_iterations || lambda >= max_lambda )
            {
                errorCode = error_correlation_max_iters_reached;
                break;
            }
            else
            {
                totalIterations++;
            }

            //###########################################################################################
            //
            // Find tentative parameter set - We are saving parameters from the last chi computation that we
            //      use if the corresponding chi was better than the "lastGoodChi".
            //      This happens most iterations and saves time. However, if the corresponding chi is
            //      worst, then we recompute the parameters.
            //
            if ( useSavedParameters )
            {
            #if DEBUG_NEWTON_RAPHSON_CUDA
                printf("CudaClass::correlate Using saved correlation coefficients\n");
            #endif
                cudaPyramidManager.transferParameters( iSector , parType_saved , parType_tentative );
            }
            else
            {
            #if DEBUG_NEWTON_RAPHSON_CUDA
                printf("CudaClass::correlate Finding next correlation coefficients the hard way\n");
            #endif

                // Recompute tentativeParameters from the lastGoodParameters using a larger lambda
                findNewParameters( iSector , pyramidLevel , parType_lastGood , parType_tentative , pinnedChi , lambda );

            } //else - finding new parameter set

            //###########################################################################################
            //
            //  Computation of CHI FOR THE TENTATIVE set of model parameters

            //  Compute chi associated with the new tentativeParameters
            findNewParameters( iSector , pyramidLevel , parType_tentative , parType_saved , pinnedChi , lambda );

            #if DEBUG_NEWTON_RAPHSON_CUDA
            printf( "CudaClass::correlate New chi is %f\n",
                    pinnedChi );
            #endif

            //  Compares delta chi based on lastGoodChi(last_good_model_parameters) and
            //      chi(model_parameters). However, it does not act on this info until the
            //      parameters are updated

            // Make sure the last chi is already available
            cudaStreamSynchronize( correlationStream );

            float delta_chi = std::abs( ( *lastGoodChi - *pinnedChi ) / ( fmaxf( *lastGoodChi , *pinnedChi ) + required_precision ) );

            #if DEBUG_NEWTON_RAPHSON_CUDA

                printf("CUDA iteration = %4d: ", iteration);

                printf("last good chi: %12.4e: ", *lastGoodChi);
                printf("new chi: %12.4e delta chi: %12.4e lambda: %12.4e required_precission = %f\n", *pinnedChi, delta_chi, lambda , required_precision );

            #endif

            if ( *pinnedChi <= *lastGoodChi ) //converging step - record the new "best" parameters
            {
                *lastGoodChi = *pinnedChi;
                lambda = fmaxf( lambda * 0.4f , min_lambda );

                // tentativeParameters (last result) -> lastGoodParameters
                cudaPyramidManager.transferParameters( iSector , parType_tentative , parType_lastGood );

                useSavedParameters = true;

                #if DEBUG_NEWTON_RAPHSON_CUDA
                    printf(" # CONVERGING\n" );
                #endif
            }
            else	//diverging step - increase lambda and revert to last "good" set of parameters
            {
                lambda = fminf( lambda * 10.0f , max_lambda );

                useSavedParameters = false;

                #if DEBUG_NEWTON_RAPHSON_CUDA
                    printf("\n");
                #endif
            }

            // Was convergence reached?
            if ( delta_chi < required_precision )
            {
                #if DEBUG_NEWTON_RAPHSON_CUDA
                     printf("CudaClass::correlate Convergence reached in %d iterations at a delta_chi =  %6f\n", totalIterations, delta_chi);
                #endif
                break;
            }

        } //iterations loop

        nvtxRangePop();
    }

    // Bring center, number of points and error status from the GPU to the CPU with one transfer on host pinned memory
    CorrelationResult *cpuCorrelationResults = cudaPyramidManager.getCorrelationResultsToCPU( iSector );

    cpuCorrelationResults->chi              = *lastGoodChi;
    cpuCorrelationResults->iterations       =  totalIterations;
    cpuCorrelationResults->errorCode        =  errorCode;

    memcpy ( initial_guess_ ,
             cpuCorrelationResults->resultingParameters ,
             number_of_model_parameters * sizeof( float ) );

    return cpuCorrelationResults;
}

errorEnum CudaClass::findNewParameters( int iSector , int pyramidLevel,
                                        parameterTypeEnum parSrc,
                                        parameterTypeEnum parDst,
                                        float *chi,
                                        float lambda )
{
    #if DEBUG_CUDA
    printf( "CudaClass::findNewParameters: Start\n" );
    #endif

    errorEnum error = error_none;

    error = NewtonRaphsonStep( iSector , pyramidLevel , parSrc , lambda );
    if ( error ) return error;

    // Put a marker on the nvvp CUDA profiler
    nvtxRangePushA ( "callCusolver" );

    error = cudaSolverManager.callCusolver( iSector , chi );
    if ( error ) return error_cuSolver;

    nvtxRangePop();

    // Put a marker on the nvvp CUDA profiler
    nvtxRangePushA ( "updateParameters" );

    // Saves the parameter increment plus the tentativeParameters into tentativeParameters via thrust operations on GPU
    cudaPyramidManager.updateParameters( iSector , number_of_model_parameters , parSrc , parDst , correlationStream );

    nvtxRangePop();

    return error_none;
}

errorEnum CudaClass::NewtonRaphsonStep( int iSector , int pyramidLevel , parameterTypeEnum parSrc , float lambda )
{
    int iGPU = 0;
    errorEnum errorCode = error_none;

    //for ( int iGPU = 0 ; iGPU < deviceCount ; ++iGPU )
    //{
        cudaError_t err = cudaSetDevice( iGPU );
        if ( err != cudaSuccess )
        {
            printf( "Failed to set device (error code %s)!\n" , cudaGetErrorString( err ) );
            exit( EXIT_FAILURE );
        }

        cudaTextureObject_t  undTexture     = cudaPyramidManager.getUndTexture ( pyramidLevel );
        cudaTextureObject_t  defTexture     = cudaPyramidManager.getDefTexture ( pyramidLevel );
        int                  numberOfPoints = cudaPyramidManager.getNumberOfPoints( iSector , pyramidLevel );
        float                scaling        = 1.f / ( (float) numberOfPoints );
        float               *undX_ptr       = cudaPyramidManager.getUndXPtr    ( iSector , pyramidLevel );
        float               *undY_ptr       = cudaPyramidManager.getUndYPtr    ( iSector , pyramidLevel );
        float               *undCenter      = cudaPyramidManager.getUndCenter  ( iSector , pyramidLevel );
        float               *globalABChi    = cudaPyramidManager.getGlobalABChi( iSector );
        float               *parameters     = cudaPyramidManager.getParameters( iSector , parSrc );


        int                  blocksPerGrid  = ( numberOfPoints + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;

        int sharedMemorySize =  sizeof( float ) *
                                ( 1 + number_of_model_parameters * ( 1 + number_of_model_parameters ) ) *
                                THREADS_PER_BLOCK;

        #if DEBUG_CUDA_POLYGON
        printf( "\nCudaClass::NewtonRaphsonStep Model numberOfPoints = %d , isector = %d , pyramidLevel = %d\n" ,
                numberOfPoints , iSector , pyramidLevel );

        printf( "CudaClass::NewtonRaphsonStep parameters used\n" );

        float *h_par = new float[ number_of_model_parameters ];
        cudaMemcpy( h_par ,
                    parameters ,
                    number_of_model_parameters * sizeof( float ),
                    cudaMemcpyDeviceToHost );

        for ( int i = 0 ; i < number_of_model_parameters ; ++i )
        {
            printf( "%14.4e" , h_par[ i ] );
        }
        printf("\n");

        printf( "CudaClass::NewtonRaphsonStep center used\n" );

        cudaMemcpy( h_par ,
                    undCenter ,
                    2 * sizeof( float ),
                    cudaMemcpyDeviceToHost );

        for ( int i = 0 ; i < 2 ; ++i )
        {
            printf( "%14.4e" , h_par[ i ] );
        }
        printf("\n");

        fflush( stdout );

        delete[] h_par;

        #endif

        kCorrelation<<< blocksPerGrid , THREADS_PER_BLOCK , sharedMemorySize , correlationStream >>>
        (
            parameters,

            fittingModel,
            interpolationModel,

            number_of_colors,
            undTexture,
            defTexture,            

            numberOfPoints,
            undX_ptr,
            undY_ptr,
            undCenter,

            globalABChi
        );

        #if DEBUG_CUDA
            printf( "CudaClass::NewtonRaphsonStep: kCorrelation kernel launched with %d blocks of %d threads with %zd bytes of shared memory per block\n",
                    blocksPerGrid,
                    THREADS_PER_BLOCK,
                    sharedMemorySize );
        #endif

        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf( "Failed to launch correlation kernel (error code %s)!\n",
                    cudaGetErrorString( err ) );
            exit( EXIT_FAILURE );
        }
    //}
    //
    //for ( int iGPU = 0 ; iGPU < deviceCount ; ++iGPU )
    //{
    #if DEBUG_CUDA
        printf( "\nCudaClass::NewtonRaphsonStep Aggregation: numberOfPoints = %d in GPU %d \n" , numberOfPoints , iGPU );
    #endif

        err = cudaSetDevice( iGPU );
        if ( err != cudaSuccess )
        {
            printf( "Failed to set device (error code %s)!\n" , cudaGetErrorString( err ) );
            exit( EXIT_FAILURE );
        }

        //-------------------------------------------------------------------------------------------
        //
        // Launch second kernel many times to perform global aggregation of block results
        //      into globalABChi, layer by layer
        //
        //-------------------------------------------------------------------------------------------

        while ( blocksPerGrid > 1 )
        {
            int reducerBlocksPerGrid  = ( blocksPerGrid + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;

            //Every call reduces the number of ABchi by a factor of 256 ( THREADS_PER_BLOCK )
            k_global_reduction <<< reducerBlocksPerGrid , THREADS_PER_BLOCK , sharedMemorySize , correlationStream >>>
            (
                blocksPerGrid, //s,
                globalABChi,
                number_of_model_parameters
            );

            err = cudaGetLastError();
            if ( err != cudaSuccess )
            {
                printf( "Failed to launch global aggregation kernel (error code %s)!\n" , cudaGetErrorString( err ) );
                exit( EXIT_FAILURE );
            }

            blocksPerGrid = reducerBlocksPerGrid;
        }

        // Build LS problem in GPU0
        k_build_LS_problem_in_GPU0 <<< 1 , 1 , 0 , correlationStream >>>
        ( globalABChi,
          number_of_model_parameters,
          scaling,
          lambda );


    //} //loop iGPU

    return errorCode;

} //NewtonRaphsonStep

void CudaClass::resetImagePyramids ( const std::string undPath ,
                                     const std::string defPath ,
                                     const std::string nxtPath ,
                                     colorEnum color_mode ,
                                     const int start ,
                                     const int step ,
                                     const int stop )
{
    cv::ImreadModes color_flag;

    switch ( color_mode )
    {
        case color_monochrome:
            color_flag = cv::IMREAD_GRAYSCALE;
            break;

        case color_color:
            color_flag = cv::IMREAD_ANYCOLOR;
            break;

        default:
            assert( false );
            break;
    }

    // Put a marker on the nvvp CUDA profiler
    nvtxRangePushA ( "CudaClass::resetImagePyramids read und image" );
    cv::Mat undCvImage = cv::imread( undPath , color_flag );
    nvtxRangePop();

    // Put a marker on the nvvp CUDA profiler
    nvtxRangePushA ( "CudaClass::resetImagePyramids read def image" );
    cv::Mat defCvImage = cv::imread( defPath , color_flag );
    nvtxRangePop();

    cv::Mat nxtCvImage;
    if ( !nxtPath.empty() )
    {
        // Put a marker on the nvvp CUDA profiler
        nvtxRangePushA ( "CudaClass::resetImagePyramids read nxt image" );
        nxtCvImage = cv::imread( nxtPath , color_flag );
        nvtxRangePop();
    }

    number_of_colors = undCvImage.channels();
    assert ( number_of_colors == defCvImage.channels() );

    cudaPyramidManager.resetImagePyramids( undCvImage , defCvImage , nxtCvImage , start , step , stop );
}

void CudaClass::resetNextPyramid( const std::string nxtPath )
{

#if DEBUG_CUDA_PYRAMID
        printf( "CudaClass::resetNextPyramid with %s\n" ,
                nxtPath.c_str() );
#endif


    // Put a marker on the nvvp CUDA profiler
    nvtxRangePushA ( "CudaClass::resetNextPyramid read image" );

    cv::Mat nxtCvImage;

    if ( tempQ.empty() )
    {
        cv::ImreadModes color_flag;

        switch ( number_of_colors )
        {
            case 1:
                color_flag = cv::IMREAD_GRAYSCALE;
                break;

            case 3:
                color_flag = cv::IMREAD_ANYCOLOR;
                break;

            default:
                assert( false );
                break;
        }
        nxtCvImage = cv::imread( nxtPath , color_flag );
    }
    else
    {
        nxtCvImage = tempQ.front();
        tempQ.pop();
    }

    nvtxRangePop();

    assert ( number_of_colors == nxtCvImage.channels() );

    cudaPyramidManager.newNxtPyramid( nxtCvImage );
}

void CudaClass::makeUndPyramidFromDef()
{
    cudaPyramidManager.makeUndPyramidFromDef();
}

void CudaClass::makeDefPyramidFromNxt()
{
    cudaPyramidManager.makeDefPyramidFromNxt();
}

void CudaClass::updatePolygon( int iSector, deformationDescriptionEnum deformationDescription )
{
    cudaPyramidManager.updatePolygon( iSector , deformationDescription );
}

errorEnum CudaClass::resetPolygon( int iSector , int x0 , int y0 , int x1 , int y1 )
{
    errorEnum corrError;    

    cudaPyramidManager.resetPolygon( iSector , x0 , y0 , x1 , y1, fittingModel );
    corrError = cudaSolverManager.setCuSolver( iSector , number_of_model_parameters , correlationStream );

    return corrError;
}

errorEnum CudaClass::resetPolygon( int iSector , float r , float dr , float a , float da ,
                                  float cx , float cy , int as )
{
    errorEnum corrError;

    cudaPyramidManager.resetPolygon( iSector , r , dr , a , da , cx , cy , as , fittingModel );
    corrError = cudaSolverManager.setCuSolver( iSector , number_of_model_parameters , correlationStream );

    return corrError;
}

errorEnum CudaClass::resetPolygon( v_points blobContour )
{
    errorEnum corrError;
    int iSector = 0;

    cudaPyramidManager.resetPolygon( blobContour , fittingModel );
    corrError = cudaSolverManager.setCuSolver( iSector , number_of_model_parameters , correlationStream );

    return corrError;
}

v_points CudaClass::getUndXY0ToCPU( int iSector )
{
    return cudaPyramidManager.getUndXY0ToCPU( iSector );
}

v_points CudaClass::getDefXY0ToCPU( int iSector )
{
    return cudaPyramidManager.getDefXY0ToCPU( iSector );
}
