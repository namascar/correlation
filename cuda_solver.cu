#include "cuda_solver.cuh"

cudaSolver::~cudaSolver()
{
    // Destroy cuSolver Handle
    if ( handle )
    {
        cusolverStatus_t cu_err = cusolverDnDestroy( handle );
        if ( cu_err != CUSOLVER_STATUS_SUCCESS )
        {
            printf( "cudaSolver::~cudaSolver Failed to destroy cuSolver handle!\n" );
            exit( EXIT_FAILURE );
        }
    }

    //  Free cuSolver buffer
    if ( buffer ) { cudaFree( buffer ); }
    if ( info   ) { cudaFree( info   ); }
}

errorEnum cudaSolver::setCuSolver( int iSector , int numberOfParameters_ , cudaStream_t correlationStream )
{
    numberOfParameters = numberOfParameters_;
    d_mat = cudaPyramidManager.getGlobalABChi( iSector );
    d_vec = d_mat + numberOfParameters * numberOfParameters;

    // cuSolver synchronizes with the correlation Stream
    cusolverStatus_t cu_err = cusolverDnSetStream( handle, correlationStream );
    if ( cu_err != CUSOLVER_STATUS_SUCCESS )
    {
        printf( "Failed to set cuSolver stream!\n" );
        exit( EXIT_FAILURE );
    }

    //  Allocate cuSolver buffer in GPU[ 0 ]
    cudaError err = cudaSetDevice( 0 );
    if ( err != cudaSuccess )
    {
        printf( "Failed to set device to 0 (error code %s)!\n", cudaGetErrorString( err ) );
        return error_cuSolver;
    }

    int allocatedCusolverBufferSize = bufferSize;

    cu_err = cusolverDnSpotrf_bufferSize( handle,
                                          CUBLAS_FILL_MODE_LOWER,
                                          numberOfParameters,
                                          d_mat,
                                          numberOfParameters,
                                         &bufferSize );

    if ( cu_err != CUSOLVER_STATUS_SUCCESS )
    {
        printf( "Error: Cholesky factorization buffer allocation failed\n");
        return error_cuSolver;
    }

    #if DEBUG_SOLVER_CUDA
            printf( "cudaSolver::setCuSolver Computed cuSolver bufferSize = %d\n", bufferSize );
    #endif

    if ( bufferSize > allocatedCusolverBufferSize )
    {
        cudaFree( buffer );
        err = cudaMalloc( &buffer , sizeof( float ) * bufferSize );
        if ( err != cudaSuccess )
        {
            printf( "Failed to allocate cusolver buffer (error code %s)!\n", cudaGetErrorString( err ) );
            return error_cuSolver;
        }

        #if DEBUG_SOLVER_CUDA
                printf( "cudaSolver::setCuSolver allocating cuSolver bufferSize = %d\n", bufferSize);
        #endif
    }
    return error_none;
}

errorEnum cudaSolver::callCusolver( int iSector , float *chi )
{
    /** The dense matrices are assumed to be stored in column-major order in memory by cuSolver.*/

    cusolverStatus_t status;

    d_mat = cudaPyramidManager.getGlobalABChi( iSector );
    d_vec = d_mat + numberOfParameters * numberOfParameters;

    //  Run cuSolver from GPU#0
    cudaError err = cudaSetDevice( 0 );
    if ( err != cudaSuccess )
    {
        printf( "Failed to set device (error code %s)!\n", cudaGetErrorString( err ) );
        return error_cuSolver;
    }

#if DEBUG_SOLVER_CUDA
    printf("cudaSolver::callCusolver Matrix A and vector b to be solved:\n");

    float *A = new float[ numberOfParameters * numberOfParameters ];
    float *b = new float[ numberOfParameters ];

    cudaMemcpy( A,
                d_mat,
                numberOfParameters * numberOfParameters * sizeof( float ),
                cudaMemcpyDeviceToHost );

    cudaMemcpy( b,
                d_vec,
                numberOfParameters * sizeof( float ),
                cudaMemcpyDeviceToHost );

    for ( int j = 0 ; j < numberOfParameters ; ++j )
    {
        for ( int i = 0 ; i < numberOfParameters ; ++i )
        {
            printf( "%14.4e" , A[ i * numberOfParameters + j ] );
        }
        printf( "      %14.4e\n" , b[ j ] );
    }

    delete[] A;
    delete[] b;

    printf( "cudaSolver::callCusolver bufferSize = %d\n", bufferSize );

#endif

    // Factorization A = L * L H
    status = cusolverDnSpotrf( handle,
                               CUBLAS_FILL_MODE_LOWER,
                               numberOfParameters,
                               d_mat,
                               numberOfParameters,
                               buffer,
                               bufferSize,
                               info );

    if ( status != CUSOLVER_STATUS_SUCCESS )
    {
        printf( "Error: Cholesky factorization failed\n");
        assert( false );
    }

#if DEBUG_SOLVER_CUDA
    printf("cudaSolver::callCusolver Factorized L ( as in A = L * LH) Matrix :\n");

    float *L = new float[ numberOfParameters * numberOfParameters ];
    cudaMemcpy( L,
                d_mat,
                numberOfParameters * numberOfParameters * sizeof( float ),
                cudaMemcpyDeviceToHost );

    for ( int j = 0 ; j < numberOfParameters ; ++j )
    {
        for ( int i = 0 ; i < numberOfParameters ; ++i )
        {
            printf( "%14.4e" , L[ i * numberOfParameters + j ] );
        }
        printf( "\n");
    }

    delete[] L;
#endif


    //  Solver
    status = cusolverDnSpotrs( handle,
                               CUBLAS_FILL_MODE_LOWER,
                               numberOfParameters,
                               1,
                               d_mat,
                               numberOfParameters,
                               d_vec,
                               numberOfParameters,
                               info );

    //cudaDeviceSynchronize();

    if ( status != CUSOLVER_STATUS_SUCCESS )
    {
        printf( "Error: Cholesky solver failed\n");
        assert( false );
    }

    // Save the chi
    cudaMemcpyAsync( chi,
                    &d_mat[ numberOfParameters * ( numberOfParameters + 1 ) ],
                     sizeof( float ),
                     cudaMemcpyDeviceToHost,
                     stream );

#if DEBUG_SOLVER_CUDA
    printf("cudaSolver::callCusolver Update:\n");

    float *temp = new float[ numberOfParameters ];
    cudaMemcpy( temp,
                d_vec,
                numberOfParameters * sizeof( float ) ,
                cudaMemcpyDeviceToHost );

    for ( int i = 0 ; i < numberOfParameters ; ++i )
    {
        printf( "%14.4e" , temp[ i ] );
    }
    printf("\n");
    fflush( stdout );

    delete[] temp;
#endif

    return error_none;
}
