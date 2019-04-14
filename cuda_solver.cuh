#ifndef CUDA_SOLVER_H
#define CUDA_SOLVER_H

#include <stdio.h>
#include <assert.h>

#include "cusolverDn.h"

#include "enums.hpp"
#include "defines.hpp"
#include "cuda_pyramid.cuh"
#include "kernels.cuh"

class cudaSolver
{    
    int                     numberOfParameters      = 0;
    int                     bufferSize              = 0;

    cudaPyramid             &cudaPyramidManager;

    cusolverDnHandle_t      handle                  = nullptr;
    cudaStream_t            stream                  = nullptr;
    float                  *buffer                  = nullptr;
    int                    *info                    = nullptr;

    float                  *d_mat                   = nullptr;
    float                  *d_vec                   = nullptr;

public:
    cudaSolver( cudaPyramid &cudaPyramidManager_ ) : cudaPyramidManager( cudaPyramidManager_ )
    {
        cudaError err;
        // Create cuSolver handle and other cuSolver mem space
        err = cudaMalloc( &info, sizeof( int ) );
        if ( err != cudaSuccess )
        {
            printf( "Failed to allocate cuSolver info (error code %s)!\n" , cudaGetErrorString( err ) );
            exit( EXIT_FAILURE );
        }

        err = cudaMemset( info , 0 , sizeof( int ) );
        if ( err != cudaSuccess )
        {
            printf( "Failed to set cuSolver info to 0 (error code %s)!\n" , cudaGetErrorString( err ) );
            exit( EXIT_FAILURE );
        }

        cusolverStatus_t cu_err = cusolverDnCreate( &handle );
        if ( cu_err != CUSOLVER_STATUS_SUCCESS )
        {
            printf( "Failed to create cuSolver handle!\n" );
            exit( EXIT_FAILURE );
        }

    #if DEBUG_SOLVER_CUDA
         printf( "cudaSolver::cudaSolver\n" );
    #endif
    }
    ~cudaSolver();

    errorEnum               setCuSolver             ( int iSector , int numberOfModelParameters_ , cudaStream_t correlationStream );
    errorEnum               callCusolver            ( int iSector , float *chi );
};

#endif // CUDA_SOLVER_H
