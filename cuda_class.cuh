#ifndef CUDA_CLASS_CUH
#define CUDA_CLASS_CUH

#include <opencv2/core/core.hpp>

#include "enums.hpp"
#include "defines.hpp"
#include "model_class.hpp"
#include "cuda_pyramid.cuh"
#include "cuda_solver.cuh"
#include "domains.hpp"

#include <stdio.h>
#include <iostream>
#include <math.h>  //for fmaxf, fminf
#include <assert.h>
#include <string>

//multithread includes
#include <thread>
#include <future>

// Test loading all images at once
#include <queue>

class CudaClass
{
    cudaStream_t                correlationStream = nullptr;
    float                      *pinnedChi , *lastGoodChi ;

    cudaPyramid                 cudaPyramidManager;
    cudaSolver                  cudaSolverManager               { cudaPyramidManager };

    int                         deviceCount                     = 0;
    int                         devicesAvailable                = 0;
    bool                        initialize_devices              = true;
    int                         number_of_colors;

    interpolationModelEnum      interpolationModel;
    fittingModelEnum            fittingModel;

    int                         number_of_model_parameters;

    float                       maximum_iterations;
    float                       required_precision;

public:

    CudaClass();
    ~CudaClass();

    int         initialize                    ( );

    void        set_deviceCount               ( int deviceCount_in );
    void        set_max_iters                 ( int maximum_iterations_ );
    void        set_precision                 ( float required_precision_ );
    void        resetNextPyramid              ( const std::string nxtPath );
    void        set_fitting_model             ( fittingModelEnum fittingModel_in );
    void        set_interpolation_model       ( interpolationModelEnum interpolationModel_in );

    void        resetImagePyramids            ( const std::string undPath ,
                                                const std::string defPath ,
                                                const std::string nxtPath ,
                                                colorEnum color_mode ,
                                                const int start ,
                                                const int step ,
                                                const int stop );
    errorEnum   resetPolygon                  ( int iSector , int x0 , int y0 , int x1 , int y1 );
    errorEnum   resetPolygon                  ( int iSector , float r , float dr , float a , float da ,
                                                float cx , float cy , int as );
    errorEnum   resetPolygon                  ( v_points blobContour );

    void        updatePolygon                  ( int iSector , deformationDescriptionEnum deformationDescription );

    void        makeUndPyramidFromDef         (  );
    void        makeDefPyramidFromNxt         (  );

    CorrelationResult   *correlate            ( int iSector , float  *initial_guess_ , frame_results &results );

    v_points    getUndXY0ToCPU                ( int iSector );
    v_points    getDefXY0ToCPU                ( int iSector );

    std::queue< cv::Mat > tempQ;

private:

    errorEnum   NewtonRaphsonStep             ( int iSector , int pyramidLevel ,
                                                parameterTypeEnum parSrc,
                                                float lambda );
    errorEnum   findNewParameters             ( int iSector , int pyramidLevel,
                                                parameterTypeEnum parSrc,
                                                parameterTypeEnum parDst,
                                                float *chi,
                                                float lambda );
};

#endif // CUDA_CLASS_CUH
