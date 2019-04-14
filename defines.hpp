#ifndef DEFINES_HPP
#define DEFINES_HPP

//----------------------------------------------------------------------
//
//   constants - parameters
//
//----------------------------------------------------------------------

#define NUMBER_OF_THREADS 20
#define MAX_GPU_COUNT 32
#define THREADS_PER_BLOCK 256

//----------------------------------------------------------------------
//
//   Operational flags
//
//----------------------------------------------------------------------

#define CUDA_ENABLED true
#define AUTO_PILOT false

//----------------------------------------------------------------------
//
//   Debugging flags
//
//----------------------------------------------------------------------

#define DEBUG_CORRELATION false
#define DEBUG_CORRELATION_PLOTS false
#define DEBUG_PYRAMID false
#define DEBUG_PYRAMID_IMAGES false
#define DEBUG_CUDA false
#define DEBUG_CUDA_KERNEL false
#define DEBUG_CUDA_POINTS false
#define DEBUG_CUDA_PYRAMID false
#define DEBUG_CUDA_POLYGON false
#define DEBUG_CUDA_POLYGON_POINTS false
#define DEBUG_SOLVER_CUDA false
#define DEBUG_CORRELATION_INFO false
#define DEBUG_SOLVER false
#define DEBUG_NEWTON_RAPHSON false  // print iteration info
#define DEBUG_NEWTON_RAPHSON_FAIL_DUMP false
#define DEBUG_NEWTON_RAPHSON_CUDA false
#define DEBUG_MODEL false
#define DEBUG_MODEL_INPUTS false
#define DEBUG_INTERPOLATION false

#define DEBUG_INTERPOLATION_MAT false
#define DEBUG_MAT_A false
#define DEBUG_THREAD false
#define DEBUG_MANAGER_POINTS false
#define DEBUG_MANAGER_INSIDE_POINTS false
#define DEBUG_DOMAIN_SELECTION false
#define DEBUG_CENTER false

#define DEBUG_TIME_MODEL_AND_INTERPOLATION false
#define DEBUG_TIME_NEWTON_RAPHSON false
#define DEBUG_TIME_CHI false
#define DEBUG_TIME_NEW_PARAMETERS false
#define DEBUG_TIME_NEW_PARAMETERS_ASSEMBLY false
#define DEBUG_TIME_NEW_PARAMETERS_SOLVER false
#define DEBUG_TIME_INTERPOLATION_PARAMETERS_ALLOCATION false
#define DEBUG_TIME_INTERPOLATION_PARAMETERS_RESET false
#define DEBUG_TIME_FRAME_CORRELATION false
#define DEBUG_TIME_MULTI_FRAME_CORRELATION true
#define DEBUG_TIME_POINT_SELECTION false
#define DEBUG_TIME_ALL_POINT_SELECTION false
#define DEBUG_TIME_ALL_MODEL false
#define DEBUG_TIME_ALL_INTERPOLATION false
#define DEBUG_TIME_ALL_NEW_PARAMETERS_ASSEMBLY false
#define DEBUG_TIME_ALL_NEW_PARAMETERS_SOLVER false

#define NVVP_PROFILE_CPU true

#endif // DEFINES_HPP
