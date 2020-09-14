#ifndef CUDA_PYRAMID_CUH
#define CUDA_PYRAMID_CUH

#include "cuda_polygon.cuh"
#include "model_class.hpp"

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Local includes
//#include "kernels.cuh"
#include "defines.hpp"
#include "domains.hpp"
#include "enums.hpp"

// Utility includes
#include <assert.h>
#include <map>
#include <vector>

// Multithread includes
#include <future>
#include <thread>

#include <nvToolsExt.h> // Marker for the nvvp profiler

class cudaImage {
  int start{0};
  int step{0};
  int stop{0};

  ImageType imageType;
  cudaStream_t stream;

  std::vector<cudaArray *> d_images_cuArray;
  std::vector<cudaTextureObject_t> textures;
  std::vector<cudaSurfaceObject_t> surfaces;
  int rowsLevel0{0};
  int colsLevel0{0};
  int number_of_colors{0};

  void makePyramid(const cv::Mat &image_in);
  void uploadImage0(const cv::Mat &image_in);
  void allocateImages(const cv::Mat &image_in);
  void createAndBindTextures();
  void createAndBindSurfaces();
  void deleteSurfaces();
  void deleteTextures();
  void deleteCudaArrays();

public:
  cudaImage(const cv::Mat &image_, int start_, int step_, int stop_,
            ImageType imageType_, cudaStream_t stream_)
      : start(start_), step(step_), stop(stop_), imageType(imageType_),
        stream(stream_) {
    // Put a marker on the nvvp CUDA profiler
    nvtxRangePushA("cudaImage::cudaImage");

    assert(image_.isContinuous());

    allocateImages(image_);
    createAndBindSurfaces();
    createAndBindTextures();
    uploadImage0(image_);

    makePyramid(image_);

    nvtxRangePop();
  }

  ~cudaImage() {
    deleteSurfaces();
    deleteTextures();
    deleteCudaArrays();
  }

  void setStart(int Start_);
  void setStep(int Step_);
  void setStop(int Stop_);
  void testPyramid();
  void setImageType(ImageType imageType_);
  void recycleImage(const cv::Mat &image_in, ImageType imageType_);
  cudaTextureObject_t getTexture(int level_);
};

class cudaPyramid {
  /**
    cudaPyramid contains all the data that is to be uploaded to the GPU
   */

  int start{0};
  int step{0};
  int stop{0};

  cudaImage *undImage{nullptr};
  cudaImage *defImage{nullptr};
  cudaImage *nxtImage{nullptr};

  cudaStream_t undStream;
  cudaStream_t defStream;
  cudaStream_t nxtStream;

  std::map<int, cudaPolygon *> polygons; // Each polygon object contains the
                                         // fitting model, und_x and y, und
                                         // centers, def_x and y, DdilDe.

  cudaImage *getNewPyramid(const cv::Mat CvImage, ImageType imageType_,
                           cudaStream_t stream_);
  void createImageStreams();
  void destroyImageStreams();
  void deleteImages();
  void deletePolygons();

public:
  cudaPyramid() { createImageStreams(); }

  ~cudaPyramid() {
    deleteImages();
    destroyImageStreams();
    deletePolygons();
#if DEBUG_CUDA_POLYGON
    printf("Deleting cudaPyramid\n");
#endif
  }

  void resetImagePyramids(const cv::Mat undCvImage, const cv::Mat defCvImage,
                          const cv::Mat nxtCvImage, const int start_,
                          const int step_, const int stop_);

  void newNxtPyramid(const cv::Mat nxtCvImage);
  void makeUndPyramidFromDef();
  void makeDefPyramidFromNxt();

  cudaTextureObject_t getUndTexture(int level_);
  cudaTextureObject_t getDefTexture(int level_);

  int getPyramidStart();
  int getPyramidStep();
  int getPyramidStop();

  // Methods to make the polygons
  void resetPolygon(int iSector, int x0, int y0, int x1, int y1,
                    fittingModelEnum fittingModel_);

  void resetPolygon(int iSector, float r, float dr, float a, float da, float cx,
                    float cy, int as, fittingModelEnum fittingModel_);

  void resetPolygon(v_points blobContour, fittingModelEnum fittingModel_);

  // Accessor methods for the polygons
  v_points getUndXY0ToCPU(int iSector);
  v_points getDefXY0ToCPU(int iSector);
  int getNumberOfPoints(int iSector, int level_);
  float *getUndXPtr(int iSector, int level_);
  float *getUndYPtr(int iSector, int level_);
  float *getUndCenter(int iSector, int level_);
  float *getGlobalABChi(int iSector);
  float *getParameters(int iSector, parameterTypeEnum parSrc);
  void initializeParametersLevel0(int iSector, float *initialGuess_);
  void transferParameters(int iSector, parameterTypeEnum parSrc,
                          parameterTypeEnum parDst);

  void updateParameters(int iSector, int numberOfModelParameters,
                        parameterTypeEnum parSrc, parameterTypeEnum parDst,
                        cudaStream_t stream);

  void scaleParametersForLevel(int iSector, int level_);
  CorrelationResult *getCorrelationResultsToCPU(int iSector);
  void updatePolygon(int iSector,
                     deformationDescriptionEnum deformationDescription);
};

#endif // CUDA_PYRAMID_CUH
