#include "cuda_pyramid.cuh"

void cudaImage::makePyramid(const cv::Mat &image_in) {
  cudaError_t err = cudaSuccess;

  // Make all pyramids (level > 1) in the GPU
  for (int ilevel = 1; ilevel < stop + 1; ++ilevel) {

#if DEBUG_CUDA_PYRAMID
    printf("cudaImage::makePyramid: Making level %d pyramid of type %d\n",
           ilevel, imageType);
#endif

    int dstRows = rowsLevel0 / (1 << ilevel);
    int dstCols = colsLevel0 / (1 << ilevel);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((dstCols * number_of_colors + threadsPerBlock.x - 1) /
                           threadsPerBlock.x,
                       (dstRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    switch (number_of_colors) {
    case 1:
      k_pyramid_bw<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
          textures[ilevel - 1], surfaces[ilevel], dstRows, dstCols);
      break;

    case 3:
      k_pyramid_color<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
          textures[ilevel - 1], surfaces[ilevel], dstRows, dstCols);
      break;

    default:
      assert(false);
      break;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Failed to launch pyramid kernel (error code %s)!\n",
             cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
}

void cudaImage::allocateImages(const cv::Mat &image_in) {
  cudaError_t err = cudaSuccess;

  // Define texture channel descriptor as monochrome
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();

  rowsLevel0 = image_in.rows;
  colsLevel0 = image_in.cols;
  number_of_colors = image_in.channels();

  deleteCudaArrays();

  //  Allocate cudaArrays on GPU0
  d_images_cuArray = std::vector<cudaArray *>(stop + 1, nullptr);
  for (int ilevel = 0; ilevel < stop + 1; ++ilevel) {
    int rows = rowsLevel0 / (1 << ilevel);
    int cols = colsLevel0 / (1 << ilevel);

    err = cudaMallocArray(&d_images_cuArray[ilevel], &channelDesc,
                          cols * number_of_colors, rows);

    if (err != cudaSuccess) {
      printf("Failed to allocate memory for d_images_cuArray at level %d(error "
             "code %s)!\n",
             ilevel, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

#if DEBUG_CUDA_PYRAMID
    printf("cuda_pyramid: allocateImages: completed cuArray allocation on "
           "level %d pyramid type %d\n",
           ilevel, imageType);
#endif
  }
}

void cudaImage::uploadImage0(const cv::Mat &image_in) {

#if DEBUG_CUDA_PYRAMID
  printf("cudaImage::uploadImage0: Uploading level 0 pyramid of type %d\n",
         imageType);
#endif

  cudaError_t err = cudaSuccess;
  err = cudaMemcpy2DToArrayAsync(d_images_cuArray[0], 0, 0, image_in.data,
             colsLevel0 * number_of_colors * sizeof(unsigned char),
             colsLevel0 * number_of_colors * sizeof(unsigned char),
             rowsLevel0,
             cudaMemcpyHostToDevice, stream);

  if (err != cudaSuccess) {
    printf(
        "Failed to copy image_in to d_images_cuArray[ 0 ] (error code: %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void cudaImage::createAndBindTextures() {
  cudaError_t err = cudaSuccess;

  // Destroy previous texture objects
  deleteTextures();

  // Create new Textures on GPU0 - Textures are used to read cudaArrays
  textures = std::vector<cudaTextureObject_t>(d_images_cuArray.size(), 0);
  for (int ilevel = 0; ilevel < d_images_cuArray.size(); ++ilevel) {
    // Specify Texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_images_cuArray[ilevel];

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    //  Create texture object and bind it to the cudaArray
    err =
        cudaCreateTextureObject(&textures[ilevel], &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
      printf("Failed to bind textures to d_image_cuArray (error code %s)!\n",
             cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

#if DEBUG_CUDA_PYRAMID
    printf("cudaImage::createAndBindTextures: Binding texture of level %d "
           "pyramid of type %d\n",
           ilevel, imageType);
#endif
  }
}

void cudaImage::createAndBindSurfaces() {
  cudaError_t err = cudaSuccess;

  // Destroy previous surface objects
  deleteSurfaces();

  //  Create new surface objects on GPU0 - Surfaces are used to write to
  //  cudaArray,
  //      so don't create the n = 0.
  surfaces = std::vector<cudaSurfaceObject_t>(d_images_cuArray.size(), 0);
  for (int ilevel = 1; ilevel < d_images_cuArray.size(); ++ilevel) {
    // Specify Surface
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_images_cuArray[ilevel];

    //  Create surface object and bind it to the cudaArray
    err = cudaCreateSurfaceObject(&surfaces[ilevel], &resDesc);
    if (err != cudaSuccess) {
      printf("Failed to bind surfaces to d_image_cuArray (error code %s)!\n",
             cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

#if DEBUG_CUDA_PYRAMID
    printf("cudaImage::createAndBindSurfaces: Binding surface of level %d "
           "pyramid of type %d\n",
           ilevel, imageType);
#endif
  }
}

void cudaImage::deleteCudaArrays() {
  cudaError_t err = cudaSuccess;

  int iGPU = 0;

  // Set GPU0 as the active one
  err = cudaSetDevice(iGPU);
  if (err != cudaSuccess) {
    printf("Failed to set device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

#if DEBUG_CUDA_PYRAMID
  int ilevel = 0;
#endif

  //  Delete cudaArrays if they exist
  for (cudaArray *thisCudaArray : d_images_cuArray) {

#if DEBUG_CUDA_PYRAMID
    printf(" Deleting cudaArray of level %d pyramid of type %d\n", ilevel++,
           imageType);
#endif
    if (thisCudaArray) {
      err = cudaFreeArray(thisCudaArray);
      thisCudaArray = nullptr;
    }

    if (err != cudaSuccess) {
      printf("Failed to free device d_image_cuArray (error code %s)!\n",
             cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  d_images_cuArray.clear();
}

void cudaImage::recycleImage(const cv::Mat &image_in, ImageType imageType_) {
  // Put a marker on the nvvp CUDA profiler
  nvtxRangePushA("cudaImage::recycleImage");

  imageType = imageType_;

  uploadImage0(image_in);
  makePyramid(image_in);

  nvtxRangePop();
}

void cudaImage::deleteSurfaces() {
  cudaError_t err = cudaSuccess;

  int iGPU = 0;

  // Set GPU0 as the active one
  err = cudaSetDevice(iGPU);
  if (err != cudaSuccess) {
    printf("Failed to set device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

#if DEBUG_CUDA_PYRAMID
  int ilevel = 1;
#endif

  //  Delete csurfaces if they exist
  for (cudaSurfaceObject_t thisSurface : surfaces) {

#if DEBUG_CUDA_PYRAMID
    printf(" Deleting surfaces of level %d pyramid of type %d\n", ilevel++,
           imageType);
#endif
    if (thisSurface) {
      err = cudaDestroySurfaceObject(thisSurface);
    }

    if (err != cudaSuccess) {
      printf("Failed to destroy surfaces surface object (error code %s)!\n",
             cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  surfaces.clear();
}

void cudaImage::deleteTextures() {
  cudaError_t err = cudaSuccess;

  int iGPU = 0;

  // Set GPU0 as the active one
  err = cudaSetDevice(iGPU);
  if (err != cudaSuccess) {
    printf("Failed to set device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

#if DEBUG_CUDA_PYRAMID
  int ilevel = 0;
#endif

  //  Delete csurfaces if they exist
  for (cudaTextureObject_t thisTexture : textures) {

#if DEBUG_CUDA_PYRAMID
    printf(" Deleting textures of level %d pyramid of type %d\n", ilevel++,
           imageType);
#endif

    if (thisTexture) {
      err = cudaDestroyTextureObject(thisTexture);
    }

    if (err != cudaSuccess) {
      printf("Failed to destroy textures texture object (error code %s)!\n",
             cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  textures.clear();
}

void cudaImage::testPyramid() {
  std::vector<cv::Mat> images(d_images_cuArray.size());

  int type;
  switch (number_of_colors) {
  case 1:
    type = CV_8U;
    break;
  case 3:
    type = CV_8UC3;
    break;
  default:
    assert(false);
    break;
  }

  for (int ilevel = 0; ilevel < d_images_cuArray.size(); ++ilevel) {
    cudaError_t err = cudaSuccess;

    std::string name = "test" + std::to_string(ilevel);

    int rows = rowsLevel0 / (1 << ilevel);
    int cols = colsLevel0 / (1 << ilevel);

    images[ilevel] = cv::Mat(rows, cols, type);

    err = cudaMemcpy2DFromArray(
                              images[ilevel].data,
                              cols * number_of_colors * sizeof(unsigned char),
                              d_images_cuArray[ilevel], 0, 0,
                              cols * number_of_colors * sizeof(unsigned char),
                              rows,
                              cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
      printf("Failed to copy images to host (error code %s)!\n",
             cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // cv::imshow( name , images[ ilevel ] );
  }
}

cudaTextureObject_t cudaImage::getTexture(int level_) {
  return textures[level_];
}

cudaTextureObject_t cudaPyramid::getUndTexture(int level_) {
  return undImage->getTexture(level_);
}

cudaTextureObject_t cudaPyramid::getDefTexture(int level_) {
  return defImage->getTexture(level_);
}

float *cudaPyramid::getUndXPtr(int iSector, int level_) {
  return polygons[iSector]->getUndXPtr(level_);
}

float *cudaPyramid::getUndYPtr(int iSector, int level_) {
  return polygons[iSector]->getUndYPtr(level_);
}

float *cudaPyramid::getUndCenter(int iSector, int level_) {
  return polygons[iSector]->getUndCenter(level_);
}

float *cudaPyramid::getGlobalABChi(int iSector) {
  return polygons[iSector]->getGlobalABChi();
}

float *cudaPyramid::getParameters(int iSector, parameterTypeEnum parSrc) {
  return polygons[iSector]->getParameters(parSrc);
}

void cudaPyramid::updateParameters(int iSector, int numberOfModelParameters,
                                   parameterTypeEnum parSrc,
                                   parameterTypeEnum parDst,
                                   cudaStream_t stream) {
  polygons[iSector]->updateParameters(numberOfModelParameters, parSrc, parDst,
                                      stream);
}

void cudaPyramid::scaleParametersForLevel(int iSector, int level_) {
  polygons[iSector]->scaleParametersForLevel(level_);
}

void cudaPyramid::initializeParametersLevel0(int iSector,
                                             float *initialGuess_) {
#if DEBUG_CUDA_PYRAMID
  printf("cudaPyramid::initializeParametersLevel0 \n");
#endif

  polygons[iSector]->initializeParametersLevel0(initialGuess_);
}

void cudaPyramid::transferParameters(int iSector, parameterTypeEnum parSrc,
                                     parameterTypeEnum parDst) {
#if DEBUG_CUDA_PYRAMID
  printf("cudaPyramid::transferParameters()\n");
#endif

  polygons[iSector]->transferParameters(parSrc, parDst);
}

v_points cudaPyramid::getUndXY0ToCPU(int iSector) {
  return polygons[iSector]->getUndXY0ToCPU();
}

v_points cudaPyramid::getDefXY0ToCPU(int iSector) {
  return polygons[iSector]->getDefXY0ToCPU();
}

CorrelationResult *cudaPyramid::getCorrelationResultsToCPU(int iSector) {
  return polygons[iSector]->getCorrelationResultsToCPU();
}

int cudaPyramid::getNumberOfPoints(int iSector, int level) {
  return polygons[iSector]->getNumberOfPoints(level);
}

void cudaPyramid::deleteImages() {
  if (undImage) {
#if DEBUG_CUDA_PYRAMID
    printf("cudaPyramid::deleteImages() undImage\n");
    fflush(stdout);
#endif
    delete undImage;
    undImage = nullptr;
  }

  if (defImage) {
#if DEBUG_CUDA_PYRAMID
    printf("cudaPyramid::deleteImages() defImage\n");
    fflush(stdout);
#endif
    delete defImage;
    defImage = nullptr;
  }

  if (nxtImage) {
#if DEBUG_CUDA_PYRAMID
    printf("cudaPyramid::deleteImages() nxtImage\n");
    fflush(stdout);
#endif
    delete nxtImage;
    nxtImage = nullptr;
  }
#if DEBUG_CUDA_PYRAMID
  printf("cudaPyramid::deleteImages() done\n");
  fflush(stdout);
#endif
}

void cudaPyramid::resetImagePyramids(const cv::Mat undCvImage,
                                     const cv::Mat defCvImage,
                                     const cv::Mat nxtCvImage, const int start_,
                                     const int step_, const int stop_) {
  deleteImages();

  start = start_;
  step = step_;
  stop = stop_;

  //    std::future< cudaImage* > undFuture , defFuture , nxtFuture;

  //    undFuture = std::async( std::launch::async , &cudaPyramid::getNewPyramid
  //    , this , undCvImage , imageType_und , undStream );
  //    defFuture = std::async( std::launch::async , &cudaPyramid::getNewPyramid
  //    , this , defCvImage , imageType_def , defStream );
  //    if ( !nxtCvImage.empty() )
  //    {
  //        nxtFuture = std::async( std::launch::async ,
  //        &cudaPyramid::getNewPyramid , this , nxtCvImage , imageType_nxt ,
  //        nxtStream );
  //    }

  //    undImage = undFuture.get();
  //    defImage = defFuture.get();
  //    if ( !nxtCvImage.empty() )
  //    {
  //        nxtImage = nxtFuture.get();
  //    }

  undImage = getNewPyramid(undCvImage, imageType_und, undStream);
  defImage = getNewPyramid(defCvImage, imageType_def, defStream);
  if (!nxtCvImage.empty()) {
    nxtImage = getNewPyramid(nxtCvImage, imageType_nxt, nxtStream);
  }

  // undImage->testPyramid( );
  // cv::waitKey( 1000 );
  // defImage->testPyramid( );
}

void cudaPyramid::newNxtPyramid(const cv::Mat nxtCvImage) {
  if (nxtImage) {
    nxtImage->recycleImage(nxtCvImage, imageType_nxt);
  } else {
    nxtImage = getNewPyramid(nxtCvImage, imageType_nxt, nxtStream);
  }
}

void cudaPyramid::makeUndPyramidFromDef() {
#if DEBUG_CUDA_PYRAMID
  printf("cudaPyramid::makeUndPyramidFromDef\n");
#endif

  cudaImage *temp = undImage;
  undImage = defImage;
  undImage->setImageType(imageType_und);
  defImage = temp;
  defImage->setImageType(imageType_def);
}

void cudaPyramid::makeDefPyramidFromNxt() {
#if DEBUG_CUDA_PYRAMID
  printf("cudaPyramid::makeDefPyramidFromNxt\n");
#endif

  cudaImage *temp = defImage;
  defImage = nxtImage;
  defImage->setImageType(imageType_def);
  nxtImage = temp;
  nxtImage->setImageType(imageType_nxt);
}

void cudaImage::setImageType(ImageType imageType_) { imageType = imageType_; }

cudaImage *cudaPyramid::getNewPyramid(const cv::Mat CvImage,
                                      ImageType imageType_,
                                      cudaStream_t stream_) {
  return new cudaImage{CvImage, start, step, stop, imageType_, stream_};
}

void cudaPyramid::createImageStreams() {
  cudaError_t err = cudaSuccess;

  err = cudaStreamCreate(&undStream);
  if (err != cudaSuccess) {
    printf("Failed to create undStream (error code %s)!\n",
           cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaStreamCreate(&defStream);
  if (err != cudaSuccess) {
    printf("Failed to create defStream (error code %s)!\n",
           cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaStreamCreate(&nxtStream);
  if (err != cudaSuccess) {
    printf("Failed to create nextStream (error code %s)!\n",
           cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
void cudaPyramid::destroyImageStreams() {
  cudaError_t err = cudaSuccess;

  err = cudaStreamDestroy(undStream);
  if (err != cudaSuccess) {
    printf("Failed to Destroy undStream (error code %s)!\n",
           cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaStreamDestroy(defStream);
  if (err != cudaSuccess) {
    printf("Failed to Destroy defStream (error code %s)!\n",
           cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaStreamDestroy(nxtStream);
  if (err != cudaSuccess) {
    printf("Failed to Destroy nextStream (error code %s)!\n",
           cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void cudaPyramid::resetPolygon(int iSector, int x0, int y0, int x1, int y1,
                               fittingModelEnum fittingModel_) {
  polygons[iSector] = new cudaPolygonRectangular(iSector, x0, y0, x1, y1, start,
                                                 step, stop, fittingModel_);
}

void cudaPyramid::resetPolygon(int iSector, float r, float dr, float a,
                               float da, float cx, float cy, int as,
                               fittingModelEnum fittingModel_) {
  polygons[iSector] = new cudaPolygonAnnular(iSector, r, dr, a, da, cx, cy, as,
                                             start, step, stop, fittingModel_);
}

void cudaPyramid::resetPolygon(v_points blobContour,
                               fittingModelEnum fittingModel_) {
  polygons[0] =
      new cudaPolygonBlob(blobContour, start, step, stop, fittingModel_);
}
void cudaPyramid::updatePolygon(
    int iSector, deformationDescriptionEnum deformationDescription) {
  polygons[iSector]->updatePolygon(deformationDescription);
}

int cudaPyramid::getPyramidStart() { return start; }

int cudaPyramid::getPyramidStep() { return step; }

int cudaPyramid::getPyramidStop() { return stop; }

void cudaPyramid::deletePolygons() {
  std::map<int, cudaPolygon *>::iterator it;

  for (it = polygons.begin(); it != polygons.end(); ++it) {
    delete it->second;
  }
}
