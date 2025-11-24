# Image Correlation

This application perform digital image correlation based on the optical flow method by Lucas and Kanade. A small video 
that illustrates how this program works is at https://youtu.be/Ak4puzdqWDM

![gui](/gui.png)

# What is this repository for?

It consists of a Qt GUI that calls a multi-threading computational engine, running on the CPU and a CUDA engine to perform the computations in the GPU.

A set of images can be selected in the GUI, which will display the first two images of the set. Selection of the domain to be correlated can be done using rectangular domains, annular domains and "blob" domain, which are defined by a continuous trace of the mouse over the image. Domains can also be adjusted numerically and via mouse clicks.

Initial guess for the correlation algorithm can be automatic, null or user-selected. In the last mode, the user can adjust the position of the domain in the first deformed image via mouse clicks. After the initial guess is defined for the first two images, the motion is assumed to be at constant velocity, so past deformations are extrapolated to produce the next initial guesses.

Reports can be saved with the resulting correlation parameters, number of iterations and chi.

Check out the video for an example (correlation_class/video_instructions.mp4)

# Setup

The project is a Windows, Visual Studio 2022. It uses Qt 6.8.1, OpenCV 4.10, CUDA 12.6, and Eigen 3.4.0.

Define following environmental variables:

1. QTDIR = "C:\Qt\6.8.1\msvc2022_64"
2. OPENCV_DIR = "C:\opencv4100Cuda126\x64\vc17"
3. CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
4. EIGEN_DIR = "C:\eigen-3.4.0"

## Configuration

File defines.h allows for defining the number of thread used in the CPU, which I currnetly have at 20. When I run the code in my laptop (smaller chip) I change that to 4-8. Also in "defines.h" one can choose if you want to use CUDA and the GPU. Since I don't have a CUDA compatible laptop, I set CUDA_ENABLED to false. The maximum number of GPUs is set to 32. I just use that variable to statically define a small array for the cuda class, so there is not that much down side. The code detects the number of available GPUs. At the moment, since the implementation of the GPU image pyramids, I set the number of GPUs = 1 always.

I use -std=c++17, -fopenmp and -use_fast_math in anticipation of using some of the abbreviated function in the future.

# Who do I talk to?

Send me an email at namascar@gmail.com with comments/suggestions.


