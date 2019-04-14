README

This application perform digital image correlation based on the optical flow method by Lucas and Kanade.

What is this repository for?

It consists of a Qt GUI that calls a multi-threading computational engine, running on the CPU and a CUDA engine to perform the computations in the GPU.

A set of images can be selected in the GUI, which will display the first two images of the set. Selection of the domain to be correlated can be done using rectangular domains, annular domains and "blob" domain, which are defined by a continuous trace of the mouse over the image. Domains can also be adjusted numerically and via mouse clicks.

Initial guess for the correlation algorithm can be automatic, null or user-selected. In the last mode, the user can adjust the position of the domain in the first deformed image via mouse clicks. After the initial guess is defined for the first two images, the motion is assumed to be at constant velocity, so past deformations are extrapolated to produce the nest initial guesses.

Reports can be saved with the resulting correlation parameters, number of iterations and chi.

Check out the video for an example (correlation_class/video_instructions.mp4)

How do I get set up?

Summary of set up

The project is currenly set up as a Qt Creator project in Ubuntu 16.04. It uses the openCV libraries, CUDA, openmp, pthreads and Eigen. Of course also uses Qt.

Configuration

File defines.h allows for defining the number of thread used in the CPU, which I currnetly have at 20. When I run the code in my laptop (smaller chip) I change that to 4-8. Also in "defines.h" one can choose if you want to use CUDA and the GPU. Since I don't have a CUDA compatible laptop, I set CUDA_ENABLED to false. The maximum number of GPUs is set to the unrealistic number of 32. I just use that variable to statically define a small array for the cuda class, so there is not that much down side. The code detects the number of available GPUs

I use -std=c++11, -fopenmp and -use_fast_math in anticipation of using some of the abreviated function in the future.

Dependencies

Project needs to link to

openCV: I use version 3.3. Other earlier version will probably work too, since I am not using functionality other than the very basic, such us opening images. Used libraries are -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

CUDA: I used a somehow more complex version of CUDA than what I need at the moment, in anticipation of playing around with the dynamic parallelism. For this I do separate compilation with relocatable device code, which requires the separate compilation ( and CUDA linking with -lcudadevrt ). Other than that, I use -lcudart -lcuda -lcusolver -lcublas. I use the CUDA version 8, with the 8.61.2 patch. That is the version I use for other things like tersorflow and openCV, so I still resist going to CUDA 9 until openCV is compatible with it.

Eigen: This is a easy one to setup. Just needs the header files.

Qt: I use core, gui, widgets and printsupport

openmp: parallelization of some loops.

I also link tot he nvToolsExt from nVidia to make CPU side debug traces in the nvvp profiler.

Who do I talk to?

Send me an email at namascar@gmail.com with comments/suggestions


