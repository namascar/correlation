#-------------------------------------------------
#
# Project created by QtCreator 2016-02-22T23:12:58
#
#-------------------------------------------------

QT += core gui
QT += widgets
QT += printsupport

TARGET = correlation
TEMPLATE = app
CONFIG += c++17

HEADERS       = \
                imageLabel.h \
                subdi.h \
                manager_class.h \
                mainapp.h \
                model_class.hpp \
                interpolation_class.hpp \
                correlation_class.hpp \
                parameters.hpp \
                enums.hpp \
                defines.hpp \
                cuda_class.cuh \
                kernels.cuh \
                pyramid_class.h \
                polygon_class.h \
                cuda_pyramid.cuh \
                cuda_polygon.cuh \
                correlationKernel.cuh \
                domains.hpp \
                cuda_solver.cuh

SOURCES       = \
                main.cpp \
                imageLabel.cpp \
                subdi.cpp \
                mainapp.cpp \
                interpolation_class.cpp \
                correlation_class.cpp \
                model_class.cpp \
                manager_class.cpp \
                parameters.cpp \
                pyramid_class.cpp \
                polygon_class.cpp

CUDA_SOURCES  += cuda_class.cu\
                 kernels.cu\
                 cuda_pyramid.cu\
                 cuda_polygon.cu\
                 correlationKernel.cu\
                 cuda_solver.cu

LIBS += -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lnvToolsExt
#eigen requires only to include the header files, not libs
INCLUDEPATH += "/usr/local/include/Eigen"
INCLUDEPATH += "/usr/local/include/opencv4"

QMAKE_LFLAGS += -fopenmp
QMAKE_CXXFLAGS += -fopenmp

# CUDA from https://cudaspace.wordpress.com/2012/07/05/qt-creator-cuda-linux-review/
# Path to cuda toolkit install
CUDA_DIR      = /usr/local/cuda
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
# libs used in my code
LIBS += -lcudart -lcuda -lcusolver -lcublas -lcudadevrt
CUDA_LIBS += -lcudart -lcuda -lcusolver -lcublas -lcudadevrt
# GPU architecture
CUDA_ARCH     = sm_61
# Here are some NVCC flags I've always used by default.
# omp from https://stackoverflow.com/questions/12289387/cuda-combined-with-openmp
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math -std=c++17 -Xcompiler -fopenmp -rdc=true --ptxas-options=-v

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = -I$$CUDA_DIR/include
CUDA_INC += "-I/usr/local/include/opencv4"

# CUDA COMPILATION done in two steps to allow Dynamic Parallelism

# CUDA compile - step 1
# per http://forums.nvidia.com/index.php?showtopic=171651
#     https://wiki.qt.io/Undocumented_QMake
cuda_compile.commands =  $$CUDA_DIR/bin/nvcc -ccbin g++ -O3 -m64 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                         $$CUDA_INC $$CUDA_LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}\
                         2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
# nvcc error printout format ever so slightly different from gcc

cuda_compile.dependency_type = TYPE_C
#cuda_compile.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

cuda_compile.input = CUDA_SOURCES
cuda_compile.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

#Set our variable out. These obj files need to be used to create the link obj file
#and used in our final gcc compilation
cuda_compile.variable_out = CUDA_OBJ
cuda_compile.variable_out += OBJECTS

# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda_compile

# CUDA linker - step 2
# per https://declanrussell.com/2015/04/16/compiling-cuda-dynamic-parallelism-with-qt-creator/
cuda_link.input = CUDA_OBJ
cuda_link.output = ${QMAKE_FILE_BASE}_link.o

cuda_link.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -arch=$$CUDA_ARCH -dlink ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda_link.dependency_type = TYPE_C
#cuda_link.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda_link

# install
target.path = $$[QT_INSTALL_EXAMPLES]/widgets/widgets/imageviewer
INSTALLS += target


wince*: {
   DEPLOYMENT_PLUGIN += qjpeg qgif
}

FORMS += \
    mainapp.ui

DISTFILES += \
    cuda_class.cu \
    kernels.cu \
    cuda_pyramid.cu \
    cuda_polygon.cu \
    correlationKernel.cu \
    cuda_solver.cu
