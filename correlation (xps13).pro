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
CONFIG += c++11

HEADERS       =imageLabel.h \
                CImg.h \
                subdi.h \
                manager_class.h \
    mainapp.h \
    model_class.hpp \
    interpolation_class.hpp \
    correlation_class.hpp \
    parameters.hpp
SOURCES       =main.cpp \
                imageLabel.cpp \
                subdi.cpp \
                manager_class.cpp \
    mainapp.cpp \
    interpolation_class.cpp \
    correlation_class.cpp \
    model_class.cpp \
    parameters.cpp

#LIBS += -L"/usr/include/ImageMagick" -lMagick++ -lMagickCore -fopenmp
LIBS += -L"-L/usr/X11R6/lib" -lX11 -lXrandr
LIBS += -L"/usr/include/opencv" -lopencv_highgui -lopencv_core -lopencv_imgproc
INCLUDEPATH += "/usr/include/eigen"
INCLUDEPATH += "/usr/include/opencv2/core"
INCLUDEPATH += "/usr/include/opencv2/highgui"

# install
target.path = $$[QT_INSTALL_EXAMPLES]/widgets/widgets/imageviewer
INSTALLS += target


wince*: {
   DEPLOYMENT_PLUGIN += qjpeg qgif
}

FORMS += \
    mainapp.ui
