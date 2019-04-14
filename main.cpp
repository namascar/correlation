/****************************************************************************
**
**  This software was developed by Javier Gonzalez on 2018
**
**  Based on the Qt examples
**
**
****************************************************************************/

#include <QApplication>

#include "imageLabel.h"
#include "mainapp.h"

int main( int argc, char* argv[] )
{
    QApplication app( argc , argv );

    MainApp gui;
    gui.show();

    return app.exec();
}
