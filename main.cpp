#include <QApplication>

#include "imageLabel.h"
#include "mainapp.h"

int main(int argc, char *argv[]) 
{
	QApplication app(argc, argv);

	MainApp gui;
	gui.show();

	return app.exec();
}
