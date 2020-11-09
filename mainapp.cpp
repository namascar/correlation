#include "mainapp.h"

MainApp::MainApp(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainApp) {
  ui->setupUi(this);

  scale_x = 1;
  scale_y = 1;

  // Declare the inside points before connect declarations. Needed for queued
  // signal-slot
  // that are in different threads
  qRegisterMetaType<v_points>();

  // Create main windows where the two current images are displayed
  und_imageLabel = new ImageLabel(this);
  und_imageLabel->setBackgroundRole(QPalette::Light);
  und_imageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  und_imageLabel->setScaledContents(true);
  und_imageLabel->mirroring = false;
  und_imageLabel->installEventFilter(this);
  und_imageLabel->realScale_x = scale_x;
  und_imageLabel->realScale_y = scale_y;
  und_imageLabel->clickScale_x = scale_x;
  und_imageLabel->clickScale_y = scale_y;
  und_imageLabel->name = "und_imageLabel";

  def_imageLabel = new ImageLabel(this);
  def_imageLabel->setBackgroundRole(QPalette::Light);
  def_imageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  def_imageLabel->setScaledContents(true);
  def_imageLabel->mirroring = true;
  def_imageLabel->installEventFilter(this);
  def_imageLabel->realScale_x = scale_x;
  def_imageLabel->realScale_y = scale_y;
  def_imageLabel->clickScale_x = scale_x;
  def_imageLabel->clickScale_y = scale_y;
  def_imageLabel->name = "def_imageLabel";

  // Create a manager object to organize handling of multiples images and
  // multiples
  // domains. Generate reports with the results and plots
  manager = new managerClass();
  manager->moveToThread(&managerThread);

  // Create scroll Areas and associate the QLabels to them
  ui->scrollArea_und_image->setBackgroundRole(QPalette::Dark);
  ui->scrollArea_und_image->setWidget(und_imageLabel);
  ui->scrollArea_und_image->setWidgetResizable(false);

  ui->scrollArea_def_image->setBackgroundRole(QPalette::Dark);
  ui->scrollArea_def_image->setWidget(def_imageLabel);
  ui->scrollArea_def_image->setWidgetResizable(false);

  // Instanciate the parameters window and initialize GUI controls
  // with default values
  subdivisions = new Subdi();

  no_yesList << "No"
             << "Yes";

  deformation_descriptionList << "Strict Lagrangian"
                              << "Lagrangian"
                              << "Eulerian";
  deformation_description_iv = def_Eulerian;
  subdivisions->deformation_descriptionBox->addItems(
      deformation_descriptionList);
  subdivisions->deformation_descriptionBox->setCurrentIndex(
      deformation_description_iv);

  error_handlingModeList << "Stop all frames"
                         << "Stop current frame"
                         << "Ignore";
  error_handlingMode_iv = errorMode_stopAll;
  subdivisions->error_handlingModeBox->addItems(error_handlingModeList);
  subdivisions->error_handlingModeBox->setCurrentIndex(error_handlingMode_iv);

  reference_imageList << "First Image"
                      << "Previous Image";
  reference_image_iv = refImage_First;
  subdivisions->reference_imageBox->addItems(reference_imageList);
  subdivisions->reference_imageBox->setCurrentIndex(reference_image_iv);
  // allows disableing the First image reference model in Lagrangian deformation
  // descriptions
  comboBox_reference_image = qobject_cast<QStandardItemModel *>(
      subdivisions->reference_imageBox->model());
  item_first_ref_image = comboBox_reference_image->item(0);
  if (deformation_description_iv == def_Eulerian) {
    item_first_ref_image->setEnabled(true);
  } else
    item_first_ref_image->setEnabled(false);

  number_of_threads = NUMBER_OF_THREADS;
  manager->set_number_of_threads(number_of_threads);
  subdivisions->number_of_GPUs_threads_Box->setValue(number_of_threads);
  subdivisions->number_of_GPUs_threads_Box->setMinimum(1);
#if CUDA_ENABLED
  processorList << "CPU"
                << "GPU";
  cuda_manager = new CudaClass;
#if DEBUG_CUDA
  printf("MainApp::MainApp cuda_manager created\n");
#endif
#else
  processorList << "CPU";
#endif
  processor_iv = processor_CPU;
  subdivisions->processorBox->addItems(processorList);
  subdivisions->processorBox->setCurrentIndex(processor_iv);

  arrow_iv = false;
  subdivisions->arrowBox->addItems(no_yesList);
  subdivisions->arrowBox->setCurrentIndex(arrow_iv);

  update_images_iv = true;
  subdivisions->update_images_Box->addItems(no_yesList);
  subdivisions->update_images_Box->setCurrentIndex(update_images_iv);

  preload_images_iv = false;
  subdivisions->preload_images_Box->addItems(no_yesList);
  subdivisions->preload_images_Box->setCurrentIndex(preload_images_iv);

  plot_inside_points_iv = false;
  subdivisions->plot_inside_pointsBox->addItems(no_yesList);
  subdivisions->plot_inside_pointsBox->setCurrentIndex(plot_inside_points_iv);

  plot_contour_points_iv = true;
  subdivisions->plot_contour_pointsBox->addItems(no_yesList);
  subdivisions->plot_contour_pointsBox->setCurrentIndex(plot_contour_points_iv);

  arrow_magnification_iv = 1.0;
  subdivisions->arrow_magnificationBox->setValue(arrow_magnification_iv);

  icList << "Null"
         << "Automatic"
         << "User Picked";
  initial_guess_iv = ic_User;
  ui->comboBox_Initial_Guess->addItems(icList);
  ui->comboBox_Initial_Guess->setCurrentIndex(initial_guess_iv);

  colorList << "Monochrome"
            << "Color";
  color_mode_iv = color_monochrome;
  subdivisions->colorBox->addItems(colorList);
  subdivisions->colorBox->setCurrentIndex(color_mode_iv);
  ui->comboBox_color->addItems(colorList);
  ui->comboBox_color->setCurrentIndex(color_mode_iv);

  updateList << "Forward"
             << "Inverse";
  update_iv = update_forward;
  subdivisions->updateBox->addItems(updateList);
  subdivisions->updateBox->setCurrentIndex(update_iv);
  ui->comboBox_update->addItems(updateList);
  ui->comboBox_update->setCurrentIndex(update_iv);

  interpolation_modelList << "Nearest"
                          << "Bilinear"
                          << "Bicubic";
  interpolation_iv = im_bicubic;
  subdivisions->interpolation_model_Box->addItems(interpolation_modelList);
  subdivisions->interpolation_model_Box->setCurrentIndex(interpolation_iv);
  ui->comboBox_interpolation_model->addItems(interpolation_modelList);
  ui->comboBox_interpolation_model->setCurrentIndex(interpolation_iv);

  domainList << "Rectangular"
             << "Annular"
             << "Blob";
  domain_iv = domain_rectangular;
  ui->QTabWidget_domain->setCurrentIndex(domain_iv);
  und_imageLabel->domain = domain_iv;
  def_imageLabel->domain = domain_iv;

  initialize_domains();

  // Fitting model
  fit_modelList << "U"
                << "U, V"
                << "U, V, Q"
                << "U, V, Ux, Uy, Vx, Vy";
  model_iv = fm_UVUxUyVxVy;
  ui->comboBox_correlation_model->addItems(fit_modelList);
  ui->comboBox_correlation_model->setCurrentIndex(model_iv);
  und_imageLabel->model = model_iv;
  def_imageLabel->model = model_iv;

  // Using the initial_guess_iv = Null and model_iv, puts togueter an initial
  // guess and shares it
  // with the manager and def_imageLabel
  selectInitialGuess();

  // Pyramid parameters
  py_start = 0;
  subdivisions->blur_start_Box->setValue(py_start);
  ui->spinBox_py_start->setValue(py_start);

  py_step = 1;
  subdivisions->blur_step_Box->setValue(py_start);
  ui->spinBox_py_step->setValue(py_step);

  py_stop = 2;
  ui->spinBox_py_stop->setValue(py_stop);

  // Convergence parameters
  max_iters = 50;
  subdivisions->max_iters_Box->setValue(max_iters);
  ui->spinBox_max_iters->setValue(max_iters);

  precision = 0.001f;
  subdivisions->precision_Box->setValue(precision);
  ui->doubleSpinBox_precision->setValue(precision);

  // Intially hide the set displays - User can toggle from MainApp
  sets = true;
  reverse_sets();

  // Create progress bar
  ui->progressBar->setValue(0);
  ui->progressBar->hide();

  //  Initial default directory name, moving forward code remembers last
  //  directory.
  last_directory_name =
      "/home/namascar/Documents/data/corr_img/stick_slip2/jpg";

  // Define MainApp behaviour
  createActions();

#if AUTO_PILOT

  open();

  domain_iv = domain_annular;
  ui->QTabWidget_domain->setCurrentIndex(domain_iv);

  plot_contour_points_iv = false;

  annularDomain.ready = true;

  annularDomain.x_center = 528;
  ui->doubleSpinBox_rect_x0->setValue(annularDomain.x_center);

  annularDomain.y_center = 563;
  ui->doubleSpinBox_rect_y0->setValue(annularDomain.y_center);

  annularDomain.r_inside = 170;
  ui->doubleSpinBox_rect_xend->setValue(annularDomain.r_inside);

  annularDomain.r_outside = 340;
  ui->doubleSpinBox_rect_yend->setValue(annularDomain.r_outside);

  annularDomain.angular_subdivisions = 1;
  ui->doubleSpinBox_rect_xend->setValue(annularDomain.r_inside);

  annularDomain.radial_subdivisions = 1;
  ui->doubleSpinBox_rect_yend->setValue(annularDomain.r_outside);

  change_domain(domain_iv);

  processor_iv = processor_GPU;
  subdivisions->processorBox->setCurrentIndex(processor_iv);
  subdivisions->number_of_GPUs_threads_Box->setValue(2);

  max_iters = 50;

  correlate();

#endif
}

MainApp::~MainApp() {
  delete ui;
  delete manager;
  subdivisions->close();
  delete subdivisions;

#if CUDA_ENABLED
  delete cuda_manager;
#endif

  for (int i = 0; i < fm_NUMBER_OF_ITEMS; ++i)
    delete[] initial_guess_archive[i];
  delete[] initial_guess_archive;

  managerThread.quit();
  managerThread.wait();
}

void MainApp::refresh_images() {
  if (current_und_file_QString != first_und_file_QString ||
      current_def_file_QString != first_def_file_QString) {
    display_two_images(first_und_file_QString, first_def_file_QString);
  }
}

bool MainApp::display_two_images(QString und_file_QString_in,
                                 QString def_file_QString_in) {
  if (current_und_file_QString != und_file_QString_in) {
    bool success = display_undeformed_image(und_file_QString_in);
    if (success) {
      current_und_file_QString = und_file_QString_in;
    } else {
      return false;
    }
  }

  if (current_def_file_QString != def_file_QString_in) {
    bool success = display_deformed_image(def_file_QString_in);
    if (success) {
      current_def_file_QString = def_file_QString_in;
    } else {
      return false;
    }
  }

  return true;
}

bool MainApp::display_undeformed_image(QString und_file_QString_in) {
  QImage und_image(und_file_QString_in);

  if (und_image.isNull()) {
    return false;
  } else {
#if AUTO_PILOT
#else
    und_pixmap = QPixmap::fromImage(und_image);
    und_imageLabel->setPixmap(und_pixmap);

    QString first_file_path = und_file_QString_in;
    QString first_file_name =
        first_file_path.remove(0, first_file_path.lastIndexOf("/") + 1);
    ui->label_undeformed_name->setText(first_file_name);

    if (ui->actionFit_to_Window->isChecked()) {
      fitToWindow();
    } else {
      scaleImage(1.0);
    }

    und_imageLabel->update();
#endif
  }

  return true;
}

bool MainApp::display_deformed_image(QString def_file_QString_in) {
  QImage def_image(def_file_QString_in);

  if (def_image.isNull()) {
    return false;
  } else {
    if (def_image.isGrayscale())
      number_of_colors = 1;
    else
      number_of_colors = 3;

#if AUTO_PILOT
#else
    def_pixmap = QPixmap::fromImage(def_image);
    def_imageLabel->setPixmap(def_pixmap);

    QString second_file_path = def_file_QString_in;
    QString second_file_name =
        second_file_path.remove(0, second_file_path.lastIndexOf("/") + 1);
    ui->label_deformed_name->setText(second_file_name);

    if (ui->actionFit_to_Window->isChecked()) {
      fitToWindow();
    } else {
      scaleImage(1.0);
    }

    def_imageLabel->update();
#endif
  }

  return true;
}

void MainApp::open() {
#if AUTO_PILOT
  //    fileNames <<
  //    "/home/namascar/Documents/data/corr_img/translation1X2Y/frame-0.png"
  //              <<
  //              "/home/namascar/Documents/data/corr_img/translation1X2Y/frame-1.png";
  fileNames << "/home/namascar/Documents/data/corr_img/Eyes/CLAHE_1024x1024/"
               "Dock_clahe_1024x1024.png"
            << "/home/namascar/Documents/data/corr_img/Eyes/CLAHE_1024x1024/"
               "Dock_clahe_1024x1024_1deg.png"
            << "/home/namascar/Documents/data/corr_img/Eyes/CLAHE_1024x1024/"
               "Dock_clahe_1024x1024_2deg.png"
            << "/home/namascar/Documents/data/corr_img/Eyes/CLAHE_1024x1024/"
               "Dock_clahe_1024x1024_3deg.png"
            << "/home/namascar/Documents/data/corr_img/Eyes/CLAHE_1024x1024/"
               "Dock_clahe_1024x1024_4deg.png"
            << "/home/namascar/Documents/data/corr_img/Eyes/CLAHE_1024x1024/"
               "Dock_clahe_1024x1024_5deg.png"
            << "/home/namascar/Documents/data/corr_img/Eyes/CLAHE_1024x1024/"
               "Dock_clahe_1024x1024_6deg.png"
            << "/home/namascar/Documents/data/corr_img/Eyes/CLAHE_1024x1024/"
               "Dock_clahe_1024x1024_7deg.png"
            << "/home/namascar/Documents/data/corr_img/Eyes/CLAHE_1024x1024/"
               "Dock_clahe_1024x1024_8deg.png"
            << "/home/namascar/Documents/data/corr_img/Eyes/CLAHE_1024x1024/"
               "Dock_clahe_1024x1024_9deg.png"
            << "/home/namascar/Documents/data/corr_img/Eyes/CLAHE_1024x1024/"
               "Dock_clahe_1024x1024_10deg.png";
//        fileNames <<
//        "/home/namascar/Documents/data/corr_img/Eyes/CLAHE_1024x1024/Dock_clahe_1024x1024.png"
//                  <<
//                  "/home/namascar/Documents/data/corr_img/Eyes/CLAHE_1024x1024/Dock_clahe_1024x1024_1deg.png";
#else
  fileNames = QFileDialog::getOpenFileNames(this, tr("Open Files"),
                                            last_directory_name);
#endif

  if (fileNames.size() >= 2) {
    QString und_file_QString = fileNames.at(0);
    first_und_file_QString = und_file_QString;
    QString def_file_QString = fileNames.at(1);
    first_def_file_QString = def_file_QString;
    QString nxt_file_QString = fileNames.size() > 2 ? fileNames.at(2) : "";
    first_nxt_file_QString = nxt_file_QString;

    last_directory_name = und_file_QString;
    last_directory_name = last_directory_name.remove(
        last_directory_name.lastIndexOf("/"), last_directory_name.length());

    if (display_two_images(und_file_QString, def_file_QString)) {
      und_imageLabel->show();
      def_imageLabel->show();

      updateGpuPyramids();
      loadNxtGpuImages();

      und_imageLabel->correlating = false;
      def_imageLabel->correlating = false;

      und_imageLabel->selecting = true;
      def_imageLabel->selecting = true;

      ui->action_Print->setEnabled(true);
      ui->actionFit_to_Window->setEnabled(true);
      ui->actionParame_ters->setEnabled(true);
      ui->pushButton_correlate->setEnabled(false);

      clearLabels();
      initialize_domains();

      und_imageLabel->suppress_selection_display = true;
      def_imageLabel->suppress_selection_display = true;

      updateActions();
    } else // images can not be loaded
    {
      QMessageBox::information(this, tr("MainWindow"),
                               tr("Cannot load images"));
    }
  } else // less than 2 images loaded
  {
    QMessageBox::information(this, tr("MainWindow"),
                             tr("Please open at least 2 images"));
  }
}

void MainApp::print() {
  Q_ASSERT(!und_imageLabel->pixmap(Qt::ReturnByValueConstant()).isNull());

#if !defined(QT_NO_PRINTER) && !defined(QT_NO_PRINTDIALOG)

  QPrintDialog dialog(&printer, this);

  if (dialog.exec()) {
    QPainter painter(&printer);
    QRect rect = painter.viewport();
    QPixmap pixmap = und_imageLabel->pixmap(Qt::ReturnByValueConstant());
    QSize size = pixmap.size();
    size.scale(rect.size(), Qt::KeepAspectRatio);
    painter.setViewport(rect.x(), rect.y(), size.width(), size.height());
    painter.setWindow(pixmap.rect());
    painter.drawPixmap(0, 0, pixmap);
  }
#endif
}

void MainApp::createActions() {
  // Create Menu actions - File
  ui->action_Open->setShortcut(tr("Ctrl+O"));
  connect(ui->action_Open, SIGNAL(triggered()), this, SLOT(open()));

  ui->action_Print->setShortcut(tr("Ctrl+P"));
  ui->action_Print->setEnabled(false);
  connect(ui->action_Print, SIGNAL(triggered()), this, SLOT(print()));

  ui->action_Exit->setShortcut(tr("Ctrl+Q"));
  connect(ui->action_Exit, SIGNAL(triggered()), this, SLOT(close_app()));

  ui->action_Save_Analysis->setShortcut(tr("Ctrl+S"));
  ui->action_Save_Analysis->setEnabled(false);
  connect(ui->action_Save_Analysis, SIGNAL(triggered()), this,
          SLOT(saveAnalysisReport()));

  // Create Menu actions - View
  ui->actionZoom_in_25->setShortcut(tr("Ctrl++"));
  ui->actionZoom_in_25->setEnabled(false);
  connect(ui->actionZoom_in_25, SIGNAL(triggered()), this, SLOT(zoomIn()));

  ui->actionZoom_Out_25->setShortcut(tr("Ctrl+-"));
  ui->actionZoom_Out_25->setEnabled(false);
  connect(ui->actionZoom_Out_25, SIGNAL(triggered()), this, SLOT(zoomOut()));

  ui->action_Normal_Size->setShortcut(tr("Ctrl+S"));
  ui->action_Normal_Size->setEnabled(false);
  connect(ui->action_Normal_Size, SIGNAL(triggered()), this,
          SLOT(normalSize()));

  ui->actionFit_to_Window->setShortcut(tr("Ctrl+F"));
  ui->actionFit_to_Window->setEnabled(false);
  ui->actionFit_to_Window->setCheckable(true);
  ui->actionFit_to_Window->setChecked(true);
  connect(ui->actionFit_to_Window, SIGNAL(triggered()), this,
          SLOT(fitToWindow()));

  // Create Menu actions - Tools
  ui->actionParame_ters->setShortcut(tr("Ctrl+t"));
  ui->actionParame_ters->setEnabled(false);
  connect(ui->actionParame_ters, SIGNAL(triggered()), this,
          SLOT(open_subdivisions()));
  connect(subdivisions->okButton, SIGNAL(released()), this,
          SLOT(close_subdivisions()));
  connect(subdivisions->deformation_descriptionBox,
          SIGNAL(currentIndexChanged(int)), this,
          SLOT(change_deformation_description(int)));
  connect(subdivisions->processorBox, SIGNAL(currentIndexChanged(int)), this,
          SLOT(change_processor(int)));
  connect(subdivisions->number_of_GPUs_threads_Box, SIGNAL(valueChanged(int)),
          this, SLOT(change_devices_used(int)));
  connect(subdivisions->preload_images_Box, SIGNAL(currentIndexChanged(int)),
          this, SLOT(change_preload_images(int)));

  ui->pushButton_correlate->setEnabled(false);
  connect(ui->pushButton_correlate, SIGNAL(released()), this,
          SLOT(correlate()));
  connect(this, SIGNAL(start_correlation()), manager,
          SLOT(perform_multiframe_correlation()));

  // connection with the und_imageLabel and def_imageLabel that select and
  // mirror domains
  connect(this, SIGNAL(valueChanged_scaleSelection()), und_imageLabel,
          SLOT(scaleSquare()));
  connect(this, SIGNAL(valueChanged_scaleSelection()), def_imageLabel,
          SLOT(scaleSquare()));
  connect(this, SIGNAL(valueChanged_domain()), und_imageLabel,
          SLOT(GUIupdated()));
  connect(und_imageLabel,
          SIGNAL(valueChanged_rectangularSelected(float, float, float, float,
                                                  float, float, float, float)),
          this, SLOT(enableRectangularCorrelate(float, float, float, float,
                                                float, float, float, float)));
  connect(und_imageLabel, SIGNAL(valueChanged_annularSelected(
                              float, float, float, float, float, float, float)),
          this, SLOT(enableAnnularCorrelate(float, float, float, float, float,
                                            float, float)));
  connect(und_imageLabel,
          SIGNAL(valueChanged_blobSelected(v_points, float, float)), this,
          SLOT(enableBlobCorrelate(v_points, float, float)));

  connect(und_imageLabel,
          SIGNAL(valueChanged_selectingRectangular(float, float, float, float,
                                                   float, float, float, float)),
          def_imageLabel,
          SLOT(mirrorSelectionRectangular(float, float, float, float, float,
                                          float, float, float)));
  connect(und_imageLabel, SIGNAL(valueChanged_selectingAnnular(
                              float, float, float, float, float, float, float)),
          def_imageLabel,
          SLOT(mirrorSelectionAnnular(float, float, float, float, float, float,
                                      float)));
  connect(und_imageLabel, SIGNAL(valueChanged_selectingBlob(
                              float, float, v_points, float, float)),
          def_imageLabel,
          SLOT(mirrorSelectionBlob(float, float, v_points, float, float)));

  connect(und_imageLabel,
          SIGNAL(valueChanged_selectingRectangular(float, float, float, float,
                                                   float, float, float, float)),
          this, SLOT(refresh_images()));
  connect(und_imageLabel, SIGNAL(valueChanged_selectingAnnular(
                              float, float, float, float, float, float, float)),
          this, SLOT(refresh_images()));
  connect(und_imageLabel, SIGNAL(valueChanged_selectingBlob(
                              float, float, v_points, float, float)),
          this, SLOT(refresh_images()));

  connect(und_imageLabel, SIGNAL(stop_correlation_display()), this,
          SLOT(stop_correlation_display()));
  connect(def_imageLabel, SIGNAL(stop_correlation_display()), this,
          SLOT(stop_correlation_display()));
  connect(def_imageLabel, SIGNAL(new_initial_conditions()), this,
          SLOT(updateResultDisplay()));
  connect(und_imageLabel, SIGNAL(update_mirror()), def_imageLabel,
          SLOT(update()));

  // Connection of the GUI controls
  connect(ui->comboBox_correlation_model, SIGNAL(currentIndexChanged(int)),
          this, SLOT(change_correlation_model(int)));
  connect(ui->comboBox_color, SIGNAL(currentIndexChanged(int)), this,
          SLOT(change_color(int)));
  connect(ui->comboBox_Initial_Guess, SIGNAL(currentIndexChanged(int)), this,
          SLOT(change_Initial_Guess(int)));
  connect(ui->comboBox_interpolation_model, SIGNAL(currentIndexChanged(int)),
          this, SLOT(change_interpolation_model(int)));
  connect(ui->comboBox_update, SIGNAL(currentIndexChanged(int)), this,
          SLOT(change_update(int)));
  connect(ui->QTabWidget_domain, SIGNAL(currentChanged(int)), this,
          SLOT(change_domain(int)));

  connect(ui->spinBox_max_iters, SIGNAL(valueChanged(int)), this,
          SLOT(change_max_iters(int)));
  connect(ui->doubleSpinBox_precision, SIGNAL(valueChanged(double)), this,
          SLOT(change_precision(double)));
  connect(ui->spinBox_py_start, SIGNAL(valueChanged(int)), this,
          SLOT(change_star_pyramid(int)));
  connect(ui->spinBox_py_step, SIGNAL(valueChanged(int)), this,
          SLOT(change_step_pyramid(int)));
  connect(ui->spinBox_py_stop, SIGNAL(valueChanged(int)), this,
          SLOT(change_stop_pyramid(int)));

  connect(ui->spinBox_rect_horizontal, SIGNAL(valueChanged(int)), this,
          SLOT(change_rect_horizontal(int)));
  connect(ui->spinBox_rect_vertical, SIGNAL(valueChanged(int)), this,
          SLOT(change_rect_vertical(int)));
  connect(ui->spinBox_ann_radial, SIGNAL(valueChanged(int)), this,
          SLOT(change_ann_radial(int)));
  connect(ui->spinBox_ann_angular, SIGNAL(valueChanged(int)), this,
          SLOT(change_ann_angular(int)));

  connect(ui->doubleSpinBox_rect_x0, SIGNAL(valueChanged(double)), this,
          SLOT(change_rect_x0(double)));
  connect(ui->doubleSpinBox_rect_xend, SIGNAL(valueChanged(double)), this,
          SLOT(change_rect_xend(double)));
  connect(ui->doubleSpinBox_rect_y0, SIGNAL(valueChanged(double)), this,
          SLOT(change_rect_y0(double)));
  connect(ui->doubleSpinBox_rect_yend, SIGNAL(valueChanged(double)), this,
          SLOT(change_rect_yend(double)));

  connect(ui->doubleSpinBox_ann_center_x, SIGNAL(valueChanged(double)), this,
          SLOT(change_ann_center_x(double)));
  connect(ui->doubleSpinBox_ann_center_y, SIGNAL(valueChanged(double)), this,
          SLOT(change_ann_center_y(double)));
  connect(ui->doubleSpinBox_ann_ro, SIGNAL(valueChanged(double)), this,
          SLOT(change_ann_ro(double)));
  connect(ui->doubleSpinBox_ann_ri, SIGNAL(valueChanged(double)), this,
          SLOT(change_ann_ri(double)));

  connect(ui->doubleSpinBox_blob_center_x, SIGNAL(valueChanged(double)), this,
          SLOT(change_blob_center_x(double)));
  connect(ui->doubleSpinBox_blob_center_y, SIGNAL(valueChanged(double)), this,
          SLOT(change_blob_center_y(double)));
  connect(ui->doubleSpinBox_blob_x_scale, SIGNAL(valueChanged(double)), this,
          SLOT(change_blob_x_scale(double)));
  connect(ui->doubleSpinBox_blob_y_scale, SIGNAL(valueChanged(double)), this,
          SLOT(change_blob_y_scale(double)));

  // Create Menu actions - Help
  connect(ui->action_About, SIGNAL(triggered()), this, SLOT(about()));

  // Create Bottom actions
  connect(ui->pushButton_show_sets, SIGNAL(released()), this,
          SLOT(reverse_sets()));

  // Create manager actions
  connect(manager, SIGNAL(clear_und_inside_points()), und_imageLabel,
          SLOT(clear_inside_points()));
  connect(manager, SIGNAL(clear_und_contour_points()), und_imageLabel,
          SLOT(clear_contour_points()));
  connect(manager, SIGNAL(clear_def_inside_points()), def_imageLabel,
          SLOT(clear_inside_points()));
  connect(manager, SIGNAL(clear_def_contour_points()), def_imageLabel,
          SLOT(clear_contour_points()));

  connect(manager, SIGNAL(correlation_is_done(bool)), this,
          SLOT(correlation_done(bool)));
  connect(manager, SIGNAL(display_results(float, float *)), this,
          SLOT(updateResultDisplay(float, float *)));

  connect(manager, SIGNAL(send_und_inside_points(v_points, bool)),
          und_imageLabel, SLOT(set_inside_points(v_points, bool)));
  connect(manager, SIGNAL(send_def_inside_points(v_points, bool)),
          def_imageLabel, SLOT(set_inside_points(v_points, bool)));
  connect(manager, SIGNAL(send_und_contour_points(v_points, bool)),
          und_imageLabel, SLOT(set_contour_points(v_points, bool)));
  connect(manager, SIGNAL(send_def_contour_points(v_points, bool)),
          def_imageLabel, SLOT(set_contour_points(v_points, bool)));
  connect(manager, SIGNAL(display_images(QString, QString)), this,
          SLOT(display_two_images(QString, QString)));
  connect(&managerThread, &QThread::finished, manager, &QObject::deleteLater);
  managerThread.start();
}

void MainApp::change_correlation_model(int model_in) {
  if (model_in != model_iv) {
    model_iv = static_cast<fittingModelEnum>(model_in);

    if (ui->comboBox_correlation_model->currentIndex() != model_iv)
      ui->comboBox_correlation_model->setCurrentIndex(model_iv);

    und_imageLabel->model = model_iv;
    def_imageLabel->model = model_iv;
    manager->set_model(model_iv);

    ui->spinBox_ann_radial->setDisabled(false);
    ui->spinBox_ann_angular->setDisabled(false);

    selectInitialGuess();
    clearLabels();
  }
}

void MainApp::change_color(int color_in) {
  color_mode_iv = static_cast<colorEnum>(color_in);
  subdivisions->colorBox->setCurrentIndex(color_mode_iv);
  updateGpuPyramids();
  loadNxtGpuImages();
}

void MainApp::change_Initial_Guess(int guess_in) {
  initial_guess_iv = static_cast<initialGuessEnum>(guess_in);
  if (initial_guess_iv == ic_User)
    def_imageLabel->selecting_ic = true;
  else
    def_imageLabel->selecting_ic = false;

  selectInitialGuess();
  clearLabels();
}

void MainApp::change_interpolation_model(int interpolation_model_in) {
  interpolation_iv =
      static_cast<interpolationModelEnum>(interpolation_model_in);
  subdivisions->interpolation_model_Box->setCurrentIndex(interpolation_iv);
}

void MainApp::change_update(int update_in) {
  update_iv = static_cast<updateEnum>(update_in);
  subdivisions->updateBox->setCurrentIndex(update_iv);
}

void MainApp::change_deformation_description(int deformation_description_in) {
  deformation_description_iv =
      static_cast<deformationDescriptionEnum>(deformation_description_in);

  switch (deformation_description_iv) {
  case def_strict_Lagrangian:
  case def_Lagrangian:
    reference_image_iv = refImage_Previous;
    item_first_ref_image->setEnabled(false);
    subdivisions->reference_imageBox->setCurrentIndex(reference_image_iv);
    break;

  case def_Eulerian:
    reference_image_iv = refImage_First;
    subdivisions->reference_imageBox->setCurrentIndex(reference_image_iv);
    item_first_ref_image->setEnabled(true);
    break;

  default:
    assert(false);
    break;
  }
}

void MainApp::change_processor(int processor_in) {
  processor_iv = static_cast<processorEnum>(processor_in);

  switch (processor_iv) {
  case processor_GPU: {
#if CUDA_ENABLED
    //  Count number of GPUs available
    deviceCount = cuda_manager->initialize();
    devicesUsed = deviceCount;
    printf("MainApp: change_processor: %d GPUs detected\n", deviceCount);
    if (deviceCount > 0) {
      updateGpuPyramids();
      loadNxtGpuImages();

      manager->set_cuda_manager(cuda_manager);

      subdivisions->number_of_GPUs_threads_Label->setText("Number of GPUs");
      subdivisions->number_of_GPUs_threads_Box->blockSignals(true);
      subdivisions->number_of_GPUs_threads_Box->setMaximum(deviceCount);
      subdivisions->number_of_GPUs_threads_Box->setValue(deviceCount);
      subdivisions->number_of_GPUs_threads_Box->blockSignals(false);
    } else {
      // put back CPU mode and Disable the GPU mode
      processor_iv = processor_CPU;
      subdivisions->processorBox->setCurrentIndex(processor_iv);
      qobject_cast<QStandardItemModel *>(subdivisions->processorBox->model())
          ->item(processor_GPU)
          ->setEnabled(false);

      QMessageBox::information(
          this, tr("MainWindow"),
          tr("There are no available device(s) that support CUDA\nDisabling "
             "GPU mode and continuing in CPU mode"));
    }
#endif
    break;
  }
  case processor_CPU: {
    subdivisions->number_of_GPUs_threads_Label->setText("Number of threads");
    subdivisions->number_of_GPUs_threads_Box->setMaximum(1000);
    subdivisions->number_of_GPUs_threads_Box->setValue(number_of_threads);

    break;
  }
  default:
    assert(false);
    break;
  }
}

void MainApp::change_devices_used(int devices_or_threads_in) {
  switch (processor_iv) {
  case processor_GPU:
    devicesUsed = devices_or_threads_in;
#if CUDA_ENABLED
    cuda_manager->set_deviceCount(devicesUsed);
#endif
    break;

  case processor_CPU:
    number_of_threads = devices_or_threads_in;
    manager->set_number_of_threads(number_of_threads);
    break;

  default:
    assert(false);
    break;
  }
}

void MainApp::change_preload_images(int preload_images_in) {
  preload_images_iv = preload_images_in;
  loadNxtGpuImages();
}

void MainApp::change_domain(int domain_in) {
  domain_iv = static_cast<domainEnum>(domain_in);

  switch (domain_iv) {
  case domain_rectangular:

    ui->pushButton_correlate->setEnabled(rectangularDomain.ready);
    def_imageLabel->selecting_ic = rectangularDomain.ready;

    break;

  case domain_annular:

    ui->pushButton_correlate->setEnabled(annularDomain.ready);
    def_imageLabel->selecting_ic = annularDomain.ready;

    break;

  case domain_blob:

    ui->pushButton_correlate->setEnabled(blobDomain.ready);
    def_imageLabel->selecting_ic = blobDomain.ready;

    break;

  default:

    break;
  }

  und_imageLabel->domain = domain_iv;
  def_imageLabel->domain = domain_iv;

  clearLabels();

  updateDomains();
}

void MainApp::change_max_iters(int max_iters_in) {
  max_iters = max_iters_in;
  subdivisions->max_iters_Box->setValue(max_iters);
}

void MainApp::change_precision(double precision_in) {
  precision = (float)precision_in;
  subdivisions->precision_Box->setValue(precision);
}

void MainApp::change_star_pyramid(int star_py_in) {
  py_start = star_py_in;
  subdivisions->blur_start_Box->setValue(py_start);
  updateGpuPyramids();
}

void MainApp::change_step_pyramid(int step_py_in) {
  py_step = step_py_in;
  subdivisions->blur_step_Box->setValue(py_step);
  updateGpuPyramids();
}

void MainApp::change_stop_pyramid(int stop_py_in) {
  py_stop = stop_py_in;
  subdivisions->blur_stop_Box->setValue(py_stop);
  updateGpuPyramids();
}

void MainApp::updateGpuPyramids() {
#if CUDA_ENABLED
  if (processor_iv == processor_GPU) {
    cuda_manager->resetImagePyramids(first_und_file_QString.toStdString(),
                                     first_def_file_QString.toStdString(),
                                     first_nxt_file_QString.toStdString(),
                                     color_mode_iv, py_start, py_step, py_stop);
  }
#endif
}

void MainApp::loadNxtGpuImages() {
#if CUDA_ENABLED
  if (processor_iv == processor_GPU && preload_images_iv) {
    cv::ImreadModes color_flag;

    switch (number_of_colors) {
    case 1:
      color_flag = cv::IMREAD_GRAYSCALE;
      break;

    case 3:
      color_flag = cv::IMREAD_ANYCOLOR;
      break;

    default:
      assert(false);
      break;
    }

    cuda_manager->tempQ.empty();
    for (int iImage = 2; iImage < fileNames.size(); ++iImage) {
      nvtxRangePushA("MainApp::loadNxtGpuImages read image");
      cuda_manager->tempQ.push(
          cv::imread(fileNames.at(iImage).toStdString(), color_flag));
      nvtxRangePop();
    }
  }
#endif
}

void MainApp::clearLabels() {
  refresh_images();

  und_imageLabel->inside_points.clear();
  und_imageLabel->contour_points.clear();
  und_imageLabel->suppress_selection_display = false;
  und_imageLabel->update();

  def_imageLabel->inside_points.clear();
  def_imageLabel->contour_points.clear();
  def_imageLabel->suppress_selection_display = false;
}

void MainApp::change_rect_horizontal(int h_subdi_in) {
  rectangularDomain.horizontal_subdivisions = h_subdi_in;
  und_imageLabel->h_subdivisions = rectangularDomain.horizontal_subdivisions;
  def_imageLabel->h_subdivisions = rectangularDomain.horizontal_subdivisions;

  manager->set_domain(rectangularDomain);

  clearLabels();
}

void MainApp::updateRectangularCenter() {
  rectangularDomain.x_center =
      (rectangularDomain.x_begin + rectangularDomain.x_end) / 2.f;
  rectangularDomain.y_center =
      (rectangularDomain.y_begin + rectangularDomain.y_end) / 2.f;

  und_imageLabel->center_x = rectangularDomain.x_center * scale_x;
  und_imageLabel->center_y = rectangularDomain.y_center * scale_y;

#if DEBUG_DOMAIN_SELECTION
  printf("%15s %30s: center x = %6.2f  center_y = %6.2f  left_x = %6.2f  "
         "right_x = %6.2f  left_y = %6.2f  right_y = %6.2f  scale_x = %6.2f  "
         "scale_y = %6.2f\n",
         "MainApp", "updateRectangularCenter", rectangularDomain.x_center,
         rectangularDomain.y_center, rectangularDomain.x_begin,
         rectangularDomain.x_end, rectangularDomain.y_begin,
         rectangularDomain.y_end, scale_x, scale_y);
#endif
}

void MainApp::change_rect_vertical(int v_subdi_in) {
  rectangularDomain.vertical_subdivisions = v_subdi_in;
  und_imageLabel->v_subdivisions = rectangularDomain.vertical_subdivisions;
  def_imageLabel->v_subdivisions = rectangularDomain.vertical_subdivisions;

  manager->set_domain(rectangularDomain);

  clearLabels();
}
void MainApp::change_rect_x0(double x0_in) {
  rectangularDomain.x_begin = x0_in;

  und_imageLabel->left_x = rectangularDomain.x_begin * scale_x;
  def_imageLabel->left_x = rectangularDomain.x_begin * scale_x;

  updateRectangularCenter();

  manager->set_domain(rectangularDomain);

  clearLabels();
}
void MainApp::change_rect_y0(double y0_in) {
  rectangularDomain.y_begin = y0_in;

  und_imageLabel->left_y = rectangularDomain.y_begin * scale_y;
  def_imageLabel->left_y = rectangularDomain.y_begin * scale_y;

  updateRectangularCenter();

  manager->set_domain(rectangularDomain);

  clearLabels();
}
void MainApp::change_rect_xend(double xend_in) {
  rectangularDomain.x_end = xend_in;

  und_imageLabel->right_x = rectangularDomain.x_end * scale_x;
  def_imageLabel->right_x = rectangularDomain.x_end * scale_x;

  updateRectangularCenter();

  manager->set_domain(rectangularDomain);

  clearLabels();
}
void MainApp::change_rect_yend(double yend_in) {
  rectangularDomain.y_end = yend_in;

  und_imageLabel->right_y = rectangularDomain.y_end * scale_y;
  def_imageLabel->right_y = rectangularDomain.y_end * scale_y;

  updateRectangularCenter();

  manager->set_domain(rectangularDomain);

  clearLabels();
}
void MainApp::change_ann_center_x(double xc_in) {
  annularDomain.x_center = xc_in;

  und_imageLabel->center_x = annularDomain.x_center * scale_x;
  def_imageLabel->center_x = annularDomain.x_center * scale_x;

  manager->set_domain(annularDomain);

  clearLabels();
}
void MainApp::change_ann_center_y(double yc_in) {
  annularDomain.y_center = yc_in;

  und_imageLabel->center_y = annularDomain.y_center * scale_y;
  def_imageLabel->center_y = annularDomain.y_center * scale_y;

  manager->set_domain(annularDomain);

  clearLabels();
}
void MainApp::change_ann_ro(double ro_in) {
  annularDomain.r_outside = ro_in;
  annularDomain.ri_by_ro = annularDomain.r_inside / annularDomain.r_outside;

  und_imageLabel->outside_radius = annularDomain.r_outside * scale_x;
  def_imageLabel->outside_radius = annularDomain.r_outside * scale_x;

  und_imageLabel->ri_by_ro = annularDomain.ri_by_ro;
  def_imageLabel->ri_by_ro = annularDomain.ri_by_ro;

  manager->set_domain(annularDomain);

  clearLabels();
}
void MainApp::change_ann_ri(double ri_in) {
  annularDomain.r_inside = ri_in;
  annularDomain.ri_by_ro = annularDomain.r_inside / annularDomain.r_outside;

  und_imageLabel->ri_by_ro = annularDomain.ri_by_ro;
  def_imageLabel->ri_by_ro = annularDomain.ri_by_ro;

  manager->set_domain(annularDomain);

  clearLabels();
}
void MainApp::change_ann_radial(int r_subdi_in) {
  annularDomain.radial_subdivisions = r_subdi_in;
  und_imageLabel->r_subdivisions = annularDomain.radial_subdivisions;
  def_imageLabel->r_subdivisions = annularDomain.radial_subdivisions;

  manager->set_domain(annularDomain);

  clearLabels();
}
void MainApp::change_ann_angular(int a_subdi_in) {
  annularDomain.angular_subdivisions = a_subdi_in;
  und_imageLabel->a_subdivisions = annularDomain.angular_subdivisions;
  def_imageLabel->a_subdivisions = annularDomain.angular_subdivisions;

  manager->set_domain(annularDomain);

  clearLabels();
}

void MainApp::change_blob_center_x(double xc_in) {
  for (int i = 0; i < (int)blobDomain.xy_contour.size(); ++i) {
    blobDomain.xy_contour[i].first += xc_in - blobDomain.x_center;

    und_imageLabel->xy_blob[i].first = blobDomain.xy_contour[i].first * scale_x;
    def_imageLabel->xy_blob[i].first = blobDomain.xy_contour[i].first * scale_x;
  }

  blobDomain.x_center = xc_in;

  manager->set_domain(blobDomain);

  clearLabels();
}
void MainApp::change_blob_center_y(double yc_in) {
  for (int i = 0; i < (int)blobDomain.xy_contour.size(); ++i) {
    blobDomain.xy_contour[i].second += yc_in - blobDomain.y_center;

    und_imageLabel->xy_blob[i].second =
        blobDomain.xy_contour[i].second * scale_y;
    def_imageLabel->xy_blob[i].second =
        blobDomain.xy_contour[i].second * scale_y;
  }

  blobDomain.y_center = yc_in;

  manager->set_domain(blobDomain);

  clearLabels();
}
void MainApp::change_blob_x_scale(double xs_in) {
  for (int i = 0; i < (int)blobDomain.xy_contour.size(); ++i) {
    blobDomain.xy_contour[i].first =
        blobDomain.x_center +
        (blobDomain.xy_contour[i].first - blobDomain.x_center) * (float)xs_in /
            blobDomain.x_scale;

    und_imageLabel->xy_blob[i].first = blobDomain.xy_contour[i].first * scale_x;
    def_imageLabel->xy_blob[i].first = blobDomain.xy_contour[i].first * scale_x;
  }

  blobDomain.x_scale = (float)xs_in;

  manager->set_domain(blobDomain);

  clearLabels();
}
void MainApp::change_blob_y_scale(double ys_in) {
  for (int i = 0; i < (int)blobDomain.xy_contour.size(); ++i) {
    blobDomain.xy_contour[i].second =
        blobDomain.y_center +
        (blobDomain.xy_contour[i].second - blobDomain.y_center) * (float)ys_in /
            blobDomain.y_scale;

    und_imageLabel->xy_blob[i].second =
        blobDomain.xy_contour[i].second * scale_y;
    def_imageLabel->xy_blob[i].second =
        blobDomain.xy_contour[i].second * scale_y;
  }

  blobDomain.y_scale = (float)ys_in;

  manager->set_domain(blobDomain);

  clearLabels();
}

void MainApp::close_app() {
  subdivisions->close();
  close();
}

void MainApp::about() {
  QMessageBox::about(
      this, tr("About Correlation V2.0"),
      tr("<p><b>Correlation</b> performs digital image correlation "
         "on a set of user selected images. First open a sequence"
         " of images, then select the correlation domain. Specify "
         "the number of subdivisions, and finally select"
         " \"correlate\".</p>"));
}

void MainApp::open_subdivisions() {
  subdivisions->colorBox->setCurrentIndex(color_mode_iv);
  subdivisions->precision_Box->setValue(precision);

  subdivisions->updateBox->setCurrentIndex(update_iv);
  subdivisions->blur_start_Box->setValue(py_start);
  subdivisions->blur_step_Box->setValue(py_step);
  subdivisions->blur_stop_Box->setValue(py_stop);

  subdivisions->interpolation_model_Box->setCurrentIndex(interpolation_iv);

  subdivisions->max_iters_Box->setValue(max_iters);

  subdivisions->show();
}

void MainApp::reverse_sets() {
  sets = !sets;
  if (sets) {
    resize(1551, 1012);
    ui->pushButton_show_sets->setText("Hide Sets");
  } else {
    resize(1551, 733);
    ui->pushButton_show_sets->setText("Show Sets");
  }
}

void MainApp::close_subdivisions() {
  // Right layout
  color_mode_iv =
      static_cast<colorEnum>(subdivisions->colorBox->currentIndex());
  ui->comboBox_color->setCurrentIndex(color_mode_iv);

  precision = subdivisions->precision_Box->value();
  ui->doubleSpinBox_precision->setValue(precision);

  deformation_description_iv = static_cast<deformationDescriptionEnum>(
      subdivisions->deformation_descriptionBox->currentIndex());

  reference_image_iv = static_cast<referenceImageEnum>(
      subdivisions->reference_imageBox->currentIndex());

  processor_iv =
      static_cast<processorEnum>(subdivisions->processorBox->currentIndex());

  // center layout
  update_iv = static_cast<updateEnum>(subdivisions->updateBox->currentIndex());
  ui->comboBox_update->setCurrentIndex(update_iv);

  py_start = subdivisions->blur_start_Box->value();
  ui->spinBox_py_start->setValue(py_start);

  py_step = subdivisions->blur_step_Box->value();
  ui->spinBox_py_step->setValue(py_step);

  py_stop = subdivisions->blur_stop_Box->value();
  ui->spinBox_py_stop->setValue(py_stop);

  interpolation_iv = static_cast<interpolationModelEnum>(
      subdivisions->interpolation_model_Box->currentIndex());
  ui->comboBox_interpolation_model->setCurrentIndex(interpolation_iv);

  // left layout
  max_iters = subdivisions->max_iters_Box->value();
  ui->spinBox_max_iters->setValue(max_iters);
  update_images_iv = subdivisions->update_images_Box->currentIndex();
  if (update_images_iv) {
    connect(manager, SIGNAL(display_images(QString, QString)), this,
            SLOT(display_two_images(QString, QString)));
  } else {
    disconnect(manager, SIGNAL(display_images(QString, QString)), this,
               SLOT(display_two_images(QString, QString)));
  }

  // far right layout
  error_handlingMode_iv = static_cast<errorHandlingModeEnum>(
      subdivisions->error_handlingModeBox->currentIndex());
  arrow_iv = subdivisions->arrowBox->currentIndex();
  arrow_magnification_iv = subdivisions->arrow_magnificationBox->value();
  plot_inside_points_iv = subdivisions->plot_inside_pointsBox->currentIndex();
  plot_contour_points_iv = subdivisions->plot_contour_pointsBox->currentIndex();

  subdivisions->close();
}

void MainApp::zoomIn() { scaleImage(1.25); }

void MainApp::zoomOut() { scaleImage(0.8); }

void MainApp::normalSize() {
  und_imageLabel->adjustSize();
  def_imageLabel->adjustSize();

  scale_x = 1;
  scale_y = 1;

  und_imageLabel->realScale_x = scale_x;
  und_imageLabel->realScale_y = scale_y;
  def_imageLabel->realScale_x = scale_x;
  def_imageLabel->realScale_y = scale_y;

  // connect(this, SIGNAL(valueChanged_scaleSelection () ), und_imageLabel,
  // SLOT( scaleSquare() ) );
  // connect(this, SIGNAL(valueChanged_scaleSelection () ), def_imageLabel,
  // SLOT( scaleSquare() ) );
  emit valueChanged_scaleSelection();
}

void MainApp::fitToWindow() {
  if (ui->actionFit_to_Window->isChecked()) {
    ui->scrollArea_und_image->setWidgetResizable(true);
    ui->scrollArea_def_image->setWidgetResizable(true);
    QPixmap pixmap = und_imageLabel->pixmap(Qt::ReturnByValueConstant());

    scale_x = float(ui->scrollArea_und_image->width()) /
              float(pixmap.width());
    scale_y = float(ui->scrollArea_und_image->height()) /
              float(pixmap.height());

    und_imageLabel->realScale_x = scale_x;
    und_imageLabel->realScale_y = scale_y;
    def_imageLabel->realScale_x = scale_x;
    def_imageLabel->realScale_y = scale_y;

    // connect(this, SIGNAL(valueChanged_scaleSelection () ), und_imageLabel,
    // SLOT( scaleSquare() ) );
    // connect(this, SIGNAL(valueChanged_scaleSelection () ), def_imageLabel,
    // SLOT( scaleSquare() ) );
    emit valueChanged_scaleSelection();
  } else {
    ui->scrollArea_und_image->setWidgetResizable(false);
    ui->scrollArea_def_image->setWidgetResizable(false);

    normalSize();
  }
  updateActions();
}

void MainApp::scaleImage(double factor) {
  Q_ASSERT(!und_imageLabel->pixmap(Qt::ReturnByValueConstant()).isNull());

  scale_x *= factor;
  scale_y *= factor;

  und_imageLabel->realScale_x = scale_x;
  und_imageLabel->realScale_y = scale_y;
  def_imageLabel->realScale_x = scale_x;
  def_imageLabel->realScale_y = scale_y;

  // connect(this, SIGNAL(valueChanged_scaleSelection () ), und_imageLabel,
  // SLOT( scaleSquare() ) );
  // connect(this, SIGNAL(valueChanged_scaleSelection () ), def_imageLabel,
  // SLOT( scaleSquare() ) );
  emit valueChanged_scaleSelection();

  QPixmap pixmap = und_imageLabel->pixmap(Qt::ReturnByValueConstant());
  QPixmap def_pixmap = def_imageLabel->pixmap(Qt::ReturnByValueConstant());
  ui->scrollArea_und_image->setWidgetResizable(false);
  ui->scrollArea_def_image->setWidgetResizable(false);
  und_imageLabel->resize(
      und_imageLabel->realScale_x * pixmap.width(),
      und_imageLabel->realScale_y * pixmap.height());
  def_imageLabel->resize(
      def_imageLabel->realScale_x * def_pixmap.width(),
      def_imageLabel->realScale_y * def_pixmap.height());

  adjustScrollBar(ui->scrollArea_und_image->horizontalScrollBar(), factor);
  adjustScrollBar(ui->scrollArea_und_image->verticalScrollBar(), factor);
  adjustScrollBar(ui->scrollArea_def_image->horizontalScrollBar(), factor);
  adjustScrollBar(ui->scrollArea_def_image->verticalScrollBar(), factor);

  ui->actionZoom_in_25->setEnabled(und_imageLabel->realScale_x < 6.0 &&
                                   und_imageLabel->realScale_y < 6.0);
  ui->actionZoom_Out_25->setEnabled(und_imageLabel->realScale_x > 0.167 &&
                                    und_imageLabel->realScale_y > 0.167);
  ui->actionZoom_in_25->setEnabled(def_imageLabel->realScale_x < 6.0 &&
                                   def_imageLabel->realScale_y < 6.0);
  ui->actionZoom_Out_25->setEnabled(def_imageLabel->realScale_x > 0.167 &&
                                    def_imageLabel->realScale_y > 0.167);
}

void MainApp::adjustScrollBar(QScrollBar *scrollBar, double factor) {
  scrollBar->setValue(int(factor * scrollBar->value() +
                          ((factor - 1) * scrollBar->pageStep() / 2)));
}

void MainApp::updateActions() {
  ui->actionZoom_in_25->setEnabled(!ui->actionFit_to_Window->isChecked());
  ui->actionZoom_Out_25->setEnabled(!ui->actionFit_to_Window->isChecked());
  ui->action_Normal_Size->setEnabled(!ui->actionFit_to_Window->isChecked());
}

void MainApp::enableRectangularCorrelate(float center_x_in, float center_y_in,
                                         float right_x_in, float right_y_in,
                                         float left_x_in, float left_y_in,
                                         float scale_x_in, float scale_y_in) {
  rectangularDomain.ready = true;
  ui->pushButton_correlate->setEnabled(true);
  def_imageLabel->selecting_ic = true;

  scale_x = scale_x_in;
  scale_y = scale_y_in;

  rectangularDomain.x_begin = left_x_in / scale_x;
  rectangularDomain.y_begin = left_y_in / scale_y;
  rectangularDomain.x_end = right_x_in / scale_x;
  rectangularDomain.y_end = right_y_in / scale_y;

  rectangularDomain.x_center = center_x_in / scale_x;
  rectangularDomain.y_center = center_y_in / scale_y;

#if DEBUG_DOMAIN_SELECTION
  printf("%15s %30s: center x = %6.2f  center_y = %6.2f  left_x = %6.2f  "
         "right_x = %6.2f  left_y = %6.2f  right_y = %6.2f  scale_x = %6.2f  "
         "scale_y = %6.2f\n",
         "MainApp", "enableRectangularCorrelate", rectangularDomain.x_center,
         rectangularDomain.y_center, rectangularDomain.x_begin,
         rectangularDomain.x_end, rectangularDomain.y_begin,
         rectangularDomain.y_end, scale_x, scale_y);
#endif

  updateDomains();
}

void MainApp::enableAnnularCorrelate(float center_x_in, float center_y_in,
                                     float inside_radius_in,
                                     float outside_radius_in, float ri_by_ro_in,
                                     float scale_x_in, float scale_y_in) {
  annularDomain.ready = true;
  ui->action_Correlate->setEnabled(true);
  ui->pushButton_correlate->setEnabled(true);
  def_imageLabel->selecting_ic = true;

  scale_x = scale_x_in;
  scale_y = scale_y_in;

  annularDomain.x_center = center_x_in / scale_x;
  annularDomain.y_center = center_y_in / scale_y;

  // Scaling by x the radius is ok, since imagelabel
  // plots an ellipse that takes care of xy scaling
  annularDomain.r_outside = outside_radius_in / scale_x;
  annularDomain.r_inside = inside_radius_in / scale_x;

  annularDomain.ri_by_ro = ri_by_ro_in;

  updateDomains();
}

void MainApp::enableBlobCorrelate(v_points xy_blob_in, float scale_x_in,
                                  float scale_y_in) {
  blobDomain.ready = true;
  ui->pushButton_correlate->setEnabled(true);
  def_imageLabel->selecting_ic = true;

  scale_x = scale_x_in;
  scale_y = scale_y_in;

  blobDomain.xy_contour.clear();

  blobDomain.x_center = 0.f;
  blobDomain.y_center = 0.f;

  for (int i = 0; i < (int)xy_blob_in.size(); ++i) {
    blobDomain.xy_contour.push_back(std::make_pair(
        xy_blob_in[i].first / scale_x, xy_blob_in[i].second / scale_y));

    blobDomain.x_center += xy_blob_in[i].first / scale_x;
    blobDomain.y_center += xy_blob_in[i].second / scale_y;
  }

  blobDomain.x_center /= (float)xy_blob_in.size();
  blobDomain.y_center /= (float)xy_blob_in.size();

  blobDomain.x_scale = 1.f;
  blobDomain.y_scale = 1.f;

  updateDomains();
}

void MainApp::updateDomains() {
  updateResultDisplay();

  switch (domain_iv) {
  case domain_rectangular:

    und_imageLabel->center_x = rectangularDomain.x_center * scale_x;
    und_imageLabel->center_y = rectangularDomain.y_center * scale_y;

    und_imageLabel->left_x = rectangularDomain.x_begin * scale_x;
    und_imageLabel->left_y = rectangularDomain.y_begin * scale_y;
    und_imageLabel->right_x = rectangularDomain.x_end * scale_x;
    und_imageLabel->right_y = rectangularDomain.y_end * scale_y;

    manager->set_domain(rectangularDomain);

    ui->doubleSpinBox_rect_x0->blockSignals(true);
    ui->doubleSpinBox_rect_xend->blockSignals(true);
    ui->doubleSpinBox_rect_y0->blockSignals(true);
    ui->doubleSpinBox_rect_yend->blockSignals(true);

    ui->doubleSpinBox_rect_x0->setValue(rectangularDomain.x_begin);
    ui->doubleSpinBox_rect_y0->setValue(rectangularDomain.y_begin);
    ui->doubleSpinBox_rect_xend->setValue(rectangularDomain.x_end);
    ui->doubleSpinBox_rect_yend->setValue(rectangularDomain.y_end);

    ui->doubleSpinBox_rect_x0->blockSignals(false);
    ui->doubleSpinBox_rect_xend->blockSignals(false);
    ui->doubleSpinBox_rect_y0->blockSignals(false);
    ui->doubleSpinBox_rect_yend->blockSignals(false);

#if DEBUG_DOMAIN_SELECTION
    std::cout << std::endl;
    printf("%15s %30s: center x = %6.2f  center_y = %6.2f  left_x = %6.2f  "
           "right_x = %6.2f  left_y = %6.2f  right_y = %6.2f  scale_x = %6.2f  "
           "scale_y = %6.2f\n",
           "MainApp", "updateDomains", und_imageLabel->center_x,
           und_imageLabel->center_y, und_imageLabel->left_x,
           und_imageLabel->left_y, und_imageLabel->right_x,
           und_imageLabel->right_y, scale_x, scale_y);
#endif

    break;

  case domain_annular:

    und_imageLabel->center_x = annularDomain.x_center * scale_x;
    und_imageLabel->center_y = annularDomain.y_center * scale_y;

    und_imageLabel->outside_radius =
        annularDomain.r_outside * scale_x; // Scale radius with x
    und_imageLabel->inside_radius = annularDomain.r_inside * scale_x;
    und_imageLabel->ri_by_ro = annularDomain.ri_by_ro;

    manager->set_domain(annularDomain);

    ui->doubleSpinBox_ann_center_x->blockSignals(true);
    ui->doubleSpinBox_ann_center_y->blockSignals(true);
    ui->doubleSpinBox_ann_ro->blockSignals(true);
    ui->doubleSpinBox_ann_ri->blockSignals(true);

    ui->doubleSpinBox_ann_center_x->setValue(annularDomain.x_center);
    ui->doubleSpinBox_ann_center_y->setValue(annularDomain.y_center);
    ui->doubleSpinBox_ann_ro->setValue(annularDomain.r_outside);
    ui->doubleSpinBox_ann_ri->setValue(annularDomain.r_inside);

    ui->doubleSpinBox_ann_center_x->blockSignals(false);
    ui->doubleSpinBox_ann_center_y->blockSignals(false);
    ui->doubleSpinBox_ann_ro->blockSignals(false);
    ui->doubleSpinBox_ann_ri->blockSignals(false);

    break;

  case domain_blob:

    und_imageLabel->xy_blob.clear();

    if (blobDomain.xy_contour.size()) {
      blobDomain.x_center = 0;
      blobDomain.y_center = 0;

      und_imageLabel->center_x = 0.f;
      und_imageLabel->center_y = 0.f;

      for (int i = 0; i < (int)blobDomain.xy_contour.size(); ++i) {
        blobDomain.x_center += blobDomain.xy_contour[i].first;
        blobDomain.y_center += blobDomain.xy_contour[i].second;

        und_imageLabel->xy_blob.push_back(
            std::make_pair(blobDomain.xy_contour[i].first * scale_x,
                           blobDomain.xy_contour[i].second * scale_y));
      }

      blobDomain.x_center /= blobDomain.xy_contour.size();
      blobDomain.y_center /= blobDomain.xy_contour.size();
    }

    und_imageLabel->center_x = blobDomain.x_center * scale_x;
    und_imageLabel->center_y = blobDomain.y_center * scale_y;

    und_imageLabel->blob_scale_x = blobDomain.x_scale;
    und_imageLabel->blob_scale_y = blobDomain.y_scale;

    manager->set_domain(blobDomain);

    ui->doubleSpinBox_blob_center_x->blockSignals(true);
    ui->doubleSpinBox_blob_center_y->blockSignals(true);
    ui->doubleSpinBox_blob_x_scale->blockSignals(true);
    ui->doubleSpinBox_blob_y_scale->blockSignals(true);

    ui->doubleSpinBox_blob_center_x->setValue(blobDomain.x_center);
    ui->doubleSpinBox_blob_center_y->setValue(blobDomain.y_center);
    ui->doubleSpinBox_blob_x_scale->setValue(blobDomain.x_scale);
    ui->doubleSpinBox_blob_y_scale->setValue(blobDomain.y_scale);

    ui->doubleSpinBox_blob_center_x->blockSignals(false);
    ui->doubleSpinBox_blob_center_y->blockSignals(false);
    ui->doubleSpinBox_blob_x_scale->blockSignals(false);
    ui->doubleSpinBox_blob_y_scale->blockSignals(false);

    break;

  default:

    assert(false);

    break;
  }
  // connect(this, SIGNAL( valueChanged_domain() ), und_imageLabel,
  // SLOT(GUIupdated() ) );
  emit valueChanged_domain();
}

void MainApp::correlate() {
  // Disable domain selection and analysis saving until we are done correlating
  ui->action_Print->setEnabled(false);
  ui->action_Open->setEnabled(false);
  ui->action_Save_Analysis->setEnabled(false);
  ui->actionParame_ters->setEnabled(false);

  ui->QTabWidget_domain->setEnabled(false);
  ui->comboBox_correlation_model->setEnabled(false);
  ui->comboBox_Initial_Guess->setEnabled(false);
  ui->comboBox_update->setEnabled(false);
  ui->comboBox_color->setEnabled(false);
  ui->comboBox_interpolation_model->setEnabled(false);

  ui->spinBox_max_iters->setEnabled(false);
  ui->doubleSpinBox_precision->setEnabled(false);
  ui->spinBox_py_start->setEnabled(false);
  ui->spinBox_py_step->setEnabled(false);
  ui->spinBox_py_stop->setEnabled(false);

  und_imageLabel->inside_points.clear();
  und_imageLabel->contour_points.clear();
  und_imageLabel->selecting = false;
  und_imageLabel->correlating = true;
  und_imageLabel->update();

  def_imageLabel->inside_points.clear();
  def_imageLabel->contour_points.clear();
  def_imageLabel->selecting = false;
  def_imageLabel->selecting_ic = false;
  def_imageLabel->correlating = true;
  // def_imageLabel->update();

  ui->pushButton_correlate->setText("STOP");

  disconnect(ui->pushButton_correlate, SIGNAL(released()), this,
             SLOT(correlate()));
  connect(ui->pushButton_correlate, SIGNAL(released()), this,
          SLOT(stop_correlate()));

  // Set up correlation parameters - first domain independent parameters and
  // images
  manager->set_filenames(&fileNames);

  manager->set_color_mode(color_mode_iv, number_of_colors);
  manager->set_update(update_iv);
  manager->set_interpolation(interpolation_iv);
  manager->set_precision(precision);
  manager->set_max_iters(max_iters);
  manager->set_model(model_iv);
  manager->set_global_initial_guess(initial_guess);
  manager->set_deformation_description(deformation_description_iv);
  manager->set_error_handling_mode(error_handlingMode_iv);
  manager->set_plot_contour_points(plot_contour_points_iv);
  manager->set_plot_inside_points(plot_inside_points_iv);
  manager->set_referenceImage(reference_image_iv);
  manager->set_processor(processor_iv);
  manager->set_pyramid(py_start, py_step, py_stop);
  manager->stop_flag = false;

// set up cuda_manager
#if CUDA_ENABLED
  if (processor_iv == processor_GPU) {
    cuda_manager->set_fitting_model(model_iv);
    cuda_manager->set_interpolation_model(interpolation_iv);
    cuda_manager->set_precision(precision);
    cuda_manager->set_max_iters(max_iters);
    // cuda_manager->setDeformationDescription ( deformation_description_iv );
    // cuda_manager->setPlotInsidePoints       ( plot_inside_points_iv );
  }
#endif

  // connect(this, SIGNAL( start_correlation() ), manager, SLOT(
  // perform_multiframe_correlation() ) );
  emit start_correlation();
}

void MainApp::selectInitialGuess() {
  number_of_model_parameters =
      ModelClass::get_number_of_model_parameters(model_iv);

  // Initialize the ic archive
  if (!initial_guess_archive) {
    initial_guess_archive = new float *[fm_NUMBER_OF_ITEMS];
    for (int i = 0; i < fm_NUMBER_OF_ITEMS; ++i) {
      initial_guess_archive[i] = nullptr;
    }
  }

  // We already have an initial guess for this model
  if (initial_guess_archive[model_iv]) {
    initial_guess = initial_guess_archive[model_iv];
  }
  // We dont have an initial guess for this model
  // Lets create a NULL initial guess
  else {
    initial_guess_archive[model_iv] = new float[number_of_model_parameters];

    for (int i = 0; i < number_of_model_parameters; ++i) {
      initial_guess_archive[model_iv][i] = 0.f;
    }

    initial_guess = initial_guess_archive[model_iv];
  }

  if (initial_guess_iv == ic_Null) {
    for (int i = 0; i < number_of_model_parameters; ++i) {
      initial_guess_archive[model_iv][i] = 0.f;
    }

    initial_guess = initial_guess_archive[model_iv];
  }

  // Manager, und_imageLabel and def_imageLabel's initial guess are pointing to
  // MainApp's initial_guess
  def_imageLabel->model_parameters = initial_guess;
  und_imageLabel->model_parameters = initial_guess;

  updateResultDisplay();

  def_imageLabel->update();
}

void MainApp::correlation_done(bool error_in) {
  //  Error message to user
  if (error_in) {
    errorEnum errorType = manager->get_errorType();
    QString errorDisplay;

    switch (errorType) {
    case error_model_out_of_image:
      errorDisplay = "Mapping points out of the deformed image.";
      break;
    case error_interpolation_out_of_image:
      errorDisplay = "Interpolating out of the deformed image.";
      break;
    case error_correlation_max_iters_reached:
      errorDisplay = "Maximun iterations reached.";
      break;
    case error_bad_domain:
      errorDisplay = "Domain has less than 3 points or crosses itself.";
      break;
    case error_cuSolver:
      errorDisplay = "cuSolver error.";
      break;
    case error_cuda:
      errorDisplay = "cuda error.";
      break;
    case error_multiThread:
      errorDisplay = "Error getting the next image.";
      break;
    case error_none:
      errorDisplay = "Unknown error.";
      break;
    default:
      printf("errorType = %d\n", errorType);
      assert(false);
      break;
    }

    QMessageBox::information(this, tr("MainWindow"), errorDisplay);
  }

  // Once the correlation is done, re-enable the domain selection and saving
  // analysis
  ui->action_Print->setEnabled(true);
  ui->action_Open->setEnabled(true);
  ui->action_Save_Analysis->setEnabled(true);
  ui->actionParame_ters->setEnabled(true);

  ui->QTabWidget_domain->setEnabled(true);
  ui->comboBox_correlation_model->setEnabled(true);
  ui->comboBox_Initial_Guess->setEnabled(true);
  ui->comboBox_update->setEnabled(true);
  ui->comboBox_color->setEnabled(true);
  ui->comboBox_interpolation_model->setEnabled(true);
  ui->pushButton_correlate->setText("Correlate");

  ui->spinBox_max_iters->setEnabled(true);
  ui->doubleSpinBox_precision->setEnabled(true);
  ui->spinBox_py_start->setEnabled(true);
  ui->spinBox_py_step->setEnabled(true);
  ui->spinBox_py_stop->setEnabled(true);

  und_imageLabel->selecting = true;
  und_imageLabel->suppress_selection_display = true;
  def_imageLabel->selecting = true;
  def_imageLabel->suppress_selection_display = true;

  if (initial_guess_iv == ic_User) {
    def_imageLabel->selecting_ic = true;
  } else {
    def_imageLabel->selecting_ic = false;
  }

  disconnect(ui->pushButton_correlate, SIGNAL(released()), this,
             SLOT(stop_correlate()));
  connect(ui->pushButton_correlate, SIGNAL(released()), this,
          SLOT(correlate()));

  updateGpuPyramids();
  loadNxtGpuImages();

  makeAnalysisReport();

#if AUTO_PILOT
  close();
#endif
}

void MainApp::makeAnalysisReport() {}

void MainApp::saveAnalysisReport() {
  reportFile = QFileDialog::getSaveFileName(
      this, tr("Save Report to File"), "/home/namascar/Documents/data/reports");
  report.open(reportFile.toStdString());
  report << manager->report.rdbuf();
  report.close();
}

void MainApp::stop_correlate() { manager->stop_flag = true; }

void MainApp::stop_correlation_display() {
  und_imageLabel->correlating = false;
  und_imageLabel->clear_inside_points();
  und_imageLabel->clear_contour_points();
  und_imageLabel->suppress_selection_display = false;
  und_imageLabel->update();

  def_imageLabel->correlating = false;
  def_imageLabel->clear_inside_points();
  def_imageLabel->clear_contour_points();
  def_imageLabel->suppress_selection_display = false;
}

void MainApp::updateResultDisplay() {
  /**
   Method is called from within mainApp (without parameters) to indicate a new
   set of initial guess is available. It displays
   the new initial guess above the imageLabel.
   */

  QString result_text, best_rotation;
  float angle;

  for (int i = 0; i < number_of_model_parameters; ++i) {
    result_text.append(QString::number(initial_guess[i]));
    result_text.append("    ");
  }
  ui->label_results_display->setText(result_text);

  switch (model_iv) {
  case fm_U:

  case fm_UV:

    angle = 0.f;

    break;

  case fm_UVQ:

    angle = initial_guess[2];

    break;

  case fm_UVUxUyVxVy:

    angle = 180.f / PI * best_rotation_UVUxUyVxVy(initial_guess);

    break;

  default:

    assert(false);

    break;
  }

  best_rotation.setNum(angle);

  ui->label_best_rotation_value->setText(best_rotation);
}

void MainApp::updateResultDisplay(float angle, float *result_parameters) {
  QString result_text, best_rotation;

  for (int i = 0; i < number_of_model_parameters; ++i) {
    result_text.append(QString::number(result_parameters[i]));
    result_text.append("    ");
  }
  ui->label_results_display->setText(result_text);

  best_rotation.setNum(angle * 180.f / PI);

  ui->label_best_rotation_value->setText(best_rotation);
}

void MainApp::initialize_domains() {
  rectangularDomain.ready = false;
  rectangularDomain.x_center = 0;
  rectangularDomain.y_center = 0;
  rectangularDomain.x_begin = 0;
  rectangularDomain.y_begin = 0;
  rectangularDomain.x_end = 0;
  rectangularDomain.y_end = 0;
  rectangularDomain.horizontal_subdivisions = 1;
  rectangularDomain.vertical_subdivisions = 1;

  ui->doubleSpinBox_rect_x0->setValue(rectangularDomain.x_begin);
  ui->doubleSpinBox_rect_y0->setValue(rectangularDomain.y_begin);
  ui->doubleSpinBox_rect_xend->setValue(rectangularDomain.x_end);
  ui->doubleSpinBox_rect_yend->setValue(rectangularDomain.y_end);
  ui->spinBox_rect_horizontal->setValue(
      rectangularDomain.horizontal_subdivisions);
  ui->spinBox_rect_vertical->setValue(rectangularDomain.vertical_subdivisions);
  und_imageLabel->h_subdivisions = rectangularDomain.horizontal_subdivisions;
  def_imageLabel->h_subdivisions = rectangularDomain.horizontal_subdivisions;
  und_imageLabel->v_subdivisions = rectangularDomain.vertical_subdivisions;
  def_imageLabel->v_subdivisions = rectangularDomain.vertical_subdivisions;

  //  Draw the initial annular domain outside the image. Setting ri = ro = 0
  //  makes
  //      ri_by_ro undefined in ImageLabel otherwise
  annularDomain.ready = false;
  annularDomain.x_center = -2.f;
  annularDomain.y_center = -2.f;
  annularDomain.r_inside = 0.5f;
  annularDomain.r_outside = 1;
  annularDomain.ri_by_ro = 0.5f;
  annularDomain.radial_subdivisions = 1;
  annularDomain.angular_subdivisions = 1;

  ui->doubleSpinBox_ann_center_x->setValue(annularDomain.x_center);
  ui->doubleSpinBox_ann_center_y->setValue(annularDomain.y_center);
  ui->doubleSpinBox_ann_ri->setValue(annularDomain.r_inside);
  ui->doubleSpinBox_ann_ro->setValue(annularDomain.r_outside);
  ui->spinBox_ann_radial->setValue(annularDomain.radial_subdivisions);
  ui->spinBox_ann_angular->setValue(annularDomain.angular_subdivisions);
  und_imageLabel->r_subdivisions = annularDomain.radial_subdivisions;
  def_imageLabel->r_subdivisions = annularDomain.radial_subdivisions;
  und_imageLabel->a_subdivisions = annularDomain.angular_subdivisions;
  def_imageLabel->a_subdivisions = annularDomain.angular_subdivisions;
  und_imageLabel->ri_by_ro = annularDomain.ri_by_ro;
  def_imageLabel->ri_by_ro = annularDomain.ri_by_ro;

  blobDomain.ready = false;
  blobDomain.x_center = 0;
  blobDomain.y_center = 0;
  blobDomain.x_scale = 1.f;
  blobDomain.y_scale = 1.f;
  blobDomain.xy_contour.clear();

  ui->doubleSpinBox_blob_center_x->setValue(blobDomain.x_center);
  ui->doubleSpinBox_blob_center_y->setValue(blobDomain.y_center);
  ui->doubleSpinBox_blob_x_scale->setValue(blobDomain.x_scale);
  ui->doubleSpinBox_blob_y_scale->setValue(blobDomain.y_scale);
}
