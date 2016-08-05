// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "ui/main_window.h"

#include <QtConcurrent/QtConcurrentRun>

namespace colmap {

MainWindow::MainWindow(const OptionManager& options)
    : options_(options),
      import_watcher_(nullptr),
      export_watcher_(nullptr),
      window_closed_(false) {
  resize(1024, 600);
  UpdateWindowTitle();

  CreateWidgets();
  CreateActions();
  CreateMenus();
  CreateToolbar();
  CreateStatusbar();
  CreateControllers();
  CreateFutures();
  CreateProgressBar();

  ShowLog();

  options_.AddAllOptions();
}

bool MainWindow::OverwriteReconstruction() {
  if (mapper_controller->NumModels() > 0) {
    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "",
        tr("Do you really want to overwrite the existing reconstruction?"),
        QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::No) {
      return false;
    } else {
      ReconstructionReset();
      log_widget_->Clear();
    }
  }
  return true;
}

void MainWindow::showEvent(QShowEvent* event) {
  after_show_event_timer_ = new QTimer(this);
  connect(after_show_event_timer_, &QTimer::timeout, this,
          &MainWindow::afterShowEvent);
  after_show_event_timer_->start(100);
}

void MainWindow::moveEvent(QMoveEvent* event) { CenterProgressBar(); }

void MainWindow::closeEvent(QCloseEvent* event) {
  if (window_closed_) {
    event->accept();
    return;
  }

  if (new_project_widget_->IsValid() && *options_.project_path == "") {
    // Project was created, but not yet saved
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(
        this, "",
        tr("You have not saved your project. Do you want to save it?"),
        QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::Yes) {
      SaveProject();
    }
  }

  QMessageBox::StandardButton reply;
  reply = QMessageBox::question(this, "", tr("Do you really want to quit?"),
                                QMessageBox::Yes | QMessageBox::No);
  if (reply == QMessageBox::No) {
    event->ignore();
  } else {
    mapper_controller->Stop();
    mapper_controller->Wait();
    ba_controller->Stop();
    ba_controller->Wait();

    log_widget_->close();
    event->accept();

    window_closed_ = true;
  }
}

void MainWindow::afterShowEvent() {
  after_show_event_timer_->stop();
  CenterProgressBar();
}

void MainWindow::CreateWidgets() {
  opengl_window_ = new OpenGLWindow(this, &options_);
  setCentralWidget(QWidget::createWindowContainer(opengl_window_));

  new_project_widget_ = new NewProjectWidget(this, &options_);
  new_project_widget_->SetDatabasePath(*options_.database_path);
  new_project_widget_->SetImagePath(*options_.image_path);

  feature_extraction_widget_ = new FeatureExtractionWidget(this, &options_);
  feature_matching_widget_ = new FeatureMatchingWidget(this, &options_);
  database_management_widget_ = new DatabaseManagementWidget(this, &options_);
  reconstruction_options_widget_ =
      new ReconstructionOptionsWidget(this, &options_);
  bundle_adjustment_options_widget_ =
      new BundleAdjustmentOptionsWidget(this, &options_);
  render_options_widget_ =
      new RenderOptionsWidget(this, &options_, opengl_window_);
  log_widget_ = new LogWidget(this, &options_);
  undistort_widget_ = new UndistortWidget(this, &options_);
  model_manager_widget_ = new ModelManagerWidget(this);
  model_stats_widget_ = new ModelStatsWidget(this);
  match_matrix_widget_ = new MatchMatrixWidget(this, &options_);

  dock_log_widget_ = new QDockWidget("Log", this);
  dock_log_widget_->setWidget(log_widget_);
  addDockWidget(Qt::RightDockWidgetArea, dock_log_widget_);
}

void MainWindow::CreateActions() {
  //////////////////////////////////////////////////////////////////////////////
  // File actions
  //////////////////////////////////////////////////////////////////////////////

  action_new_project_ =
      new QAction(QIcon(":/media/project-new.png"), tr("New project"), this);
  action_new_project_->setShortcuts(QKeySequence::New);
  connect(action_new_project_, &QAction::triggered, this,
          &MainWindow::NewProject);

  action_open_project_ =
      new QAction(QIcon(":/media/project-open.png"), tr("Open project"), this);
  action_open_project_->setShortcuts(QKeySequence::Open);
  connect(action_open_project_, &QAction::triggered, this,
          &MainWindow::OpenProject);

  action_save_project_ =
      new QAction(QIcon(":/media/project-save.png"), tr("Save project"), this);
  action_save_project_->setShortcuts(QKeySequence::Save);
  connect(action_save_project_, &QAction::triggered, this,
          &MainWindow::SaveProject);

  action_save_project_as_ = new QAction(QIcon(":/media/project-save-as.png"),
                                        tr("Save project as..."), this);
  action_save_project_as_->setShortcuts(QKeySequence::SaveAs);
  connect(action_save_project_as_, &QAction::triggered, this,
          &MainWindow::SaveProjectAs);

  action_import_ =
      new QAction(QIcon(":/media/import.png"), tr("Import model"), this);
  connect(action_import_, &QAction::triggered, this, &MainWindow::Import);
  blocking_actions_.push_back(action_import_);

  action_import_from_ = new QAction(QIcon(":/media/import-from.png"),
                                    tr("Import model from..."), this);
  connect(action_import_from_, &QAction::triggered, this,
          &MainWindow::ImportFrom);
  blocking_actions_.push_back(action_import_from_);

  action_export_ =
      new QAction(QIcon(":/media/export.png"), tr("Export model"), this);
  connect(action_export_, &QAction::triggered, this, &MainWindow::Export);
  blocking_actions_.push_back(action_export_);

  action_export_all_ = new QAction(QIcon(":/media/export-all.png"),
                                   tr("Export all models"), this);
  connect(action_export_all_, &QAction::triggered, this,
          &MainWindow::ExportAll);
  blocking_actions_.push_back(action_export_all_);

  action_export_as_ = new QAction(QIcon(":/media/export-as.png"),
                                  tr("Export model as..."), this);
  connect(action_export_as_, &QAction::triggered, this, &MainWindow::ExportAs);
  blocking_actions_.push_back(action_export_as_);

  action_quit_ = new QAction(tr("Quit"), this);
  connect(action_quit_, &QAction::triggered, this, &MainWindow::close);

  //////////////////////////////////////////////////////////////////////////////
  // Processing action
  //////////////////////////////////////////////////////////////////////////////

  action_feature_extraction_ = new QAction(
      QIcon(":/media/feature-extraction.png"), tr("Extract features"), this);
  connect(action_feature_extraction_, &QAction::triggered, this,
          &MainWindow::FeatureExtraction);
  blocking_actions_.push_back(action_feature_extraction_);

  action_feature_matching_ = new QAction(QIcon(":/media/feature-matching.png"),
                                         tr("Match features"), this);
  connect(action_feature_matching_, &QAction::triggered, this,
          &MainWindow::FeatureMatching);
  blocking_actions_.push_back(action_feature_matching_);

  action_database_management_ = new QAction(
      QIcon(":/media/database-management.png"), tr("Manage database"), this);
  connect(action_database_management_, &QAction::triggered, this,
          &MainWindow::DatabaseManagement);
  blocking_actions_.push_back(action_database_management_);

  //////////////////////////////////////////////////////////////////////////////
  // Reconstruction actions
  //////////////////////////////////////////////////////////////////////////////

  action_reconstruction_start_ =
      new QAction(QIcon(":/media/reconstruction-start.png"),
                  tr("Start / resume reconstruction"), this);
  connect(action_reconstruction_start_, &QAction::triggered, this,
          &MainWindow::ReconstructionStart);
  blocking_actions_.push_back(action_reconstruction_start_);

  action_reconstruction_step_ =
      new QAction(QIcon(":/media/reconstruction-step.png"),
                  tr("Reconstruct next image"), this);
  connect(action_reconstruction_step_, &QAction::triggered, this,
          &MainWindow::ReconstructionStep);
  blocking_actions_.push_back(action_reconstruction_step_);

  action_reconstruction_pause_ =
      new QAction(QIcon(":/media/reconstruction-pause.png"),
                  tr("Pause reconstruction"), this);
  connect(action_reconstruction_pause_, &QAction::triggered, this,
          &MainWindow::ReconstructionPause);
  action_reconstruction_pause_->setEnabled(false);
  blocking_actions_.push_back(action_reconstruction_pause_);

  action_reconstruction_reset_ =
      new QAction(QIcon(":/media/reconstruction-reset.png"),
                  tr("Reset reconstruction"), this);
  connect(action_reconstruction_reset_, &QAction::triggered, this,
          &MainWindow::OverwriteReconstruction);

  action_reconstruction_normalize_ =
      new QAction(QIcon(":/media/reconstruction-normalize.png"),
                  tr("Normalize reconstruction"), this);
  connect(action_reconstruction_normalize_, &QAction::triggered, this,
          &MainWindow::ReconstructionNormalize);
  blocking_actions_.push_back(action_reconstruction_normalize_);

  action_reconstruction_options_ =
      new QAction(QIcon(":/media/reconstruction-options.png"),
                  tr("Reconstruction options"), this);
  connect(action_reconstruction_options_, &QAction::triggered, this,
          &MainWindow::ReconstructionOptions);
  blocking_actions_.push_back(action_reconstruction_options_);

  action_bundle_adjustment_ = new QAction(
      QIcon(":/media/bundle-adjustment.png"), tr("Bundle adjustment"), this);
  connect(action_bundle_adjustment_, &QAction::triggered, this,
          &MainWindow::BundleAdjustment);
  action_bundle_adjustment_->setEnabled(false);
  blocking_actions_.push_back(action_bundle_adjustment_);

  action_bundle_adjustment_options_ =
      new QAction(QIcon(":/media/bundle-adjustment-options.png"),
                  tr("Bundle adjustment options"), this);
  connect(action_bundle_adjustment_options_, &QAction::triggered, this,
          &MainWindow::BundleAdjustmentOptions);
  blocking_actions_.push_back(action_bundle_adjustment_options_);

  //////////////////////////////////////////////////////////////////////////////
  // Render actions
  //////////////////////////////////////////////////////////////////////////////

  action_render_toggle_ = new QAction(QIcon(":/media/render-enabled.png"),
                                      tr("Disable rendering"), this);
  connect(action_render_toggle_, &QAction::triggered, this,
          &MainWindow::RenderToggle);

  action_render_reset_view_ = new QAction(
      QIcon(":/media/render-reset-view.png"), tr("Reset view"), this);
  connect(action_render_reset_view_, &QAction::triggered, opengl_window_,
          &OpenGLWindow::ResetView);

  action_render_options_ = new QAction(QIcon(":/media/render-options.png"),
                                       tr("Render options"), this);
  connect(action_render_options_, &QAction::triggered, this,
          &MainWindow::RenderOptions);

  connect(model_manager_widget_, static_cast<void (QComboBox::*)(int)>(
                                     &QComboBox::currentIndexChanged),
          this, &MainWindow::SelectModelIdx);

  //////////////////////////////////////////////////////////////////////////////
  // Extras actions
  //////////////////////////////////////////////////////////////////////////////

  action_model_stats_ = new QAction(QIcon(":/media/model-stats.png"),
                                    tr("Show model statistics"), this);
  connect(action_model_stats_, &QAction::triggered, this,
          &MainWindow::ShowModelStats);

  action_match_matrix_ = new QAction(QIcon(":/media/match-matrix.png"),
                                     tr("Show match matrix"), this);
  connect(action_match_matrix_, &QAction::triggered, this,
          &MainWindow::ShowMatchMatrix);

  action_log_show_ =
      new QAction(QIcon(":/media/log.png"), tr("Show log"), this);
  connect(action_log_show_, &QAction::triggered, this, &MainWindow::ShowLog);

  action_grab_image_ =
      new QAction(QIcon(":/media/grab-image.png"), tr("Grab image"), this);
  connect(action_grab_image_, &QAction::triggered, this,
          &MainWindow::GrabImage);

  action_grab_movie_ =
      new QAction(QIcon(":/media/grab-movie.png"), tr("Grab movie"), this);
  connect(action_grab_movie_, &QAction::triggered, opengl_window_,
          &OpenGLWindow::GrabMovie);

  action_undistort_ =
      new QAction(QIcon(":/media/undistort.png"), tr("Undistort images"), this);
  connect(action_undistort_, &QAction::triggered, this,
          &MainWindow::UndistortImages);
  blocking_actions_.push_back(action_undistort_);

  action_extract_colors_ = new QAction(tr("Extract colors"), this);
  connect(action_extract_colors_, &QAction::triggered, this,
          &MainWindow::ExtractColors);

  action_reset_options_ = new QAction(tr("Restore default options"), this);
  connect(action_reset_options_, &QAction::triggered, this,
          &MainWindow::ResetOptions);

  //////////////////////////////////////////////////////////////////////////////
  // Misc actions
  //////////////////////////////////////////////////////////////////////////////

  action_render_ = new QAction(tr("Render"), this);
  connect(action_render_, &QAction::triggered, this, &MainWindow::Render,
          Qt::BlockingQueuedConnection);

  action_render_now_ = new QAction(tr("Render now"), this);
  render_options_widget_->action_render_now = action_render_now_;
  connect(action_render_now_, &QAction::triggered, this, &MainWindow::RenderNow,
          Qt::BlockingQueuedConnection);

  action_reconstruction_finish_ =
      new QAction(tr("Finish reconstruction"), this);
  connect(action_reconstruction_finish_, &QAction::triggered, this,
          &MainWindow::ReconstructionFinish, Qt::BlockingQueuedConnection);

  action_bundle_adjustment_finish_ =
      new QAction(tr("Finish bundle-adjustment"), this);
  connect(action_bundle_adjustment_finish_, &QAction::triggered, this,
          &MainWindow::BundleAdjustmentFinish);

  action_about_ = new QAction(tr("About"), this);
  connect(action_about_, &QAction::triggered, this, &MainWindow::About);
  action_license_ = new QAction(tr("License"), this);
  connect(action_license_, &QAction::triggered, this, &MainWindow::License);
}

void MainWindow::CreateMenus() {
  QMenu* file_menu = new QMenu(tr("File"), this);
  file_menu->addAction(action_new_project_);
  file_menu->addAction(action_open_project_);
  file_menu->addAction(action_save_project_);
  file_menu->addAction(action_save_project_as_);
  file_menu->addAction(action_import_);
  file_menu->addAction(action_import_from_);
  file_menu->addAction(action_export_);
  file_menu->addAction(action_export_all_);
  file_menu->addAction(action_export_as_);
  file_menu->addSeparator();
  file_menu->addAction(action_quit_);
  menuBar()->addAction(file_menu->menuAction());

  QMenu* preprocessing_menu = new QMenu(tr("Processing"), this);
  preprocessing_menu->addAction(action_feature_extraction_);
  preprocessing_menu->addAction(action_feature_matching_);
  preprocessing_menu->addAction(action_database_management_);
  menuBar()->addAction(preprocessing_menu->menuAction());

  QMenu* reconstruction_menu = new QMenu(tr("Reconstruction"), this);
  reconstruction_menu->addAction(action_reconstruction_start_);
  reconstruction_menu->addAction(action_reconstruction_pause_);
  reconstruction_menu->addAction(action_reconstruction_step_);
  reconstruction_menu->addAction(action_reconstruction_reset_);
  reconstruction_menu->addAction(action_reconstruction_normalize_);
  reconstruction_menu->addAction(action_reconstruction_options_);
  reconstruction_menu->addAction(action_bundle_adjustment_);
  reconstruction_menu->addAction(action_bundle_adjustment_options_);
  menuBar()->addAction(reconstruction_menu->menuAction());

  QMenu* render_menu = new QMenu(tr("Render"), this);
  render_menu->addAction(action_render_toggle_);
  render_menu->addAction(action_render_reset_view_);
  render_menu->addAction(action_render_options_);
  menuBar()->addAction(render_menu->menuAction());

  QMenu* extras_menu = new QMenu(tr("Extras"), this);
  extras_menu->addAction(action_model_stats_);
  extras_menu->addAction(action_match_matrix_);
  extras_menu->addAction(action_log_show_);
  extras_menu->addAction(action_grab_image_);
  extras_menu->addAction(action_grab_movie_);
  extras_menu->addAction(action_undistort_);
  extras_menu->addAction(action_extract_colors_);
  extras_menu->addAction(action_reset_options_);
  menuBar()->addAction(extras_menu->menuAction());

  QMenu* help_menu = new QMenu(tr("Help"), this);
  help_menu->addAction(action_about_);
  help_menu->addAction(action_license_);
  menuBar()->addAction(help_menu->menuAction());

  // TODO: Make the native menu bar work on OSX. Simply setting this to true
  // will result in a menubar which is not clickable until the main window is
  // defocused and refocused.
  menuBar()->setNativeMenuBar(false);
}

void MainWindow::CreateToolbar() {
  file_toolbar_ = addToolBar(tr("File"));
  file_toolbar_->addAction(action_new_project_);
  file_toolbar_->addAction(action_open_project_);
  file_toolbar_->addAction(action_save_project_);
  file_toolbar_->addAction(action_import_);
  file_toolbar_->addAction(action_export_);
  file_toolbar_->setIconSize(QSize(16, 16));

  preprocessing_toolbar_ = addToolBar(tr("Processing"));
  preprocessing_toolbar_->addAction(action_feature_extraction_);
  preprocessing_toolbar_->addAction(action_feature_matching_);
  preprocessing_toolbar_->addAction(action_database_management_);
  preprocessing_toolbar_->setIconSize(QSize(16, 16));

  reconstruction_toolbar_ = addToolBar(tr("Reconstruction"));
  reconstruction_toolbar_->addAction(action_reconstruction_start_);
  reconstruction_toolbar_->addAction(action_reconstruction_step_);
  reconstruction_toolbar_->addAction(action_reconstruction_pause_);
  reconstruction_toolbar_->addAction(action_reconstruction_normalize_);
  reconstruction_toolbar_->addAction(action_reconstruction_options_);
  reconstruction_toolbar_->addAction(action_bundle_adjustment_);
  reconstruction_toolbar_->addAction(action_bundle_adjustment_options_);
  reconstruction_toolbar_->setIconSize(QSize(16, 16));

  render_toolbar_ = addToolBar(tr("Render"));
  render_toolbar_->addAction(action_render_toggle_);
  render_toolbar_->addAction(action_render_reset_view_);
  render_toolbar_->addAction(action_render_options_);
  render_toolbar_->addWidget(model_manager_widget_);
  render_toolbar_->setIconSize(QSize(16, 16));

  extras_toolbar_ = addToolBar(tr("Extras"));
  extras_toolbar_->addAction(action_model_stats_);
  extras_toolbar_->addAction(action_match_matrix_);
  extras_toolbar_->addAction(action_log_show_);
  extras_toolbar_->addAction(action_grab_image_);
  extras_toolbar_->addAction(action_grab_movie_);
  extras_toolbar_->setIconSize(QSize(16, 16));
}

void MainWindow::CreateStatusbar() {
  QFont font;
  font.setPointSize(11);

  statusbar_timer_label_ = new QLabel("Time 00:00:00:00", this);
  statusbar_timer_label_->setFont(font);
  statusbar_timer_label_->setAlignment(Qt::AlignCenter);
  statusBar()->addWidget(statusbar_timer_label_, 1);
  statusbar_timer_ = new QTimer(this);
  connect(statusbar_timer_, &QTimer::timeout, this, &MainWindow::UpdateTimer);
  statusbar_timer_->start(1000);

  opengl_window_->statusbar_status_label =
      new QLabel("0 Images - 0 Points", this);
  opengl_window_->statusbar_status_label->setFont(font);
  opengl_window_->statusbar_status_label->setAlignment(Qt::AlignCenter);
  statusBar()->addWidget(opengl_window_->statusbar_status_label, 1);
}

void MainWindow::CreateControllers() {
  if (mapper_controller) {
    mapper_controller->Stop();
    mapper_controller->Wait();
  }

  mapper_controller.reset(new IncrementalMapperController(options_));
  mapper_controller->SetCallback("InitialImagePairRegistered", [this]() {
    if (!mapper_controller->IsStopped()) {
      action_render_now_->trigger();
    }
  });
  mapper_controller->SetCallback("NextImageRegistered", [this]() {
    if (!mapper_controller->IsStopped()) {
      action_render_->trigger();
    }
  });
  mapper_controller->SetCallback("LastImageRegistered", [this]() {
    if (!mapper_controller->IsStopped()) {
      action_render_now_->trigger();
    }
  });
  mapper_controller->SetCallback("Finished", [this]() {
    if (!mapper_controller->IsStopped()) {
      action_render_now_->trigger();
      action_reconstruction_finish_->trigger();
    }
  });

  if (ba_controller) {
    ba_controller->Stop();
    ba_controller->Wait();
  }

  ba_controller.reset(new BundleAdjustmentController(options_));
  ba_controller->SetCallback(
      "Finished", [this]() { action_bundle_adjustment_finish_->trigger(); });
}

void MainWindow::CreateFutures() {
  import_watcher_ = new QFutureWatcher<void>(this);
  connect(import_watcher_, &QFutureWatcher<void>::finished, this,
          &MainWindow::ImportFinished);

  export_watcher_ = new QFutureWatcher<void>(this);
  connect(export_watcher_, &QFutureWatcher<void>::finished, this,
          &MainWindow::ExportFinished);

  extract_colors_watcher_ = new QFutureWatcher<void>(this);
  connect(extract_colors_watcher_, &QFutureWatcher<void>::finished, this,
          &MainWindow::ExtractColorsFinished);
}

void MainWindow::CreateProgressBar() {
  progress_bar_ = new QProgressDialog(this);
  progress_bar_->setWindowModality(Qt::ApplicationModal);
  progress_bar_->setWindowFlags(Qt::Popup);
  progress_bar_->setCancelButton(nullptr);
  progress_bar_->setMaximum(0);
  progress_bar_->setMinimum(0);
  progress_bar_->setValue(0);
  progress_bar_->hide();
  progress_bar_->close();
}

void MainWindow::CenterProgressBar() {
  const QPoint global = mapToGlobal(rect().center());
  progress_bar_->move(global.x() - progress_bar_->width() / 2,
                      global.y() - progress_bar_->height() / 2);
}

void MainWindow::NewProject() {
  new_project_widget_->show();
  new_project_widget_->raise();
}

bool MainWindow::OpenProject() {
  if (!OverwriteReconstruction()) {
    return false;
  }

  const std::string project_path =
      QFileDialog::getOpenFileName(this, tr("Select project file"), "",
                                   tr("Project file (*.ini)"))
          .toUtf8()
          .constData();
  // If selection not canceled
  if (project_path != "") {
    if (options_.ReRead(project_path)) {
      *options_.project_path = project_path;
      new_project_widget_->SetDatabasePath(*options_.database_path);
      new_project_widget_->SetImagePath(*options_.image_path);
      UpdateWindowTitle();
      return true;
    }
  }

  return false;
}

void MainWindow::SaveProject() {
  if (!boost::filesystem::is_regular_file(*options_.project_path)) {
    std::string project_path =
        QFileDialog::getSaveFileName(this, tr("Select project file"), "",
                                     tr("Project file (*.ini)"))
            .toUtf8()
            .constData();
    // If selection not canceled
    if (project_path != "") {
      if (!HasFileExtension(project_path, ".ini")) {
        project_path += ".ini";
      }
      *options_.project_path = project_path;
      options_.Write(*options_.project_path);
    }
  } else {
    // Project path was chosen previously, either here or via command-line.
    options_.Write(*options_.project_path);
  }

  UpdateWindowTitle();
}

void MainWindow::SaveProjectAs() {
  const std::string new_project_path =
      QFileDialog::getSaveFileName(this, tr("Select project file"), "",
                                   tr("Project file (*.ini)"))
          .toUtf8()
          .constData();
  if (new_project_path != "") {
    *options_.project_path = new_project_path;
    options_.Write(*options_.project_path);
  }

  UpdateWindowTitle();
}

void MainWindow::Import() {
  if (!OverwriteReconstruction()) {
    return;
  }

  std::string path =
      QFileDialog::getExistingDirectory(this, tr("Select source..."), "",
                                        QFileDialog::ShowDirsOnly)
          .toUtf8()
          .constData();

  // Selection canceled?
  if (path == "") {
    return;
  }

  path = EnsureTrailingSlash(path);

  const std::string project_path = path + "project.ini";
  const std::string cameras_path = path + "cameras.txt";
  const std::string images_path = path + "images.txt";
  const std::string points3D_path = path + "points3D.txt";

  if (!boost::filesystem::is_regular_file(cameras_path) ||
      !boost::filesystem::is_regular_file(images_path) ||
      !boost::filesystem::is_regular_file(points3D_path)) {
    QMessageBox::critical(this, "",
                          tr("`cameras.txt`, `images.txt` and "
                             "`points3D.txt` must exist in the chosen "
                             "directory."));
    return;
  }

  if (!boost::filesystem::is_regular_file(project_path)) {
    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "", tr("Directory does not contain a `project.ini`. In order to "
                     "resume the reconstruction you need to specify a valid "
                     "database and image path. Do you want to select the paths "
                     "now (or press No to only load and visualize the model)?"),
        QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::Yes) {
      OpenProject();
    }
  } else {
    if (options_.ReRead(project_path)) {
      ReconstructionReset();
    }
  }

  progress_bar_->setLabelText(tr("Importing model"));
  progress_bar_->raise();
  progress_bar_->show();

  import_watcher_->setFuture(QtConcurrent::run([this, path]() {
    const size_t model_idx = this->mapper_controller->AddModel();
    this->mapper_controller->Model(model_idx).Read(path);
    model_manager_widget_->UpdateModels(mapper_controller->Models());
    model_manager_widget_->SetModelIdx(model_idx);
    action_bundle_adjustment_->setEnabled(true);
  }));
}

void MainWindow::ImportFrom() {
  if (!OverwriteReconstruction()) {
    return;
  }

  std::string path =
      QFileDialog::getOpenFileName(this, tr("Select source..."), "")
          .toUtf8()
          .constData();

  // Selection canceled?
  if (path == "") {
    return;
  }

  if (!boost::filesystem::is_regular_file(path)) {
    QMessageBox::critical(this, "", tr("Invalid file"));
    return;
  }

  if (!HasFileExtension(path, ".ply")) {
    QMessageBox::critical(this, "",
                          tr("Invalid file format (supported formats: PLY)"));
    return;
  }

  progress_bar_->setLabelText(tr("Importing model"));
  progress_bar_->raise();
  progress_bar_->show();

  import_watcher_->setFuture(QtConcurrent::run([this, path]() {
    const size_t model_idx = this->mapper_controller->AddModel();
    this->mapper_controller->Model(model_idx).ImportPLY(path);
    this->options_.render_options->min_track_len = 0;
    model_manager_widget_->UpdateModels(mapper_controller->Models());
    model_manager_widget_->SetModelIdx(model_idx);
    action_bundle_adjustment_->setEnabled(true);
  }));
}

void MainWindow::ImportFinished() {
  RenderSelectedModel();
  progress_bar_->hide();
}

void MainWindow::Export() {
  if (!IsSelectedModelValid()) {
    return;
  }

  std::string path =
      QFileDialog::getExistingDirectory(this, tr("Select destination..."), "",
                                        QFileDialog::ShowDirsOnly)
          .toUtf8()
          .constData();

  // Selection canceled?
  if (path == "") {
    return;
  }

  path = EnsureTrailingSlash(path);

  const std::string project_path = path + "project.ini";
  const std::string cameras_path = path + "cameras.txt";
  const std::string images_path = path + "images.txt";
  const std::string points3D_path = path + "points3D.txt";
  const std::string images_vrml_path = path + "images.wrl";
  const std::string points3D_vrml_path = path + "points3D.wrl";

  if (boost::filesystem::is_regular_file(project_path) ||
      boost::filesystem::is_regular_file(cameras_path) ||
      boost::filesystem::is_regular_file(images_path) ||
      boost::filesystem::is_regular_file(points3D_path) ||
      boost::filesystem::is_regular_file(images_vrml_path) ||
      boost::filesystem::is_regular_file(points3D_vrml_path)) {
    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "", tr("The files `cameras.txt`, `images.txt`, `points3D.txt`, "
                     "`images.wrl`, and `points3D.wrl` already exist in the "
                     "selected destination. Do you want to overwrite them?"),
        QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::No) {
      return;
    }
  }

  progress_bar_->setLabelText(tr("Exporting model"));
  progress_bar_->raise();
  progress_bar_->show();

  export_watcher_->setFuture(QtConcurrent::run([this, path, project_path]() {
    this->options_.Write(project_path);
    this->mapper_controller->Model(SelectedModelIdx()).Write(path);
  }));
}

void MainWindow::ExportAll() {
  if (!IsSelectedModelValid()) {
    return;
  }

  std::string path =
      QFileDialog::getExistingDirectory(this, tr("Select destination..."), "",
                                        QFileDialog::ShowDirsOnly)
          .toUtf8()
          .constData();

  // Selection canceled?
  if (path == "") {
    return;
  }

  path = EnsureTrailingSlash(path);

  progress_bar_->setLabelText(tr("Exporting models"));
  progress_bar_->raise();
  progress_bar_->show();

  export_watcher_->setFuture(QtConcurrent::run([this, path]() {
    for (size_t i = 0; i < this->mapper_controller->NumModels(); ++i) {
      const std::string model_path = path + std::to_string(i);

      if (!boost::filesystem::is_directory(model_path)) {
        boost::filesystem::create_directory(model_path);
      }

      this->options_.Write(model_path + "/project.ini");
      this->mapper_controller->Model(i).Write(model_path);
    }
  }));
}

void MainWindow::ExportAs() {
  if (!IsSelectedModelValid()) {
    return;
  }

  QString default_filter("NVM (*.nvm)");
  const std::string path =
      QFileDialog::getSaveFileName(
          this, tr("Select project file"), "",
          "NVM (*.nvm);;Bundler (*.out);;PLY (*.ply);;VRML (*.wrl)",
          &default_filter)
          .toUtf8()
          .constData();

  // Selection canceled?
  if (path == "") {
    return;
  }

  progress_bar_->setLabelText(tr("Exporting model"));
  progress_bar_->raise();
  progress_bar_->show();

  export_watcher_->setFuture(QtConcurrent::run([this, path, default_filter]() {
    const Reconstruction& model = mapper_controller->Model(SelectedModelIdx());
    try {
      if (default_filter == "NVM (*.nvm)") {
        model.ExportNVM(path);
      } else if (default_filter == "Bundler (*.out)") {
        model.ExportBundler(path, path + ".list.txt");
      } else if (default_filter == "PLY (*.ply)") {
        model.ExportPLY(path);
      } else if (default_filter == "VRML (*.wrl)") {
        const auto base_path = path.substr(0, path.find_last_of("."));
        model.ExportVRML(base_path + ".images.wrl", base_path + ".points3D.wrl",
                         1, Eigen::Vector3d(1, 0, 0));
      }
    } catch (std::domain_error& error) {
      std::cerr << "ERROR: " << error.what() << std::endl;
    }
  }));
}

void MainWindow::ExportFinished() { progress_bar_->hide(); }

void MainWindow::FeatureExtraction() {
  if (options_.Check()) {
    feature_extraction_widget_->show();
    feature_extraction_widget_->raise();
  } else {
    ShowInvalidProjectError();
  }
}

void MainWindow::FeatureMatching() {
  if (options_.Check()) {
    feature_matching_widget_->show();
    feature_matching_widget_->raise();
  } else {
    ShowInvalidProjectError();
  }
}

void MainWindow::DatabaseManagement() {
  if (options_.Check()) {
    database_management_widget_->show();
    database_management_widget_->raise();
  } else {
    ShowInvalidProjectError();
  }
}

void MainWindow::ReconstructionStart() {
  if (!mapper_controller->IsStarted() && !options_.Check()) {
    ShowInvalidProjectError();
    return;
  }

  if (mapper_controller->IsFinished() && HasSelectedModel()) {
    QMessageBox::critical(this, "",
                          tr("Reset reconstruction before starting."));
    return;
  }

  if (mapper_controller->IsStarted()) {
    // Resume existing reconstruction.
    timer_.Resume();
    mapper_controller->Resume();
  } else {
    // Start new reconstruction.
    timer_.Restart();
    mapper_controller->Start();
  }

  DisableBlockingActions();
  action_reconstruction_pause_->setEnabled(true);
}

void MainWindow::ReconstructionStep() {
  if (mapper_controller->IsFinished() && HasSelectedModel()) {
    QMessageBox::critical(this, "",
                          tr("Reset reconstruction before starting."));
    return;
  }

  action_reconstruction_step_->setEnabled(false);
  ReconstructionStart();
  ReconstructionPause();
  action_reconstruction_step_->setEnabled(true);
}

void MainWindow::ReconstructionPause() {
  timer_.Pause();
  mapper_controller->Pause();
  EnableBlockingActions();
  action_reconstruction_pause_->setEnabled(false);
}

void MainWindow::ReconstructionOptions() {
  reconstruction_options_widget_->show();
  reconstruction_options_widget_->raise();
}

void MainWindow::ReconstructionFinish() {
  timer_.Pause();
  mapper_controller->Stop();
  EnableBlockingActions();
  action_reconstruction_step_->setEnabled(false);
  action_reconstruction_pause_->setEnabled(false);
}

void MainWindow::ReconstructionReset() {
  timer_.Reset();
  UpdateTimer();
  CreateControllers();
  EnableBlockingActions();
  RenderClear();
  log_widget_->Clear();
}

void MainWindow::ReconstructionNormalize() {
  if (!IsSelectedModelValid()) {
    return;
  }
  action_reconstruction_step_->setEnabled(false);
  mapper_controller->Model(SelectedModelIdx()).Normalize();
  action_reconstruction_step_->setEnabled(true);
}

void MainWindow::BundleAdjustment() {
  if (!IsSelectedModelValid()) {
    return;
  }

  DisableBlockingActions();
  action_reconstruction_pause_->setDisabled(true);

  ba_controller->reconstruction = &mapper_controller->Model(SelectedModelIdx());
  ba_controller->Start();
}

void MainWindow::BundleAdjustmentFinish() {
  EnableBlockingActions();
  RenderNow();
}

void MainWindow::BundleAdjustmentOptions() {
  bundle_adjustment_options_widget_->show();
  bundle_adjustment_options_widget_->raise();
}

void MainWindow::Render() {
  if (mapper_controller->NumModels() == 0) {
    return;
  }

  const Reconstruction& model = mapper_controller->Model(SelectedModelIdx());

  int refresh_rate;
  if (options_.render_options->adapt_refresh_rate) {
    refresh_rate = static_cast<int>(model.NumRegImages() / 50 + 1);
  } else {
    refresh_rate = options_.render_options->refresh_rate;
  }

  if (!render_options_widget_->automatic_update ||
      render_options_widget_->counter % refresh_rate != 0) {
    render_options_widget_->counter += 1;
    return;
  }

  render_options_widget_->counter += 1;

  RenderNow();
}

void MainWindow::RenderNow() {
  model_manager_widget_->UpdateModels(mapper_controller->Models());
  RenderSelectedModel();
}

void MainWindow::RenderSelectedModel() {
  if (mapper_controller->NumModels() == 0) {
    RenderClear();
    return;
  }

  const size_t model_idx = SelectedModelIdx();
  opengl_window_->reconstruction = &mapper_controller->Model(model_idx);
  opengl_window_->Update();
}

void MainWindow::RenderClear() {
  model_manager_widget_->SetModelIdx(ModelManagerWidget::kNewestModelIdx);
  opengl_window_->Clear();
}

void MainWindow::RenderOptions() {
  render_options_widget_->show();
  render_options_widget_->raise();
}

void MainWindow::SelectModelIdx(const size_t) { RenderSelectedModel(); }

size_t MainWindow::SelectedModelIdx() {
  size_t model_idx = model_manager_widget_->ModelIdx();
  if (model_idx == ModelManagerWidget::kNewestModelIdx) {
    if (mapper_controller->NumModels() > 0) {
      model_idx = mapper_controller->NumModels() - 1;
    }
  }
  return model_idx;
}

bool MainWindow::HasSelectedModel() {
  const size_t model_idx = model_manager_widget_->ModelIdx();
  if (model_idx == ModelManagerWidget::kNewestModelIdx) {
    if (mapper_controller->NumModels() == 0) {
      return false;
    }
  }
  return true;
}

bool MainWindow::IsSelectedModelValid() {
  if (!HasSelectedModel()) {
    QMessageBox::critical(this, "", tr("No model selected."));
    return false;
  }
  return true;
}

void MainWindow::GrabImage() {
  QString file_name = QFileDialog::getSaveFileName(this, tr("Save image"), "",
                                                   tr("Images (*.png *.jpg)"));
  if (file_name != "") {
    if (!HasFileExtension(file_name.toUtf8().constData(), ".png") &&
        !HasFileExtension(file_name.toUtf8().constData(), ".jpg")) {
      file_name += ".png";
    }
    QImage image = opengl_window_->GrabImage();
    image.save(file_name);
  }
}

void MainWindow::UndistortImages() {
  if (!IsSelectedModelValid()) {
    return;
  }
  undistort_widget_->reconstruction =
      mapper_controller->Model(SelectedModelIdx());
  undistort_widget_->show();
  undistort_widget_->raise();
}

void MainWindow::ShowModelStats() {
  if (!IsSelectedModelValid()) {
    return;
  }
  model_stats_widget_->show();
  model_stats_widget_->raise();
  model_stats_widget_->Update(mapper_controller->Model(SelectedModelIdx()));
}

void MainWindow::ShowMatchMatrix() {
  match_matrix_widget_->show();
  match_matrix_widget_->raise();
  match_matrix_widget_->Update();
}

void MainWindow::ShowLog() {
  log_widget_->show();
  log_widget_->raise();
  dock_log_widget_->show();
  dock_log_widget_->raise();
}

void MainWindow::ExtractColors() {
  if (!IsSelectedModelValid()) {
    return;
  }

  progress_bar_->setLabelText(tr("Extracting colors"));
  progress_bar_->raise();
  progress_bar_->show();

  extract_colors_watcher_->setFuture(QtConcurrent::run([this]() {
    auto& model = mapper_controller->Model(SelectedModelIdx());
    model.ExtractColorsForAllImages(*this->options_.image_path);
  }));
}

void MainWindow::ExtractColorsFinished() {
  RenderNow();
  progress_bar_->hide();
}

void MainWindow::ResetOptions() {
  const std::string project_path = *options_.project_path;
  const std::string log_path = *options_.log_path;
  const std::string image_path = *options_.image_path;
  const std::string database_path = *options_.database_path;

  options_.Reset();
  options_.AddAllOptions();

  *options_.project_path = project_path;
  *options_.log_path = log_path;
  *options_.image_path = image_path;
  *options_.database_path = database_path;
}

void MainWindow::About() {
  QMessageBox::about(
      this, tr("About"),
      QString().sprintf("COLMAP %s<br /><br />"
                        "Author:<br />Johannes L. Schönberger<br /><br />"
                        "Email:<br />jsch@cs.unc.edu",
                        COLMAP_VERSION));
}

void MainWindow::License() {
  QTextEdit* text_viewer = new QTextEdit(this);
  text_viewer->setReadOnly(true);
  text_viewer->setWindowFlags(Qt::Dialog);
  text_viewer->resize(size().width() - 20, size().height() - 20);
  text_viewer->setWindowTitle("License");

  QString license_content = "";

  license_content += "<h2>COLMAP</h2>";
  license_content +=
      "Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu><br>"
      "This program is free software: you can redistribute it and/or modify<br>"
      "it under the terms of the GNU General Public License as published by<br>"
      "the Free Software Foundation, either version 3 of the License, or<br>"
      "(at your option) any later version.<br><br>"
      "This program is distributed in the hope that it will be useful,<br>"
      "but WITHOUT ANY WARRANTY; without even the implied warranty of<br>"
      "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the<br>"
      "GNU General Public License for more details.<br><br>"
      "You should have received a copy of the GNU General Public License<br>"
      "along with this program.  If not, see <http://www.gnu.org/licenses/>.";

  license_content += "<h2>External</h2>";

  license_content += "<h3>FLANN</h3>";
  license_content +=
      "The BSD License<br>"
      "<br>"
      "Copyright (c) 2008-2011  Marius Muja (mariusm@cs.ubc.ca). "
      "All rights reserved.<br>"
      "Copyright (c) 2008-2011  David G. Lowe (lowe@cs.ubc.ca). "
      "All rights reserved.<br>"
      "<br>"
      "Redistribution and use in source and binary forms, with or without<br>"
      "modification, are permitted provided that the following conditions<br>"
      "are met:<br>"
      "<br>"
      " * Redistributions of source code must retain the above copyright<br>"
      "   notice, this list of conditions and the following disclaimer.<br>"
      " * Redistributions in binary form must reproduce the above copyright<br>"
      "   notice, this list of conditions and the following disclaimer in<br>"
      "   the documentation and/or other materials provided with the<br>"
      "   distribution.<br>"
      " * Neither the name of the \"University of British Columbia\" nor<br>"
      "   the names of its contributors may be used to endorse or promote<br>"
      "   products derived from this software without specific prior<br>"
      "   written permission.<br>"
      "<br>"
      "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS<br>"
      "\"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT<br>"
      "LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS<br>"
      "FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE<br>"
      "COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,<br>"
      "INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES<br>"
      "(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS<br>"
      "OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS<br>"
      "INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,<br>"
      "WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE<br>"
      "OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,<br>"
      "EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.";

  license_content += "<h3>PBA</h3>";
  license_content +=
      "Copyright (c) 2011  Changchang Wu (ccwu@cs.washington.edu)<br>"
      "and the University of Washington at Seattle<br>"
      "<br>"
      "This library is free software; you can redistribute it and/or<br>"
      "modify it under the terms of the GNU General Public<br>"
      "License as published by the Free Software Foundation; either<br>"
      "Version 3 of the License, or (at your option) any later version.<br>"
      "<br>"
      "This library is distributed in the hope that it will be useful,<br>"
      "but WITHOUT ANY WARRANTY; without even the implied warranty of<br>"
      "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU<br>"
      "General Public License for more details.";

  license_content += "<h3>SiftGPU</h3>";
  license_content +=
      "Copyright (c) 2007 University of North Carolina at Chapel Hill<br>"
      "All Rights Reserved<br>"
      "<br>"
      "Permission to use, copy, modify and distribute this software and its<br>"
      "documentation for educational, research and non-profit purposes,<br>"
      "without fee, and without a written agreement is hereby granted,<br>"
      "provided that the above copyright notice and the following paragraph<br>"
      "appear in all copies.<br>"
      "<br>"
      "The University of North Carolina at Chapel Hill make no<br>"
      "representations about the suitability of this software for any<br>"
      "purpose. It is provided 'as is' without express or implied warranty.";

  license_content += "<h3>SQLite</h3>";
  license_content +=
      "The author disclaims copyright to this source code. In place of<br>"
      "a legal notice, here is a blessing:<br>"
      "May you do good and not evil.<br>"
      "May you find forgiveness for yourself and forgive others.<br>"
      "May you share freely, never taking more than you give.";

  license_content += "<h3>VLFeat</h3>";
  license_content +=
      "Copyright (C) 2007-11, Andrea Vedaldi and Brian Fulkerson<br>"
      "Copyright (C) 2012-13, The VLFeat Team<br>"
      "All rights reserved.<br>"
      "<br>"
      "Redistribution and use in source and binary forms, with or without<br>"
      "modification, are permitted provided that the following conditions <br>"
      "are met:<br>"
      "1. Redistributions of source code must retain the above copyright<br>"
      "   notice, this list of conditions and the following disclaimer.<br>"
      "2. Redistributions in binary form must reproduce the above copyright<br>"
      "   notice, this list of conditions and the following disclaimer in<br>"
      "   the documentation and/or other materials provided with the<br>"
      "   distribution.<br>"
      "<br>"
      "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS<br>"
      "AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT<br>"
      "LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS<br>"
      "FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE<br>"
      "COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,<br>"
      "INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,<br>"
      "BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;<br>"
      "LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER<br>"
      "CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT<br>"
      "LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN<br>"
      "ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE<br>"
      "POSSIBILITY OF SUCH DAMAGE.";

  text_viewer->setHtml(license_content);
  text_viewer->show();
}

void MainWindow::RenderToggle() {
  if (render_options_widget_->automatic_update) {
    render_options_widget_->automatic_update = false;
    render_options_widget_->counter = 0;
    action_render_toggle_->setIcon(QIcon(":/media/render-disabled.png"));
    action_render_toggle_->setText(tr("Enable continuous rendering"));
  } else {
    render_options_widget_->automatic_update = true;
    render_options_widget_->counter = 0;
    Render();
    action_render_toggle_->setIcon(QIcon(":/media/render-enabled.png"));
    action_render_toggle_->setText(tr("Disable continuous rendering"));
  }
}

void MainWindow::UpdateTimer() {
  const int elapsed_time = static_cast<int>(timer_.ElapsedSeconds());
  const int seconds = elapsed_time % 60;
  const int minutes = (elapsed_time / 60) % 60;
  const int hours = (elapsed_time / 3600) % 24;
  const int days = elapsed_time / 86400;
  statusbar_timer_label_->setText(QString().sprintf(
      "Time %02d:%02d:%02d:%02d", days, hours, minutes, seconds));
}

void MainWindow::ShowInvalidProjectError() {
  QMessageBox::critical(this, "",
                        tr("You must create or open a valid project."));
}

void MainWindow::EnableBlockingActions() {
  for (auto& action : blocking_actions_) {
    action->setEnabled(true);
  }
}

void MainWindow::DisableBlockingActions() {
  for (auto& action : blocking_actions_) {
    action->setDisabled(true);
  }
}

void MainWindow::UpdateWindowTitle() {
  if (*options_.project_path == "") {
    setWindowTitle(QString::fromStdString("COLMAP"));
  } else {
    std::string project_title = *options_.project_path;
    if (project_title.size() > 80) {
      project_title =
          "..." + project_title.substr(project_title.size() - 77, 77);
    }
    setWindowTitle(QString::fromStdString("COLMAP - " + project_title));
  }
}

}  // namespace colmap
