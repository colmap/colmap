// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "util/version.h"

namespace colmap {

MainWindow::MainWindow(const OptionManager& options)
    : options_(options),
      thread_control_widget_(new ThreadControlWidget(this)),
      window_closed_(false) {
  resize(1024, 600);
  UpdateWindowTitle();

  CreateWidgets();
  CreateActions();
  CreateMenus();
  CreateToolbar();
  CreateStatusbar();
  CreateControllers();

  ShowLog();

  options_.AddAllOptions();
}

const ReconstructionManager& MainWindow::GetReconstructionManager() const {
  return reconstruction_manager_;
}

void MainWindow::showEvent(QShowEvent* event) {
  after_show_event_->trigger();
  event->accept();
}

void MainWindow::afterShowEvent() { opengl_window_->PaintGL(); }

void MainWindow::closeEvent(QCloseEvent* event) {
  if (window_closed_) {
    event->accept();
    return;
  }

  if (project_widget_->IsValid() && *options_.project_path == "") {
    // Project was created, but not yet saved
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(
        this, "",
        tr("You have not saved your project. Do you want to save it?"),
        QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::Yes) {
      ProjectSave();
    }
  }

  QMessageBox::StandardButton reply;
  reply = QMessageBox::question(this, "", tr("Do you really want to quit?"),
                                QMessageBox::Yes | QMessageBox::No);
  if (reply == QMessageBox::No) {
    event->ignore();
  } else {
    if (mapper_controller_) {
      mapper_controller_->Stop();
      mapper_controller_->Wait();
    }

    log_widget_->close();
    event->accept();

    window_closed_ = true;
  }
}

void MainWindow::CreateWidgets() {
  opengl_window_ = new OpenGLWindow(this, &options_);

#ifdef _MSC_VER
  setCentralWidget(QWidget::createWindowContainer(opengl_window_, this,
                                                  Qt::MSWindowsOwnDC));
#else
  setCentralWidget(QWidget::createWindowContainer(opengl_window_, this));
#endif

  project_widget_ = new ProjectWidget(this, &options_);
  project_widget_->SetDatabasePath(*options_.database_path);
  project_widget_->SetImagePath(*options_.image_path);

  feature_extraction_widget_ = new FeatureExtractionWidget(this, &options_);
  feature_matching_widget_ = new FeatureMatchingWidget(this, &options_);
  database_management_widget_ = new DatabaseManagementWidget(this, &options_);
  automatic_reconstruction_widget_ = new AutomaticReconstructionWidget(this);
  reconstruction_options_widget_ =
      new ReconstructionOptionsWidget(this, &options_);
  bundle_adjustment_widget_ = new BundleAdjustmentWidget(this, &options_);
  dense_reconstruction_widget_ = new DenseReconstructionWidget(this, &options_);
  render_options_widget_ =
      new RenderOptionsWidget(this, &options_, opengl_window_);
  log_widget_ = new LogWidget(this);
  undistortion_widget_ = new UndistortionWidget(this, &options_);
  reconstruction_manager_widget_ =
      new ReconstructionManagerWidget(this, &reconstruction_manager_);
  reconstruction_stats_widget_ = new ReconstructionStatsWidget(this);
  match_matrix_widget_ = new MatchMatrixWidget(this, &options_);
  license_widget_ = new LicenseWidget(this);

  dock_log_widget_ = new QDockWidget("Log", this);
  dock_log_widget_->setWidget(log_widget_);
  addDockWidget(Qt::RightDockWidgetArea, dock_log_widget_);
}

void MainWindow::CreateActions() {
  after_show_event_ = new QAction(tr("After show event"), this);
  connect(after_show_event_, &QAction::triggered, this,
          &MainWindow::afterShowEvent, Qt::QueuedConnection);

  //////////////////////////////////////////////////////////////////////////////
  // File actions
  //////////////////////////////////////////////////////////////////////////////

  action_project_new_ =
      new QAction(QIcon(":/media/project-new.png"), tr("New project"), this);
  action_project_new_->setShortcuts(QKeySequence::New);
  connect(action_project_new_, &QAction::triggered, this,
          &MainWindow::ProjectNew);

  action_project_open_ =
      new QAction(QIcon(":/media/project-open.png"), tr("Open project"), this);
  action_project_open_->setShortcuts(QKeySequence::Open);
  connect(action_project_open_, &QAction::triggered, this,
          &MainWindow::ProjectOpen);

  action_project_edit_ =
      new QAction(QIcon(":/media/project-edit.png"), tr("Edit project"), this);
  connect(action_project_edit_, &QAction::triggered, this,
          &MainWindow::ProjectEdit);

  action_project_save_ =
      new QAction(QIcon(":/media/project-save.png"), tr("Save project"), this);
  action_project_save_->setShortcuts(QKeySequence::Save);
  connect(action_project_save_, &QAction::triggered, this,
          &MainWindow::ProjectSave);

  action_project_save_as_ = new QAction(QIcon(":/media/project-save-as.png"),
                                        tr("Save project as..."), this);
  action_project_save_as_->setShortcuts(QKeySequence::SaveAs);
  connect(action_project_save_as_, &QAction::triggered, this,
          &MainWindow::ProjectSaveAs);

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

  action_export_as_text_ = new QAction(QIcon(":/media/export-as-text.png"),
                                       tr("Export model as text"), this);
  connect(action_export_as_text_, &QAction::triggered, this,
          &MainWindow::ExportAsText);
  blocking_actions_.push_back(action_export_as_text_);

  action_quit_ = new QAction(tr("Quit"), this);
  connect(action_quit_, &QAction::triggered, this, &MainWindow::close);

  //////////////////////////////////////////////////////////////////////////////
  // Processing action
  //////////////////////////////////////////////////////////////////////////////

  action_feature_extraction_ = new QAction(
      QIcon(":/media/feature-extraction.png"), tr("Feature extraction"), this);
  connect(action_feature_extraction_, &QAction::triggered, this,
          &MainWindow::FeatureExtraction);
  blocking_actions_.push_back(action_feature_extraction_);

  action_feature_matching_ = new QAction(QIcon(":/media/feature-matching.png"),
                                         tr("Feature matching"), this);
  connect(action_feature_matching_, &QAction::triggered, this,
          &MainWindow::FeatureMatching);
  blocking_actions_.push_back(action_feature_matching_);

  action_database_management_ =
      new QAction(QIcon(":/media/database-management.png"),
                  tr("Database management"), this);
  connect(action_database_management_, &QAction::triggered, this,
          &MainWindow::DatabaseManagement);
  blocking_actions_.push_back(action_database_management_);

  //////////////////////////////////////////////////////////////////////////////
  // Reconstruction actions
  //////////////////////////////////////////////////////////////////////////////

  action_automatic_reconstruction_ =
      new QAction(QIcon(":/media/automatic-reconstruction.png"),
                  tr("Automatic reconstruction"), this);
  connect(action_automatic_reconstruction_, &QAction::triggered, this,
          &MainWindow::AutomaticReconstruction);

  action_reconstruction_start_ =
      new QAction(QIcon(":/media/reconstruction-start.png"),
                  tr("Start reconstruction"), this);
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
          &MainWindow::ReconstructionOverwrite);

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

  action_dense_reconstruction_ =
      new QAction(QIcon(":/media/dense-reconstruction.png"),
                  tr("Dense reconstruction"), this);
  connect(action_dense_reconstruction_, &QAction::triggered, this,
          &MainWindow::DenseReconstruction);

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

  connect(
      reconstruction_manager_widget_,
      static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
      this, &MainWindow::SelectReconstructionIdx);

  //////////////////////////////////////////////////////////////////////////////
  // Extras actions
  //////////////////////////////////////////////////////////////////////////////

  action_reconstruction_stats_ =
      new QAction(QIcon(":/media/reconstruction-stats.png"),
                  tr("Show model statistics"), this);
  connect(action_reconstruction_stats_, &QAction::triggered, this,
          &MainWindow::ReconstructionStats);

  action_match_matrix_ = new QAction(QIcon(":/media/match-matrix.png"),
                                     tr("Show match matrix"), this);
  connect(action_match_matrix_, &QAction::triggered, this,
          &MainWindow::MatchMatrix);

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
      new QAction(QIcon(":/media/undistort.png"), tr("Undistortion"), this);
  connect(action_undistort_, &QAction::triggered, this,
          &MainWindow::UndistortImages);
  blocking_actions_.push_back(action_undistort_);

  action_extract_colors_ = new QAction(tr("Extract colors"), this);
  connect(action_extract_colors_, &QAction::triggered, this,
          &MainWindow::ExtractColors);

  action_reset_options_ = new QAction(tr("Set default options"), this);
  connect(action_reset_options_, &QAction::triggered, this,
          &MainWindow::ResetOptions);

  action_set_options_for_individual_ =
      new QAction(tr("Set options for individual images"), this);
  connect(action_set_options_for_individual_, &QAction::triggered, this,
          &MainWindow::SetOptionsForIndividual);

  action_set_options_for_video_ =
      new QAction(tr("Set options for video frames"), this);
  connect(action_set_options_for_video_, &QAction::triggered, this,
          &MainWindow::SetOptionsForVideo);

  action_set_options_for_internet_ =
      new QAction(tr("Set options for Internet images"), this);
  connect(action_set_options_for_internet_, &QAction::triggered, this,
          &MainWindow::SetOptionsForInternet);

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

  action_about_ = new QAction(tr("About"), this);
  connect(action_about_, &QAction::triggered, this, &MainWindow::About);
  action_documentation_ = new QAction(tr("Documentation"), this);
  connect(action_documentation_, &QAction::triggered, this,
          &MainWindow::Documentation);
  action_support_ = new QAction(tr("Support"), this);
  connect(action_support_, &QAction::triggered, this, &MainWindow::Support);
  action_license_ = new QAction(tr("License"), this);
  connect(action_license_, &QAction::triggered, license_widget_,
          &QTextEdit::show);
}

void MainWindow::CreateMenus() {
  QMenu* file_menu = new QMenu(tr("File"), this);
  file_menu->addAction(action_project_new_);
  file_menu->addAction(action_project_open_);
  file_menu->addAction(action_project_edit_);
  file_menu->addAction(action_project_save_);
  file_menu->addAction(action_project_save_as_);
  file_menu->addSeparator();
  file_menu->addAction(action_import_);
  file_menu->addAction(action_import_from_);
  file_menu->addSeparator();
  file_menu->addAction(action_export_);
  file_menu->addAction(action_export_all_);
  file_menu->addAction(action_export_as_);
  file_menu->addAction(action_export_as_text_);
  file_menu->addSeparator();
  file_menu->addAction(action_quit_);
  menuBar()->addAction(file_menu->menuAction());

  QMenu* preprocessing_menu = new QMenu(tr("Processing"), this);
  preprocessing_menu->addAction(action_feature_extraction_);
  preprocessing_menu->addAction(action_feature_matching_);
  preprocessing_menu->addAction(action_database_management_);
  menuBar()->addAction(preprocessing_menu->menuAction());

  QMenu* reconstruction_menu = new QMenu(tr("Reconstruction"), this);
  reconstruction_menu->addAction(action_automatic_reconstruction_);
  reconstruction_menu->addSeparator();
  reconstruction_menu->addAction(action_reconstruction_start_);
  reconstruction_menu->addAction(action_reconstruction_pause_);
  reconstruction_menu->addAction(action_reconstruction_step_);
  reconstruction_menu->addSeparator();
  reconstruction_menu->addAction(action_reconstruction_reset_);
  reconstruction_menu->addAction(action_reconstruction_normalize_);
  reconstruction_menu->addAction(action_reconstruction_options_);
  reconstruction_menu->addSeparator();
  reconstruction_menu->addAction(action_bundle_adjustment_);
  reconstruction_menu->addAction(action_dense_reconstruction_);
  menuBar()->addAction(reconstruction_menu->menuAction());

  QMenu* render_menu = new QMenu(tr("Render"), this);
  render_menu->addAction(action_render_toggle_);
  render_menu->addAction(action_render_reset_view_);
  render_menu->addAction(action_render_options_);
  menuBar()->addAction(render_menu->menuAction());

  QMenu* extras_menu = new QMenu(tr("Extras"), this);
  extras_menu->addAction(action_log_show_);
  extras_menu->addAction(action_match_matrix_);
  extras_menu->addAction(action_reconstruction_stats_);
  extras_menu->addSeparator();
  extras_menu->addAction(action_grab_image_);
  extras_menu->addAction(action_grab_movie_);
  extras_menu->addSeparator();
  extras_menu->addAction(action_undistort_);
  extras_menu->addAction(action_extract_colors_);
  extras_menu->addSeparator();
  extras_menu->addAction(action_reset_options_);
  extras_menu->addAction(action_set_options_for_individual_);
  extras_menu->addAction(action_set_options_for_video_);
  extras_menu->addAction(action_set_options_for_internet_);
  menuBar()->addAction(extras_menu->menuAction());

  QMenu* help_menu = new QMenu(tr("Help"), this);
  help_menu->addAction(action_about_);
  help_menu->addAction(action_documentation_);
  help_menu->addAction(action_support_);
  help_menu->addAction(action_license_);
  menuBar()->addAction(help_menu->menuAction());

  // TODO: Make the native menu bar work on OSX. Simply setting this to true
  // will result in a menubar which is not clickable until the main window is
  // defocused and refocused.
  menuBar()->setNativeMenuBar(false);
}

void MainWindow::CreateToolbar() {
  file_toolbar_ = addToolBar(tr("File"));
  file_toolbar_->addAction(action_project_new_);
  file_toolbar_->addAction(action_project_open_);
  file_toolbar_->addAction(action_project_edit_);
  file_toolbar_->addAction(action_project_save_);
  file_toolbar_->addAction(action_import_);
  file_toolbar_->addAction(action_export_);
  file_toolbar_->setIconSize(QSize(16, 16));

  preprocessing_toolbar_ = addToolBar(tr("Processing"));
  preprocessing_toolbar_->addAction(action_feature_extraction_);
  preprocessing_toolbar_->addAction(action_feature_matching_);
  preprocessing_toolbar_->addAction(action_database_management_);
  preprocessing_toolbar_->setIconSize(QSize(16, 16));

  reconstruction_toolbar_ = addToolBar(tr("Reconstruction"));
  reconstruction_toolbar_->addAction(action_automatic_reconstruction_);
  reconstruction_toolbar_->addAction(action_reconstruction_start_);
  reconstruction_toolbar_->addAction(action_reconstruction_step_);
  reconstruction_toolbar_->addAction(action_reconstruction_pause_);
  reconstruction_toolbar_->addAction(action_reconstruction_options_);
  reconstruction_toolbar_->addAction(action_bundle_adjustment_);
  reconstruction_toolbar_->addAction(action_dense_reconstruction_);
  reconstruction_toolbar_->setIconSize(QSize(16, 16));

  render_toolbar_ = addToolBar(tr("Render"));
  render_toolbar_->addAction(action_render_toggle_);
  render_toolbar_->addAction(action_render_reset_view_);
  render_toolbar_->addAction(action_render_options_);
  render_toolbar_->addWidget(reconstruction_manager_widget_);
  render_toolbar_->setIconSize(QSize(16, 16));

  extras_toolbar_ = addToolBar(tr("Extras"));
  extras_toolbar_->addAction(action_log_show_);
  extras_toolbar_->addAction(action_match_matrix_);
  extras_toolbar_->addAction(action_reconstruction_stats_);
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
  if (mapper_controller_) {
    mapper_controller_->Stop();
    mapper_controller_->Wait();
  }

  mapper_controller_.reset(new IncrementalMapperController(
      options_.mapper.get(), *options_.image_path, *options_.database_path,
      &reconstruction_manager_));
  mapper_controller_->AddCallback(
      IncrementalMapperController::INITIAL_IMAGE_PAIR_REG_CALLBACK, [this]() {
        if (!mapper_controller_->IsStopped()) {
          action_render_now_->trigger();
        }
      });
  mapper_controller_->AddCallback(
      IncrementalMapperController::NEXT_IMAGE_REG_CALLBACK, [this]() {
        if (!mapper_controller_->IsStopped()) {
          action_render_->trigger();
        }
      });
  mapper_controller_->AddCallback(
      IncrementalMapperController::LAST_IMAGE_REG_CALLBACK, [this]() {
        if (!mapper_controller_->IsStopped()) {
          action_render_now_->trigger();
        }
      });
  mapper_controller_->AddCallback(
      IncrementalMapperController::FINISHED_CALLBACK, [this]() {
        if (!mapper_controller_->IsStopped()) {
          action_render_now_->trigger();
          action_reconstruction_finish_->trigger();
        }
        if (reconstruction_manager_.Size() == 0) {
          action_reconstruction_reset_->trigger();
        }
      });
}

void MainWindow::ProjectNew() {
  if (ReconstructionOverwrite()) {
    project_widget_->Reset();
    project_widget_->show();
    project_widget_->raise();
  }
}

bool MainWindow::ProjectOpen() {
  if (!ReconstructionOverwrite()) {
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
      project_widget_->SetDatabasePath(*options_.database_path);
      project_widget_->SetImagePath(*options_.image_path);
      UpdateWindowTitle();
      return true;
    } else {
      ShowInvalidProjectError();
    }
  }

  return false;
}

void MainWindow::ProjectEdit() {
  project_widget_->show();
  project_widget_->raise();
}

void MainWindow::ProjectSave() {
  if (!ExistsFile(*options_.project_path)) {
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

void MainWindow::ProjectSaveAs() {
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
  const std::string import_path =
      QFileDialog::getExistingDirectory(this, tr("Select source..."), "",
                                        QFileDialog::ShowDirsOnly)
          .toUtf8()
          .constData();

  // Selection canceled?
  if (import_path == "") {
    return;
  }

  const std::string project_path = JoinPaths(import_path, "project.ini");
  const std::string cameras_bin_path = JoinPaths(import_path, "cameras.bin");
  const std::string images_bin_path = JoinPaths(import_path, "images.bin");
  const std::string points3D_bin_path = JoinPaths(import_path, "points3D.bin");
  const std::string cameras_txt_path = JoinPaths(import_path, "cameras.txt");
  const std::string images_txt_path = JoinPaths(import_path, "images.txt");
  const std::string points3D_txt_path = JoinPaths(import_path, "points3D.txt");

  if ((!ExistsFile(cameras_bin_path) || !ExistsFile(images_bin_path) ||
       !ExistsFile(points3D_bin_path)) &&
      (!ExistsFile(cameras_txt_path) || !ExistsFile(images_txt_path) ||
       !ExistsFile(points3D_txt_path))) {
    QMessageBox::critical(this, "",
                          tr("cameras, images, and points3D files do not exist "
                             "in chosen directory."));
    return;
  }

  if (!ReconstructionOverwrite()) {
    return;
  }

  bool edit_project = false;
  if (ExistsFile(project_path)) {
    options_.ReRead(project_path);
  } else {
    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "",
        tr("Directory does not contain a <i>project.ini</i>. To "
           "resume the reconstruction, you need to specify a valid "
           "database and image path. Do you want to select the paths "
           "now (or press <i>No</i> to only visualize the reconstruction)?"),
        QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::Yes) {
      edit_project = true;
    }
  }

  thread_control_widget_->StartFunction(
      "Importing...", [this, import_path, edit_project]() {
        const size_t idx = reconstruction_manager_.Read(import_path);
        reconstruction_manager_widget_->Update();
        reconstruction_manager_widget_->SelectReconstruction(idx);
        action_bundle_adjustment_->setEnabled(true);
        action_render_now_->trigger();
        if (edit_project) {
          action_project_edit_->trigger();
        }
      });
}

void MainWindow::ImportFrom() {
  const std::string import_path =
      QFileDialog::getOpenFileName(this, tr("Select source..."), "")
          .toUtf8()
          .constData();

  // Selection canceled?
  if (import_path == "") {
    return;
  }

  if (!ExistsFile(import_path)) {
    QMessageBox::critical(this, "", tr("Invalid file"));
    return;
  }

  if (!HasFileExtension(import_path, ".ply")) {
    QMessageBox::critical(this, "",
                          tr("Invalid file format (supported formats: PLY)"));
    return;
  }

  thread_control_widget_->StartFunction("Importing...", [this, import_path]() {
    const size_t reconstruction_idx = reconstruction_manager_.Add();
    reconstruction_manager_.Get(reconstruction_idx).ImportPLY(import_path);
    options_.render->min_track_len = 0;
    reconstruction_manager_widget_->Update();
    reconstruction_manager_widget_->SelectReconstruction(reconstruction_idx);
    action_render_now_->trigger();
  });
}

void MainWindow::Export() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }

  const std::string export_path =
      QFileDialog::getExistingDirectory(this, tr("Select destination..."), "",
                                        QFileDialog::ShowDirsOnly)
          .toUtf8()
          .constData();

  // Selection canceled?
  if (export_path == "") {
    return;
  }

  const std::string cameras_name = "cameras.bin";
  const std::string images_name = "images.bin";
  const std::string points3D_name = "points3D.bin";

  const std::string project_path = JoinPaths(export_path, "project.ini");
  const std::string cameras_path = JoinPaths(export_path, cameras_name);
  const std::string images_path = JoinPaths(export_path, images_name);
  const std::string points3D_path = JoinPaths(export_path, points3D_name);

  if (ExistsFile(cameras_path) || ExistsFile(images_path) ||
      ExistsFile(points3D_path)) {
    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "",
        StringPrintf(
            "The files <i>%s</i>, <i>%s</i>, or <i>%s</i> already "
            "exist in the selected destination. Do you want to overwrite them?",
            cameras_name.c_str(), images_name.c_str(), points3D_name.c_str())
            .c_str(),
        QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::No) {
      return;
    }
  }

  thread_control_widget_->StartFunction(
      "Exporting...", [this, export_path, project_path]() {
        const auto& reconstruction =
            reconstruction_manager_.Get(SelectedReconstructionIdx());
        reconstruction.WriteBinary(export_path);
        options_.Write(project_path);
      });
}

void MainWindow::ExportAll() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }

  const std::string export_path =
      QFileDialog::getExistingDirectory(this, tr("Select destination..."), "",
                                        QFileDialog::ShowDirsOnly)
          .toUtf8()
          .constData();

  // Selection canceled?
  if (export_path == "") {
    return;
  }

  thread_control_widget_->StartFunction("Exporting...", [this, export_path]() {
    reconstruction_manager_.Write(export_path, &options_);
  });
}

void MainWindow::ExportAs() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }

  QString filter("NVM (*.nvm)");
  const std::string export_path =
      QFileDialog::getSaveFileName(
          this, tr("Select destination..."), "",
          "NVM (*.nvm);;Bundler (*.out);;PLY (*.ply);;VRML (*.wrl)", &filter)
          .toUtf8()
          .constData();

  // Selection canceled?
  if (export_path == "") {
    return;
  }

  thread_control_widget_->StartFunction(
      "Exporting...", [this, export_path, filter]() {
        const Reconstruction& reconstruction =
            reconstruction_manager_.Get(SelectedReconstructionIdx());
        if (filter == "NVM (*.nvm)") {
          reconstruction.ExportNVM(export_path);
        } else if (filter == "Bundler (*.out)") {
          reconstruction.ExportBundler(export_path, export_path + ".list.txt");
        } else if (filter == "PLY (*.ply)") {
          reconstruction.ExportPLY(export_path);
        } else if (filter == "VRML (*.wrl)") {
          const auto base_path =
              export_path.substr(0, export_path.find_last_of("."));
          reconstruction.ExportVRML(base_path + ".images.wrl",
                                    base_path + ".points3D.wrl", 1,
                                    Eigen::Vector3d(1, 0, 0));
        }
      });
}

void MainWindow::ExportAsText() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }

  const std::string export_path =
      QFileDialog::getExistingDirectory(this, tr("Select destination..."), "",
                                        QFileDialog::ShowDirsOnly)
          .toUtf8()
          .constData();

  // Selection canceled?
  if (export_path == "") {
    return;
  }

  const std::string cameras_name = "cameras.txt";
  const std::string images_name = "images.txt";
  const std::string points3D_name = "points3D.txt";

  const std::string project_path = JoinPaths(export_path, "project.ini");
  const std::string cameras_path = JoinPaths(export_path, cameras_name);
  const std::string images_path = JoinPaths(export_path, images_name);
  const std::string points3D_path = JoinPaths(export_path, points3D_name);

  if (ExistsFile(cameras_path) || ExistsFile(images_path) ||
      ExistsFile(points3D_path)) {
    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "",
        StringPrintf(
            "The files <i>%s</i>, <i>%s</i>, or <i>%s</i> already "
            "exist in the selected destination. Do you want to overwrite them?",
            cameras_name.c_str(), images_name.c_str(), points3D_name.c_str())
            .c_str(),
        QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::No) {
      return;
    }
  }

  thread_control_widget_->StartFunction(
      "Exporting...", [this, export_path, project_path]() {
        const auto& reconstruction =
            reconstruction_manager_.Get(SelectedReconstructionIdx());
        reconstruction.WriteText(export_path);
        options_.Write(project_path);
      });
}

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

void MainWindow::AutomaticReconstruction() {
  automatic_reconstruction_widget_->show();
  automatic_reconstruction_widget_->raise();
}

void MainWindow::ReconstructionStart() {
  if (!mapper_controller_->IsStarted() && !options_.Check()) {
    ShowInvalidProjectError();
    return;
  }

  if (mapper_controller_->IsFinished() && HasSelectedReconstruction()) {
    QMessageBox::critical(this, "",
                          tr("Reset reconstruction before starting."));
    return;
  }

  if (mapper_controller_->IsStarted()) {
    // Resume existing reconstruction.
    timer_.Resume();
    mapper_controller_->Resume();
  } else {
    // Start new reconstruction.
    CreateControllers();
    timer_.Restart();
    mapper_controller_->Start();
    action_reconstruction_start_->setText(tr("Resume reconstruction"));
  }

  DisableBlockingActions();
  action_reconstruction_pause_->setEnabled(true);
}

void MainWindow::ReconstructionStep() {
  if (mapper_controller_->IsFinished() && HasSelectedReconstruction()) {
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
  mapper_controller_->Pause();
  EnableBlockingActions();
  action_reconstruction_pause_->setEnabled(false);
}

void MainWindow::ReconstructionOptions() {
  reconstruction_options_widget_->show();
  reconstruction_options_widget_->raise();
}

void MainWindow::ReconstructionFinish() {
  timer_.Pause();
  mapper_controller_->Stop();
  EnableBlockingActions();
  action_reconstruction_start_->setEnabled(false);
  action_reconstruction_step_->setEnabled(false);
  action_reconstruction_pause_->setEnabled(false);
}

void MainWindow::ReconstructionReset() {
  CreateControllers();

  reconstruction_manager_.Clear();
  reconstruction_manager_widget_->Update();

  timer_.Reset();
  UpdateTimer();

  EnableBlockingActions();
  action_reconstruction_start_->setText(tr("Start reconstruction"));
  action_reconstruction_pause_->setEnabled(false);

  RenderClear();
}

void MainWindow::ReconstructionNormalize() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }
  action_reconstruction_step_->setEnabled(false);
  reconstruction_manager_.Get(SelectedReconstructionIdx()).Normalize();
  action_reconstruction_step_->setEnabled(true);
}

bool MainWindow::ReconstructionOverwrite() {
  if (reconstruction_manager_.Size() == 0) {
    ReconstructionReset();
    return true;
  }

  QMessageBox::StandardButton reply = QMessageBox::question(
      this, "",
      tr("Do you really want to overwrite the existing reconstruction?"),
      QMessageBox::Yes | QMessageBox::No);
  if (reply == QMessageBox::No) {
    return false;
  } else {
    ReconstructionReset();
    return true;
  }
}

void MainWindow::BundleAdjustment() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }

  bundle_adjustment_widget_->Show(
      &reconstruction_manager_.Get(SelectedReconstructionIdx()));
}

void MainWindow::DenseReconstruction() {
  if (HasSelectedReconstruction()) {
    dense_reconstruction_widget_->Show(
        &reconstruction_manager_.Get(SelectedReconstructionIdx()));
  } else {
    dense_reconstruction_widget_->Show(nullptr);
  }
}

void MainWindow::Render() {
  if (reconstruction_manager_.Size() == 0) {
    return;
  }

  const Reconstruction& reconstruction =
      reconstruction_manager_.Get(SelectedReconstructionIdx());

  int refresh_rate;
  if (options_.render->adapt_refresh_rate) {
    refresh_rate = static_cast<int>(reconstruction.NumRegImages() / 50 + 1);
  } else {
    refresh_rate = options_.render->refresh_rate;
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
  reconstruction_manager_widget_->Update();
  RenderSelectedReconstruction();
}

void MainWindow::RenderSelectedReconstruction() {
  if (reconstruction_manager_.Size() == 0) {
    RenderClear();
    return;
  }

  const size_t reconstruction_idx = SelectedReconstructionIdx();
  opengl_window_->reconstruction =
      &reconstruction_manager_.Get(reconstruction_idx);
  opengl_window_->Update();
}

void MainWindow::RenderClear() {
  reconstruction_manager_widget_->SelectReconstruction(
      ReconstructionManagerWidget::kNewestReconstructionIdx);
  opengl_window_->Clear();
}

void MainWindow::RenderOptions() {
  render_options_widget_->show();
  render_options_widget_->raise();
}

void MainWindow::SelectReconstructionIdx(const size_t) {
  RenderSelectedReconstruction();
}

size_t MainWindow::SelectedReconstructionIdx() {
  size_t reconstruction_idx =
      reconstruction_manager_widget_->SelectedReconstructionIdx();
  if (reconstruction_idx ==
      ReconstructionManagerWidget::kNewestReconstructionIdx) {
    if (reconstruction_manager_.Size() > 0) {
      reconstruction_idx = reconstruction_manager_.Size() - 1;
    }
  }
  return reconstruction_idx;
}

bool MainWindow::HasSelectedReconstruction() {
  const size_t reconstruction_idx =
      reconstruction_manager_widget_->SelectedReconstructionIdx();
  if (reconstruction_idx ==
      ReconstructionManagerWidget::kNewestReconstructionIdx) {
    if (reconstruction_manager_.Size() == 0) {
      return false;
    }
  }
  return true;
}

bool MainWindow::IsSelectedReconstructionValid() {
  if (!HasSelectedReconstruction()) {
    QMessageBox::critical(this, "", tr("No reconstruction selected"));
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
  if (!IsSelectedReconstructionValid()) {
    return;
  }
  undistortion_widget_->Show(
      reconstruction_manager_.Get(SelectedReconstructionIdx()));
}

void MainWindow::ReconstructionStats() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }
  reconstruction_stats_widget_->show();
  reconstruction_stats_widget_->raise();
  reconstruction_stats_widget_->Show(
      reconstruction_manager_.Get(SelectedReconstructionIdx()));
}

void MainWindow::MatchMatrix() { match_matrix_widget_->Show(); }

void MainWindow::ShowLog() {
  log_widget_->show();
  log_widget_->raise();
  dock_log_widget_->show();
  dock_log_widget_->raise();
}

void MainWindow::ExtractColors() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }

  thread_control_widget_->StartFunction("Extracting colors...", [this]() {
    auto& reconstruction =
        reconstruction_manager_.Get(SelectedReconstructionIdx());
    reconstruction.ExtractColorsForAllImages(*options_.image_path);
  });
}

void MainWindow::ResetOptions() {
  const std::string project_path = *options_.project_path;
  const std::string image_path = *options_.image_path;
  const std::string database_path = *options_.database_path;

  options_.Reset();
  options_.AddAllOptions();

  *options_.project_path = project_path;
  *options_.image_path = image_path;
  *options_.database_path = database_path;
}

void MainWindow::SetOptionsForIndividual() {
  const std::string project_path = *options_.project_path;
  const std::string image_path = *options_.image_path;
  const std::string database_path = *options_.database_path;

  options_.Reset();
  options_.AddAllOptions();
  options_.InitForIndividualData();

  *options_.project_path = project_path;
  *options_.image_path = image_path;
  *options_.database_path = database_path;
}

void MainWindow::SetOptionsForVideo() {
  const std::string project_path = *options_.project_path;
  const std::string image_path = *options_.image_path;
  const std::string database_path = *options_.database_path;

  options_.Reset();
  options_.AddAllOptions();
  options_.InitForVideoData();

  *options_.project_path = project_path;
  *options_.image_path = image_path;
  *options_.database_path = database_path;
}

void MainWindow::SetOptionsForInternet() {
  const std::string project_path = *options_.project_path;
  const std::string image_path = *options_.image_path;
  const std::string database_path = *options_.database_path;

  options_.Reset();
  options_.AddAllOptions();
  options_.InitForInternetData();

  *options_.project_path = project_path;
  *options_.image_path = image_path;
  *options_.database_path = database_path;
}

void MainWindow::About() {
  QMessageBox::about(
      this, tr("About"),
      QString().sprintf("<span style='font-weight:normal'><b>%s</b><br />"
                        "<small>(%s)</small><br /><br />"
                        "<b>Author:</b> Johannes L. Schönberger<br /><br />"
                        "<b>Email:</b> jsch at inf.ethz.ch</span>",
                        GetVersionInfo().c_str(), GetBuildInfo().c_str()));
}

void MainWindow::Documentation() {
  QDesktopServices::openUrl(QUrl("https://colmap.github.io/"));
}

void MainWindow::Support() {
  QDesktopServices::openUrl(
      QUrl("https://groups.google.com/forum/#!forum/colmap"));
}

void MainWindow::RenderToggle() {
  if (render_options_widget_->automatic_update) {
    render_options_widget_->automatic_update = false;
    render_options_widget_->counter = 0;
    action_render_toggle_->setIcon(QIcon(":/media/render-disabled.png"));
    action_render_toggle_->setText(tr("Enable rendering"));
  } else {
    render_options_widget_->automatic_update = true;
    render_options_widget_->counter = 0;
    Render();
    action_render_toggle_->setIcon(QIcon(":/media/render-enabled.png"));
    action_render_toggle_->setText(tr("Disable rendering"));
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
                        tr("You must create a valid project using: <i>File > "
                           "New project</i> or <i>File > Edit project</i>"));
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
