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

#ifndef COLMAP_SRC_UI_MAIN_WINDOW_H_
#define COLMAP_SRC_UI_MAIN_WINDOW_H_

#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include <Eigen/Core>

#include <QtCore>
#include <QtGui>
#include <QtWidgets>

#include "base/reconstruction.h"
#include "sfm/controllers.h"
#include "ui/bundle_adjustment_options_widget.h"
#include "ui/database_management_widget.h"
#include "ui/feature_extraction_widget.h"
#include "ui/feature_matching_widget.h"
#include "ui/log_widget.h"
#include "ui/match_matrix_widget.h"
#include "ui/new_project_widget.h"
#include "ui/opengl_window.h"
#include "ui/reconstruction_manager_widget.h"
#include "ui/reconstruction_options_widget.h"
#include "ui/reconstruction_stats_widget.h"
#include "ui/render_options_widget.h"
#include "ui/undistortion_widget.h"
#include "util/bitmap.h"

namespace colmap {

class MainWindow : public QMainWindow {
 public:
  MainWindow(const OptionManager& options);

  const ReconstructionManager& GetReconstructionManager() const;

  bool OverwriteReconstruction();

 protected:
  void showEvent(QShowEvent* event);
  void moveEvent(QMoveEvent* event);
  void closeEvent(QCloseEvent* event);

  // Custom event that is called after `showEvent` by a timer.
  void afterShowEvent();

 private:
  void CreateWidgets();
  void CreateActions();
  void CreateMenus();
  void CreateToolbar();
  void CreateStatusbar();
  void CreateControllers();
  void CreateFutures();
  void CreateProgressBar();

  void CenterProgressBar();

  void NewProject();
  bool OpenProject();
  void SaveProject();
  void SaveProjectAs();
  void Import();
  void ImportFrom();
  void ImportFinished();
  void Export();
  void ExportAll();
  void ExportAs();
  void ExportFinished();

  void FeatureExtraction();
  void FeatureMatching();
  void DatabaseManagement();

  void ReconstructionStart();
  void ReconstructionStep();
  void ReconstructionPause();
  void ReconstructionReset();
  void ReconstructionOptions();
  void ReconstructionFinish();
  void ReconstructionNormalize();

  void BundleAdjustment();
  void BundleAdjustmentOptions();
  void BundleAdjustmentFinish();

  void Render();
  void RenderNow();
  void RenderToggle();
  void RenderOptions();
  void RenderSelectedReconstruction();
  void RenderClear();

  void SelectReconstructionIdx(const size_t);
  size_t SelectedReconstructionIdx();
  bool HasSelectedReconstruction();
  bool IsSelectedReconstructionValid();

  void GrabImage();
  void UndistortImages();

  void ShowReconstructionStats();
  void ShowMatchMatrix();
  void ShowLog();
  void ExtractColors();
  void ExtractColorsFinished();
  void ResetOptions();

  void About();
  void License();

  void ShowInvalidProjectError();
  void UpdateTimer();

  void EnableBlockingActions();
  void DisableBlockingActions();

  void UpdateWindowTitle();

  OptionManager options_;

  ReconstructionManager reconstruction_manager_;
  std::unique_ptr<IncrementalMapperController> mapper_controller_;
  std::unique_ptr<BundleAdjustmentController> ba_controller_;

  OpenGLWindow* opengl_window_;

  Timer timer_;

  QTimer* after_show_event_timer_;

  NewProjectWidget* new_project_widget_;
  FeatureExtractionWidget* feature_extraction_widget_;
  FeatureMatchingWidget* feature_matching_widget_;
  DatabaseManagementWidget* database_management_widget_;
  ReconstructionOptionsWidget* reconstruction_options_widget_;
  BundleAdjustmentOptionsWidget* bundle_adjustment_options_widget_;
  RenderOptionsWidget* render_options_widget_;
  LogWidget* log_widget_;
  UndistortionWidget* undistortion_widget_;
  ReconstructionManagerWidget* reconstruction_manager_widget_;
  ReconstructionStatsWidget* reconstruction_stats_widget_;
  MatchMatrixWidget* match_matrix_widget_;

  QToolBar* file_toolbar_;
  QToolBar* preprocessing_toolbar_;
  QToolBar* reconstruction_toolbar_;
  QToolBar* render_toolbar_;
  QToolBar* extras_toolbar_;

  QDockWidget* dock_log_widget_;

  QTimer* statusbar_timer_;
  QLabel* statusbar_timer_label_;

  QAction* action_new_project_;
  QAction* action_open_project_;
  QAction* action_save_project_;
  QAction* action_save_project_as_;
  QAction* action_import_;
  QAction* action_import_from_;
  QAction* action_export_;
  QAction* action_export_all_;
  QAction* action_export_as_;
  QAction* action_quit_;

  QAction* action_feature_extraction_;
  QAction* action_feature_matching_;
  QAction* action_database_management_;

  QAction* action_reconstruction_start_;
  QAction* action_reconstruction_step_;
  QAction* action_reconstruction_pause_;
  QAction* action_reconstruction_reset_;
  QAction* action_reconstruction_finish_;
  QAction* action_reconstruction_normalize_;
  QAction* action_reconstruction_options_;

  QAction* action_bundle_adjustment_;
  QAction* action_bundle_adjustment_finish_;
  QAction* action_bundle_adjustment_options_;

  QAction* action_render_;
  QAction* action_render_now_;
  QAction* action_render_toggle_;
  QAction* action_render_reset_view_;
  QAction* action_render_options_;

  QAction* action_reconstruction_stats_;
  QAction* action_match_matrix_;
  QAction* action_log_show_;
  QAction* action_grab_image_;
  QAction* action_grab_movie_;
  QAction* action_undistort_;
  QAction* action_extract_colors_;
  QAction* action_reset_options_;

  QAction* action_about_;
  QAction* action_license_;

  QProgressDialog* progress_bar_;

  QFutureWatcher<void>* import_watcher_;
  QFutureWatcher<void>* export_watcher_;
  QFutureWatcher<void>* extract_colors_watcher_;

  std::vector<QAction*> blocking_actions_;

  // Necessary for OS X to avoid duplicate closeEvents.
  bool window_closed_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_MAIN_WINDOW_H_
