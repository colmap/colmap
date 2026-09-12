// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/ui/image_viewer_widget.h"
#include "colmap/ui/thread_control_widget.h"

#include <QtCore>
#include <QtWidgets>

namespace colmap {

class MainWindow;

class DenseReconstructionOptionsWidget : public QWidget {
 public:
  DenseReconstructionOptionsWidget(QWidget* parent, OptionManager* options);
};

class DenseReconstructionWidget : public QWidget {
 public:
  DenseReconstructionWidget(MainWindow* main_window, OptionManager* options);

  void Show(std::shared_ptr<const Reconstruction> reconstruction);

 private:
  void showEvent(QShowEvent* event);

  void Undistort();
  void Stereo();
  void Fusion();
  void PoissonMeshing();
  void DelaunayMeshing();

  void SelectWorkspacePath();
  std::filesystem::path GetWorkspacePath();
  void RefreshWorkspace();

  void WriteFusedPoints();
  void WriteSurfaceMesh();
  void LoadAndDisplayMesh(const std::filesystem::path& mesh_path);

  QWidget* GenerateTableButtonWidget(const std::string& image_name,
                                     const std::string& type);

  MainWindow* main_window_;
  OptionManager* options_;
  std::shared_ptr<const Reconstruction> reconstruction_;
  ThreadControlWidget* thread_control_widget_;
  DenseReconstructionOptionsWidget* options_widget_;
  ImageViewerWidget* image_viewer_widget_;
  QLineEdit* workspace_path_text_;
  QTableWidget* table_widget_;
  QPushButton* undistortion_button_;
  QPushButton* stereo_button_;
  QPushButton* fusion_button_;
  QPushButton* poisson_meshing_button_;
  QPushButton* delaunay_meshing_button_;
  QAction* refresh_workspace_action_;
  QAction* write_fused_points_action_;
  QAction* write_surface_mesh_action_;

  bool photometric_done_;
  bool geometric_done_;

  std::filesystem::path images_path_;
  std::filesystem::path depth_maps_path_;
  std::filesystem::path normal_maps_path_;
};

}  // namespace colmap
