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

#ifndef COLMAP_SRC_UI_DENSE_RECONSTRUCTION_WIDGET_H_
#define COLMAP_SRC_UI_DENSE_RECONSTRUCTION_WIDGET_H_

#include <QtCore>
#include <QtWidgets>

#include "mvs/fusion.h"
#include "ui/image_viewer_widget.h"
#include "ui/options_widget.h"
#include "ui/thread_control_widget.h"
#include "util/option_manager.h"

namespace colmap {

class MainWindow;

class DenseReconstructionOptionsWidget : public QWidget {
 public:
  DenseReconstructionOptionsWidget(QWidget* parent, OptionManager* options);
};

class DenseReconstructionWidget : public QWidget {
 public:
  DenseReconstructionWidget(MainWindow* main_window, OptionManager* options);

  void Show(Reconstruction* reconstruction);

 private:
  void showEvent(QShowEvent* event);

  void Undistort();
  void Stereo();
  void Fusion();
  void Meshing();

  void SelectWorkspacePath();
  std::string GetWorkspacePath();
  void RefreshWorkspace();

  void WriteFusedPoints();
  void ShowMeshingInfo();

  QWidget* GenerateTableButtonWidget(const std::string& image_name,
                                     const std::string& type);

  MainWindow* main_window_;
  OptionManager* options_;
  Reconstruction* reconstruction_;
  ThreadControlWidget* thread_control_widget_;
  DenseReconstructionOptionsWidget* options_widget_;
  ImageViewerWidget* image_viewer_widget_;
  QLineEdit* workspace_path_text_;
  QTableWidget* table_widget_;
  QPushButton* undistortion_button_;
  QPushButton* stereo_button_;
  QPushButton* fusion_button_;
  QPushButton* meshing_button_;
  QAction* refresh_workspace_action_;
  QAction* write_fused_points_action_;
  QAction* show_meshing_info_action_;

  bool photometric_done_;
  bool geometric_done_;

  std::string images_path_;
  std::string depth_maps_path_;
  std::string normal_maps_path_;

  std::vector<mvs::FusedPoint> fused_points_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_DENSE_RECONSTRUCTION_WIDGET_H_
