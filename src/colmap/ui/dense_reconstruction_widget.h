// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/mvs/fusion.h"
#include "colmap/ui/image_viewer_widget.h"
#include "colmap/ui/options_widget.h"
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
  std::string GetWorkspacePath();
  void RefreshWorkspace();

  void WriteFusedPoints();
  void ShowMeshingInfo();

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
  QAction* show_meshing_info_action_;

  bool photometric_done_;
  bool geometric_done_;

  std::string images_path_;
  std::string depth_maps_path_;
  std::string normal_maps_path_;

  std::vector<PlyPoint> fused_points_;
  std::vector<std::vector<int>> fused_points_visibility_;
};

}  // namespace colmap
