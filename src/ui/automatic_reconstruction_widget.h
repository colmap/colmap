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

#ifndef COLMAP_SRC_UI_AUTOMATIC_RECONSTRUCTION_WIDGET_H_
#define COLMAP_SRC_UI_AUTOMATIC_RECONSTRUCTION_WIDGET_H_

#include "controllers/automatic_reconstruction.h"
#include "ui/options_widget.h"
#include "ui/thread_control_widget.h"

namespace colmap {

class MainWindow;

class AutomaticReconstructionWidget : public OptionsWidget {
 public:
  AutomaticReconstructionWidget(MainWindow* main_window);

  void Run();

 private:
  void RenderResult();

  MainWindow* main_window_;
  AutomaticReconstructionController::Options options_;
  ThreadControlWidget* thread_control_widget_;
  QComboBox* data_type_cb_;
  QComboBox* quality_cb_;
  QAction* render_result_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_AUTOMATIC_RECONSTRUCTION_WIDGET_H_
