// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/automatic_reconstruction.h"
#include "colmap/ui/options_widget.h"
#include "colmap/ui/thread_control_widget.h"

namespace colmap {

class MainWindow;

class AutomaticReconstructionWidget : public OptionsWidget {
 public:
  explicit AutomaticReconstructionWidget(MainWindow* main_window);

  void Run();

 private:
  void RenderResult();

  MainWindow* main_window_;
  AutomaticReconstructionController::Options options_;
  ThreadControlWidget* thread_control_widget_;
  QComboBox* data_type_cb_;
  QComboBox* quality_cb_;
  QComboBox* mesher_cb_;
#ifdef CASPAR_ENABLED
  QComboBox* ba_backend_cb_;
#endif
  QAction* render_result_;
};

}  // namespace colmap
