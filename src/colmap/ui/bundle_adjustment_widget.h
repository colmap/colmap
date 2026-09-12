// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/ui/options_widget.h"
#include "colmap/ui/thread_control_widget.h"

#include <QtCore>
#include <QtWidgets>

namespace colmap {

class MainWindow;

class BundleAdjustmentWidget : public OptionsWidget {
 public:
  BundleAdjustmentWidget(MainWindow* main_window, OptionManager* options);

  void Show(std::shared_ptr<Reconstruction> reconstruction);

 private:
  void Run();
  void Render();

  MainWindow* main_window_;
  OptionManager* options_;
  std::shared_ptr<Reconstruction> reconstruction_;
  ThreadControlWidget* thread_control_widget_;
  QAction* render_action_;
};

}  // namespace colmap
