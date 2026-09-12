// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/controllers/undistorters.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/ui/options_widget.h"
#include "colmap/ui/thread_control_widget.h"

#include <QtCore>
#include <QtWidgets>

namespace colmap {

class UndistortionWidget : public OptionsWidget {
 public:
  UndistortionWidget(QWidget* parent, const OptionManager* options);

  void Show(std::shared_ptr<const Reconstruction> reconstruction);
  bool IsValid() const;

 private:
  void Undistort();

  const OptionManager* options_;
  std::shared_ptr<const Reconstruction> reconstruction_;

  ThreadControlWidget* thread_control_widget_;

  QComboBox* output_format_;
  COLMAPUndistorter::Options colmap_options_;
  UndistortCameraOptions camera_options_;
  std::filesystem::path output_path_;
  int num_threads_ = -1;
};

}  // namespace colmap
