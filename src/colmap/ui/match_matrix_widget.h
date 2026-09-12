// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/ui/image_viewer_widget.h"

namespace colmap {

// Widget to visualize match matrix.
class MatchMatrixWidget : public ImageViewerWidget {
 public:
  MatchMatrixWidget(QWidget* parent, OptionManager* options);

  void Show();

 private:
  OptionManager* options_;
};

}  // namespace colmap
