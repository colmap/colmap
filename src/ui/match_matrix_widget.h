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

#ifndef COLMAP_SRC_UI_MATCH_MATRIX_WIDGET_H_
#define COLMAP_SRC_UI_MATCH_MATRIX_WIDGET_H_

#include "ui/image_viewer_widget.h"
#include "util/option_manager.h"

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

#endif  // COLMAP_SRC_UI_MATCH_MATRIX_WIDGET_H_
