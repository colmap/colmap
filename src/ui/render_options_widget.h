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

#ifndef COLMAP_SRC_UI_RENDER_OPTIONS_WIDGET_H_
#define COLMAP_SRC_UI_RENDER_OPTIONS_WIDGET_H_

#include <QtCore>
#include <QtWidgets>

#include "sfm/incremental_mapper.h"
#include "ui/opengl_window.h"
#include "ui/options_widget.h"

namespace colmap {

class RenderOptionsWidget : public OptionsWidget {
 public:
  RenderOptionsWidget(QWidget* parent, OptionManager* options,
                      OpenGLWindow* opengl_window);

  size_t counter;
  bool automatic_update;

  QAction* action_render_now;

 private:
  void closeEvent(QCloseEvent* event);

  void Apply();
  void ApplyProjection();
  void ApplyColormap();
  void ApplyBackgroundColor();

  void SelectBackgroundColor();

  OptionManager* options_;
  OpenGLWindow* opengl_window_;

  QComboBox* projection_cb_;
  QComboBox* point3D_colormap_cb_;
  double point3D_colormap_scale_;
  double point3D_colormap_min_q_;
  double point3D_colormap_max_q_;
  QDoubleSpinBox* bg_red_spinbox_;
  QDoubleSpinBox* bg_green_spinbox_;
  QDoubleSpinBox* bg_blue_spinbox_;
  double bg_color_[3];
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_RENDER_OPTIONS_WIDGET_H_
