// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_UI_RENDER_OPTIONS_WIDGET_H_
#define COLMAP_SRC_UI_RENDER_OPTIONS_WIDGET_H_

#include <QtCore>
#include <QtWidgets>

#include "sfm/incremental_mapper.h"
#include "ui/model_viewer_widget.h"
#include "ui/options_widget.h"

namespace colmap {

class RenderOptionsWidget : public OptionsWidget {
 public:
  RenderOptionsWidget(QWidget* parent, OptionManager* options,
                      ModelViewerWidget* model_viewer_widget);

  size_t counter;
  bool automatic_update;

  QAction* action_render_now;

 private:
  void closeEvent(QCloseEvent* event);

  void Apply();
  void ApplyProjection();
  void ApplyPointColormap();
  void ApplyImageColormap();
  void ApplyBackgroundColor();

  void SelectColor(const std::string& title, Eigen::Vector4f* color);
  void SelectPointColormap(const int idx);
  void SelectImageColormap(const int idx);

  void IncreasePointSize();
  void DecreasePointSize();
  void IncreaseCameraSize();
  void DecreaseCameraSize();

  void ImageColormapNameFilterAddWord();
  void ImageColormapNameFilterClearWords();

  OptionManager* options_;
  ModelViewerWidget* model_viewer_widget_;

  Eigen::Vector4f background_color_;

  QComboBox* projection_cb_;

  QComboBox* point3D_colormap_cb_;

  double point3D_colormap_scale_;
  double point3D_colormap_min_q_;
  double point3D_colormap_max_q_;

  QComboBox* image_colormap_cb_;
  QPushButton* select_image_plane_color_;
  QPushButton* select_image_frame_color_;
  QHBoxLayout* image_colormap_name_filter_layout_;
  Eigen::Vector4f image_plane_color_;
  Eigen::Vector4f image_frame_color_;
  ImageColormapNameFilter image_colormap_name_filter_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_RENDER_OPTIONS_WIDGET_H_
