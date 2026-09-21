// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/ui/model_viewer_widget.h"
#include "colmap/ui/options_widget.h"

#include <QtCore>
#include <QtWidgets>

namespace colmap {

class RenderOptionsWidget : public OptionsWidget {
 public:
  RenderOptionsWidget(QWidget* parent,
                      OptionManager* options,
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
  void SelectPointColormap(int idx);
  void SelectImageColormap(int idx);

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
