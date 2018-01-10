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

#include "ui/render_options_widget.h"

namespace colmap {

RenderOptionsWidget::RenderOptionsWidget(QWidget* parent,
                                         OptionManager* options,
                                         ModelViewerWidget* model_viewer_widget)
    : OptionsWidget(parent),
      counter(0),
      automatic_update(true),
      options_(options),
      model_viewer_widget_(model_viewer_widget),
      point3D_colormap_scale_(1),
      point3D_colormap_min_q_(0.02),
      point3D_colormap_max_q_(0.98) {
  bg_color_[0] = 1.0;
  bg_color_[1] = 1.0;
  bg_color_[2] = 1.0;

  setWindowFlags(Qt::Widget | Qt::WindowStaysOnTopHint | Qt::Tool);
  setWindowModality(Qt::NonModal);
  setWindowTitle("Render options");

  QLabel* point_size_label = new QLabel(tr("Point size"), this);
  point_size_label->setFont(font());
  point_size_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  QHBoxLayout* point_size_layout = new QHBoxLayout();
  QPushButton* decrease_point_size = new QPushButton("-", this);
  connect(decrease_point_size, &QPushButton::released, this,
          &RenderOptionsWidget::DecreasePointSize);
  QPushButton* increase_point_size = new QPushButton("+", this);
  connect(increase_point_size, &QPushButton::released, this,
          &RenderOptionsWidget::IncreasePointSize);
  point_size_layout->addWidget(decrease_point_size);
  point_size_layout->addWidget(increase_point_size);
  grid_layout_->addWidget(point_size_label, grid_layout_->rowCount(), 0);
  grid_layout_->addLayout(point_size_layout, grid_layout_->rowCount() - 1, 1);

  QLabel* camera_size_label = new QLabel(tr("Camera size"), this);
  camera_size_label->setFont(font());
  camera_size_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  QHBoxLayout* camera_size_layout = new QHBoxLayout();
  QPushButton* decrease_camera_size = new QPushButton("-", this);
  connect(decrease_camera_size, &QPushButton::released, this,
          &RenderOptionsWidget::DecreaseCameraSize);
  QPushButton* increase_camera_size = new QPushButton("+", this);
  connect(increase_camera_size, &QPushButton::released, this,
          &RenderOptionsWidget::IncreaseCameraSize);
  camera_size_layout->addWidget(decrease_camera_size);
  camera_size_layout->addWidget(increase_camera_size);
  grid_layout_->addWidget(camera_size_label, grid_layout_->rowCount(), 0);
  grid_layout_->addLayout(camera_size_layout, grid_layout_->rowCount() - 1, 1);

  AddSpacer();

  projection_cb_ = new QComboBox(this);
  projection_cb_->addItem("Perspective");
  projection_cb_->addItem("Orthographic");

  QLabel* projection_label = new QLabel(tr("Projection"), this);
  projection_label->setFont(font());
  projection_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(projection_label, grid_layout_->rowCount(), 0);
  grid_layout_->addWidget(projection_cb_, grid_layout_->rowCount() - 1, 1);

  AddSpacer();

  AddOptionDouble(&options->render->max_error, "Max. error [px]");
  AddOptionInt(&options->render->min_track_len, "Min. track length", 0);

  AddSpacer();

  point3D_colormap_cb_ = new QComboBox(this);
  point3D_colormap_cb_->addItem("Photometric");
  point3D_colormap_cb_->addItem("Error");
  point3D_colormap_cb_->addItem("Track-Length");
  point3D_colormap_cb_->addItem("Ground-Resolution");

  QLabel* colormap_label = new QLabel(tr("Point colormap"), this);
  colormap_label->setFont(font());
  colormap_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(colormap_label, grid_layout_->rowCount(), 0);
  grid_layout_->addWidget(point3D_colormap_cb_, grid_layout_->rowCount() - 1,
                          1);

  AddOptionDouble(&point3D_colormap_min_q_, "Point colormap minq", 0, 1, 0.001,
                  3);
  AddOptionDouble(&point3D_colormap_max_q_, "Point colormap maxq", 0, 1, 0.001,
                  3);
  AddOptionDouble(&point3D_colormap_scale_, "Point colormap scale", -1e7, 1e7);

  AddSpacer();

  bg_red_spinbox_ =
      AddOptionDouble(&bg_color_[0], "Background red", 0.0, 1.0, 0.001, 3);
  bg_green_spinbox_ =
      AddOptionDouble(&bg_color_[1], "Background green", 0.0, 1.0, 0.001, 3);
  bg_blue_spinbox_ =
      AddOptionDouble(&bg_color_[2], "Background blue", 0.0, 1.0, 0.001, 3);

  QPushButton* select_color = new QPushButton(tr("Select color"), this);
  grid_layout_->addWidget(select_color, grid_layout_->rowCount(), 1);
  connect(select_color, &QPushButton::released, this,
          &RenderOptionsWidget::SelectBackgroundColor);

  AddSpacer();

  AddOptionBool(&options->render->adapt_refresh_rate, "Adaptive refresh rate");
  AddOptionInt(&options->render->refresh_rate, "Refresh rate [frames]", 1);

  AddSpacer();

  AddOptionBool(&options->render->image_connections, "Image connections");

  AddSpacer();

  QPushButton* apply = new QPushButton(tr("Apply"), this);
  grid_layout_->addWidget(apply, grid_layout_->rowCount(), 1);
  connect(apply, &QPushButton::released, this, &RenderOptionsWidget::Apply);
}

void RenderOptionsWidget::closeEvent(QCloseEvent* event) {
  // Just overwrite parent closeEvent to prevent automatic write of options
}

void RenderOptionsWidget::Apply() {
  WriteOptions();

  counter = 0;

  ApplyProjection();
  ApplyColormap();
  ApplyBackgroundColor();

  model_viewer_widget_->ReloadReconstruction();
}

void RenderOptionsWidget::ApplyProjection() {
  switch (projection_cb_->currentIndex()) {
    case 0:
      options_->render->projection_type =
          RenderOptions::ProjectionType::PERSPECTIVE;
      break;
    case 1:
      options_->render->projection_type =
          RenderOptions::ProjectionType::ORTHOGRAPHIC;
      break;
    default:
      options_->render->projection_type =
          RenderOptions::ProjectionType::PERSPECTIVE;
      break;
  }
}

void RenderOptionsWidget::ApplyColormap() {
  PointColormapBase* point3D_color_map;

  switch (point3D_colormap_cb_->currentIndex()) {
    case 0:
      point3D_color_map = new PointColormapPhotometric();
      break;
    case 1:
      point3D_color_map = new PointColormapError();
      break;
    case 2:
      point3D_color_map = new PointColormapTrackLen();
      break;
    case 3:
      point3D_color_map = new PointColormapGroundResolution();
      break;
    default:
      point3D_color_map = new PointColormapPhotometric();
      break;
  }

  point3D_color_map->scale = static_cast<float>(point3D_colormap_scale_);
  point3D_color_map->min_q = static_cast<float>(point3D_colormap_min_q_);
  point3D_color_map->max_q = static_cast<float>(point3D_colormap_max_q_);

  model_viewer_widget_->SetPointColormap(point3D_color_map);
}

void RenderOptionsWidget::ApplyBackgroundColor() {
  model_viewer_widget_->SetBackgroundColor(bg_color_[0], bg_color_[1],
                                           bg_color_[2]);
}

void RenderOptionsWidget::SelectBackgroundColor() {
  const QColor initial_color(255 * bg_color_[0], 255 * bg_color_[1],
                             255 * bg_color_[2]);
  const QColor selected_color = QColorDialog::getColor(initial_color);
  bg_red_spinbox_->setValue(selected_color.red() / 255.0);
  bg_green_spinbox_->setValue(selected_color.green() / 255.0);
  bg_blue_spinbox_->setValue(selected_color.blue() / 255.0);
}

void RenderOptionsWidget::IncreasePointSize() {
  const float kDelta = 100;
  model_viewer_widget_->ChangePointSize(kDelta);
}

void RenderOptionsWidget::DecreasePointSize() {
  const float kDelta = -100;
  model_viewer_widget_->ChangePointSize(kDelta);
}

void RenderOptionsWidget::IncreaseCameraSize() {
  const float kDelta = 100;
  model_viewer_widget_->ChangeCameraSize(kDelta);
}

void RenderOptionsWidget::DecreaseCameraSize() {
  const float kDelta = -100;
  model_viewer_widget_->ChangeCameraSize(kDelta);
}

}  // namespace colmap
