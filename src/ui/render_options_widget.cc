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
                                         OpenGLWindow* opengl_window)
    : OptionsWidget(parent),
      counter(0),
      automatic_update(true),
      options_(options),
      opengl_window_(opengl_window),
      point3D_colormap_scale_(1),
      point3D_colormap_min_q_(0.02),
      point3D_colormap_max_q_(0.98) {
  bg_color_[0] = 1.0;
  bg_color_[1] = 1.0;
  bg_color_[2] = 1.0;

  setWindowFlags(Qt::Widget | Qt::WindowStaysOnTopHint | Qt::Tool);
  setWindowModality(Qt::NonModal);
  setWindowTitle("Render options");

  AddOptionDouble(&options->render->max_error, "Max. error [px]");
  AddOptionInt(&options->render->min_track_len, "Min. track length", 0);

  AddSpacer();

  projection_cb_ = new QComboBox(this);
  projection_cb_->addItem("Perspective");
  projection_cb_->addItem("Orthographic");

  QLabel* projection_label = new QLabel(tr("Projection"), this);
  projection_label->setFont(font());
  projection_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(projection_label, grid_layout_->rowCount(), 0);
  grid_layout_->addWidget(projection_cb_, grid_layout_->rowCount() - 1, 1);

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

  opengl_window_->Upload();
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

  opengl_window_->SetPointColormap(point3D_color_map);
}

void RenderOptionsWidget::ApplyBackgroundColor() {
  opengl_window_->SetBackgroundColor(bg_color_[0], bg_color_[1], bg_color_[2]);
}

void RenderOptionsWidget::SelectBackgroundColor() {
  const QColor initial_color(255 * bg_color_[0], 255 * bg_color_[1],
                             255 * bg_color_[2]);
  const QColor selected_color = QColorDialog::getColor(initial_color);
  bg_red_spinbox_->setValue(selected_color.red() / 255.0);
  bg_green_spinbox_->setValue(selected_color.green() / 255.0);
  bg_blue_spinbox_->setValue(selected_color.blue() / 255.0);
}

}  // namespace colmap
