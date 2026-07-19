// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/ui/render_options_widget.h"

#include "colmap/ui/colormaps.h"
#include "colmap/ui/render_options.h"
#include "colmap/util/logging.h"
#include "colmap/util/string.h"

#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>

namespace colmap {

RenderOptionsWidget::RenderOptionsWidget(QWidget* parent,
                                         OptionManager* options,
                                         ModelViewerWidget* model_viewer_widget)
    : OptionsWidget(parent),
      counter(0),
      automatic_update(true),
      options_(options),
      model_viewer_widget_(model_viewer_widget),
      background_color_(1.0f, 1.0f, 1.0f, 1.0f),
      point3D_colormap_scale_(1),
      point3D_colormap_min_q_(0.02),
      point3D_colormap_max_q_(0.98),
      image_plane_color_(ImageColormapUniform::kDefaultPlaneColor),
      image_frame_color_(ImageColormapUniform::kDefaultFrameColor),
      image_colormap_max_error_(2.0) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Render options");

  QHBoxLayout* point_size_layout = new QHBoxLayout();
  QPushButton* decrease_point_size = new QPushButton("-", this);
  connect(decrease_point_size,
          &QPushButton::released,
          this,
          &RenderOptionsWidget::DecreasePointSize);
  QPushButton* increase_point_size = new QPushButton("+", this);
  connect(increase_point_size,
          &QPushButton::released,
          this,
          &RenderOptionsWidget::IncreasePointSize);
  point_size_layout->addWidget(decrease_point_size);
  point_size_layout->addWidget(increase_point_size);
  AddLayoutRow("Point size", point_size_layout);

  QHBoxLayout* camera_size_layout = new QHBoxLayout();
  QPushButton* decrease_camera_size = new QPushButton("-", this);
  connect(decrease_camera_size,
          &QPushButton::released,
          this,
          &RenderOptionsWidget::DecreaseCameraSize);
  QPushButton* increase_camera_size = new QPushButton("+", this);
  connect(increase_camera_size,
          &QPushButton::released,
          this,
          &RenderOptionsWidget::IncreaseCameraSize);
  camera_size_layout->addWidget(decrease_camera_size);
  camera_size_layout->addWidget(increase_camera_size);
  AddLayoutRow("Camera size", camera_size_layout);

  AddSpacer();

  projection_cb_ = new QComboBox(this);
  projection_cb_->addItem("Perspective");
  projection_cb_->addItem("Orthographic");
  AddWidgetRow("Projection", projection_cb_);

  AddSpacer();

  QPushButton* select_background_color =
      new QPushButton(tr("Select color"), this);
  grid_layout_->addWidget(
      select_background_color, grid_layout_->rowCount() - 1, 1);
  connect(select_background_color, &QPushButton::released, this, [&]() {
    SelectColor("Background color", &background_color_);
  });
  AddWidgetRow("Background", select_background_color);

  AddSpacer();

  AddOptionDouble(&options->render->max_error, "Point max. error [px]");
  AddOptionInt(&options->render->min_track_len, "Point min. track length", 0);

  AddSpacer();

  point3D_colormap_cb_ = new QComboBox(this);
  point3D_colormap_cb_->addItem("Photometric");
  point3D_colormap_cb_->addItem("Error");
  point3D_colormap_cb_->addItem("Track-Length");
  point3D_colormap_cb_->addItem("Ground-Resolution");
  AddWidgetRow("Point colormap", point3D_colormap_cb_);

  AddOptionDouble(
      &point3D_colormap_min_q_, "Point colormap minq", 0, 1, 0.001, 3);
  AddOptionDouble(
      &point3D_colormap_max_q_, "Point colormap maxq", 0, 1, 0.001, 3);
  AddOptionDouble(&point3D_colormap_scale_, "Point colormap scale", -1e7, 1e7);

  // Show the above items only for other colormaps than the photometric one.
  HideOption(&point3D_colormap_min_q_);
  HideOption(&point3D_colormap_max_q_);
  HideOption(&point3D_colormap_scale_);
  connect(point3D_colormap_cb_,
          (void (QComboBox::*)(int))&QComboBox::currentIndexChanged,
          this,
          &RenderOptionsWidget::SelectPointColormap);

  AddSpacer();

  image_colormap_cb_ = new QComboBox(this);
  image_colormap_cb_->addItem("Uniform color");
  image_colormap_cb_->addItem("Images with words in name");
  image_colormap_cb_->addItem("Reprojection error");
  AddWidgetRow("Image colormap", image_colormap_cb_);

  // Upper bound of the absolute reprojection-error color scale (0 px = blue,
  // this value = red). Shown only for the reprojection-error colormap.
  QDoubleSpinBox* image_colormap_max_error_sb = AddOptionDouble(
      &image_colormap_max_error_, "Reproj. error for red [px]", 0.1, 1e4, 0.1, 2);
  image_colormap_max_error_sb->setToolTip(
      "Absolute color scale for the reprojection-error colormap.\n"
      "A camera frustum whose mean reprojection error is 0 px is blue; one at\n"
      "this value (or higher) is red; values in between fade blue->green->red.\n"
      "\n"
      "Lower it to spread the colors across low-error cameras (see fine\n"
      "differences); raise it so only genuinely high-error cameras stand out.\n"
      "Your model's actual per-image error range is printed to the Log when you\n"
      "click Apply.");
  HideOption(&image_colormap_max_error_);

  select_image_plane_color_ = new QPushButton(tr("Select color"), this);
  connect(select_image_plane_color_, &QPushButton::released, this, [&]() {
    SelectColor("Image plane color", &image_plane_color_);
  });
  AddWidgetRow("Image plane", select_image_plane_color_);

  select_image_frame_color_ = new QPushButton(tr("Select color"), this);
  connect(select_image_frame_color_, &QPushButton::released, this, [&]() {
    SelectColor("Image frame color", &image_frame_color_);
  });
  AddWidgetRow("Image frame", select_image_frame_color_);

  AddOptionBool(&options->render->show_camera_orientation,
                "Camera orientation\n(top-left corner)");

  image_colormap_name_filter_layout_ = new QHBoxLayout();
  QPushButton* image_colormap_add_word = new QPushButton("Add", this);
  connect(image_colormap_add_word,
          &QPushButton::released,
          this,
          &RenderOptionsWidget::ImageColormapNameFilterAddWord);
  QPushButton* image_colormap_clear_words = new QPushButton("Clear", this);
  connect(image_colormap_clear_words,
          &QPushButton::released,
          this,
          &RenderOptionsWidget::ImageColormapNameFilterClearWords);
  image_colormap_name_filter_layout_->addWidget(image_colormap_add_word);
  image_colormap_name_filter_layout_->addWidget(image_colormap_clear_words);
  AddLayoutRow("Words", image_colormap_name_filter_layout_);

  HideLayout(image_colormap_name_filter_layout_);
  connect(image_colormap_cb_,
          (void (QComboBox::*)(int))&QComboBox::currentIndexChanged,
          this,
          &RenderOptionsWidget::SelectImageColormap);

  AddSpacer();

  AddOptionBool(&options->render->adapt_refresh_rate, "Adaptive refresh rate");
  AddOptionInt(&options->render->refresh_rate, "Refresh rate [frames]", 1);

  AddSpacer();

  AddOptionBool(&options->render->image_connections, "Image connections");

  AddSpacer();

  AddOptionBool(&options->render->mesh_wireframe, "Mesh wireframe");
  AddOptionBool(&options->render->mesh_color, "Mesh color");

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
  ApplyPointColormap();
  ApplyImageColormap();
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

void RenderOptionsWidget::ApplyPointColormap() {
  std::unique_ptr<PointColormapBase> point3D_color_map;

  switch (point3D_colormap_cb_->currentIndex()) {
    case 0:
      point3D_color_map = std::make_unique<PointColormapPhotometric>();
      break;
    case 1:
      point3D_color_map = std::make_unique<PointColormapError>();
      break;
    case 2:
      point3D_color_map = std::make_unique<PointColormapTrackLen>();
      break;
    case 3:
      point3D_color_map = std::make_unique<PointColormapGroundResolution>();
      break;
    default:
      point3D_color_map = std::make_unique<PointColormapPhotometric>();
      break;
  }

  point3D_color_map->scale = static_cast<float>(point3D_colormap_scale_);
  point3D_color_map->min_q = static_cast<float>(point3D_colormap_min_q_);
  point3D_color_map->max_q = static_cast<float>(point3D_colormap_max_q_);

  model_viewer_widget_->SetPointColormap(std::move(point3D_color_map));
}

void RenderOptionsWidget::ApplyImageColormap() {
  std::unique_ptr<ImageColormapBase> image_color_map;

  switch (image_colormap_cb_->currentIndex()) {
    case 0: {
      auto uniform = std::make_unique<ImageColormapUniform>();
      uniform->uniform_plane_color = image_plane_color_;
      uniform->uniform_frame_color = image_frame_color_;
      image_color_map = std::move(uniform);
      break;
    }
    case 1:
      image_color_map = std::make_unique<ImageColormapNameFilter>(
          image_colormap_name_filter_);
      break;
    case 2: {
      auto reproj_error = std::make_unique<ImageColormapReprojectionError>();
      reproj_error->max_error = static_cast<float>(image_colormap_max_error_);
      image_color_map = std::move(reproj_error);
      // Print the model's actual per-image error range so the abstract color
      // scale maps to concrete numbers the user can reason about.
      LogReprojectionErrorScale();
      break;
    }
    default:
      image_color_map = std::make_unique<ImageColormapUniform>();
      break;
  }

  model_viewer_widget_->SetImageColormap(std::move(image_color_map));
}

void RenderOptionsWidget::LogReprojectionErrorScale() {
  const std::shared_ptr<Reconstruction>& reconstruction =
      model_viewer_widget_->reconstruction;
  if (reconstruction == nullptr) {
    return;
  }
  // Mean reprojection error per image, accumulated from the 3D point tracks.
  std::unordered_map<image_t, std::pair<double, int>> error_sums;
  for (const auto& point3D : reconstruction->Points3D()) {
    for (const auto& track_el : point3D.second.track.Elements()) {
      auto& sum = error_sums[track_el.image_id];
      sum.first += point3D.second.error;
      sum.second += 1;
    }
  }
  if (error_sums.empty()) {
    return;
  }
  double min_error = std::numeric_limits<double>::max();
  double max_error = 0.0;
  double sum_error = 0.0;
  for (const auto& [image_id, sum] : error_sums) {
    const double mean_error = sum.second > 0 ? sum.first / sum.second : 0.0;
    min_error = std::min(min_error, mean_error);
    max_error = std::max(max_error, mean_error);
    sum_error += mean_error;
  }
  LOG(INFO) << StringPrintf(
      "Reprojection-error colormap: %zu images, per-image error %.2f-%.2f px "
      "(mean %.2f px). Color scale: 0 px = blue, %.2f px = red.",
      error_sums.size(),
      min_error,
      max_error,
      sum_error / error_sums.size(),
      image_colormap_max_error_);
}

void RenderOptionsWidget::ApplyBackgroundColor() {
  model_viewer_widget_->SetBackgroundColor(
      background_color_(0), background_color_(1), background_color_(2));
}

void RenderOptionsWidget::SelectColor(const std::string& title,
                                      Eigen::Vector4f* color) {
  const QColor initial_color(static_cast<int>(255 * (*color)(0)),
                             static_cast<int>(255 * (*color)(1)),
                             static_cast<int>(255 * (*color)(2)),
                             static_cast<int>(255 * (*color)(3)));
  const QColor selected_color =
      QColorDialog::getColor(initial_color, this, title.c_str());
  (*color)(0) = selected_color.red() / 255.0;
  (*color)(1) = selected_color.green() / 255.0;
  (*color)(2) = selected_color.blue() / 255.0;
  (*color)(3) = selected_color.alpha() / 255.0;
}

void RenderOptionsWidget::SelectPointColormap(const int idx) {
  if (idx == 0) {
    HideOption(&point3D_colormap_scale_);
    HideOption(&point3D_colormap_min_q_);
    HideOption(&point3D_colormap_max_q_);
  } else {
    ShowOption(&point3D_colormap_scale_);
    ShowOption(&point3D_colormap_min_q_);
    ShowOption(&point3D_colormap_max_q_);
  }
}

void RenderOptionsWidget::SelectImageColormap(const int idx) {
  if (idx == 0) {
    ShowWidget(select_image_plane_color_);
    ShowWidget(select_image_frame_color_);
    HideLayout(image_colormap_name_filter_layout_);
    HideOption(&image_colormap_max_error_);
  } else if (idx == 1) {
    HideWidget(select_image_plane_color_);
    HideWidget(select_image_frame_color_);
    ShowLayout(image_colormap_name_filter_layout_);
    HideOption(&image_colormap_max_error_);
  } else {
    // Reprojection-error colormap: only the absolute error scale is adjustable.
    HideWidget(select_image_plane_color_);
    HideWidget(select_image_frame_color_);
    HideLayout(image_colormap_name_filter_layout_);
    ShowOption(&image_colormap_max_error_);
  }
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

void RenderOptionsWidget::ImageColormapNameFilterAddWord() {
  bool word_ok;
  const QString word =
      QInputDialog::getText(this, "", "Word:", QLineEdit::Normal, "", &word_ok);
  if (!word_ok || word == "") {
    return;
  }

  Eigen::Vector4f plane_color(ImageColormapBase::kDefaultPlaneColor);
  SelectColor("Image plane color", &plane_color);

  Eigen::Vector4f frame_color(ImageColormapBase::kDefaultFrameColor);
  SelectColor("Image frame color", &frame_color);

  image_colormap_name_filter_.AddColorForWord(
      word.toUtf8().constData(), plane_color, frame_color);
}

void RenderOptionsWidget::ImageColormapNameFilterClearWords() {
  image_colormap_name_filter_ = ImageColormapNameFilter();
}

}  // namespace colmap
