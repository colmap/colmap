// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "ui/image_viewer_widget.h"

#include <iostream>

#include "base/camera_models.h"
#include "ui/opengl_window.h"
#include "util/bitmap.h"
#include "util/misc.h"

namespace colmap {

const double BasicImageViewerWidget::kZoomFactor = 1.33;

BasicImageViewerWidget::BasicImageViewerWidget(QWidget* parent,
                                               const std::string& switch_text)
    : QWidget(parent),
      current_scale_(1.0),
      switch_(true),
      switch_text_(switch_text) {
  setWindowFlags(Qt::Window);
  resize(parent->width() - 20, parent->height() - 20);

  QFont font;
  font.setPointSize(10);
  setFont(font);

  grid_ = new QGridLayout(this);
  grid_->setContentsMargins(5, 5, 5, 5);

  image_label_ = new QLabel(this);
  image_scroll_area_ = new QScrollArea(this);
  image_scroll_area_->setWidget(image_label_);
  image_scroll_area_->setSizePolicy(QSizePolicy::Expanding,
                                    QSizePolicy::Expanding);

  grid_->addWidget(image_scroll_area_, 1, 0);

  button_layout_ = new QHBoxLayout();

  show_button_ =
      new QPushButton(tr(std::string("Hide " + switch_text_).c_str()), this);
  show_button_->setFont(font);
  button_layout_->addWidget(show_button_);
  connect(show_button_, &QPushButton::released, this,
          &BasicImageViewerWidget::ShowOrHide);

  QPushButton* zoom_in_button = new QPushButton(tr("+"), this);
  zoom_in_button->setFont(font);
  zoom_in_button->setFixedWidth(50);
  button_layout_->addWidget(zoom_in_button);
  connect(zoom_in_button, &QPushButton::released, this,
          &BasicImageViewerWidget::ZoomIn);

  QPushButton* zoom_out_button = new QPushButton(tr("-"), this);
  zoom_out_button->setFont(font);
  zoom_out_button->setFixedWidth(50);
  button_layout_->addWidget(zoom_out_button);
  connect(zoom_out_button, &QPushButton::released, this,
          &BasicImageViewerWidget::ZoomOut);

  grid_->addLayout(button_layout_, 2, 0, Qt::AlignRight);
}

void BasicImageViewerWidget::closeEvent(QCloseEvent* event) {
  // Release the images, since zoomed in images can use a lot of memory
  image1_ = QPixmap();
  image2_ = QPixmap();
  image_label_->clear();
}

void BasicImageViewerWidget::Show(const std::string& path,
                                  const FeatureKeypoints& keypoints,
                                  const std::vector<char>& tri_mask) {
  Bitmap bitmap;
  if (!bitmap.Read(path, true)) {
    std::cerr << "ERROR: Cannot read image at path " << path << std::endl;
    return;
  }

  // Image without keypoints
  image1_ = QPixmap::fromImage(BitmapToQImageRGB(bitmap));

  // Image with keypoints
  image2_ = image1_;

  const size_t num_tri_keypoints = std::count_if(
      tri_mask.begin(), tri_mask.end(), [](const bool tri) { return tri; });

  FeatureKeypoints keypoints_tri(num_tri_keypoints);
  FeatureKeypoints keypoints_not_tri(keypoints.size() - num_tri_keypoints);
  size_t i_tri = 0;
  size_t i_not_tri = 0;
  for (size_t i = 0; i < tri_mask.size(); ++i) {
    if (tri_mask[i]) {
      keypoints_tri[i_tri] = keypoints[i];
      i_tri += 1;
    } else {
      keypoints_not_tri[i_not_tri] = keypoints[i];
      i_not_tri += 1;
    }
  }

  DrawKeypoints(&image2_, keypoints_tri, Qt::magenta);
  DrawKeypoints(&image2_, keypoints_not_tri, Qt::red);

  current_scale_ = 1.0;
  const double scale = (image_scroll_area_->height() - 5) /
                       static_cast<double>(image1_.height());
  ScaleImage(scale);
}

void BasicImageViewerWidget::ScaleImage(const double scale) {
  current_scale_ *= scale;

  const Qt::TransformationMode transform_mode =
      current_scale_ > 1.0 ? Qt::FastTransformation : Qt::SmoothTransformation;

  if (switch_) {
    image_label_->setPixmap(image2_.scaledToWidth(
        static_cast<int>(current_scale_ * image2_.width()), transform_mode));
  } else {
    image_label_->setPixmap(image1_.scaledToWidth(
        static_cast<int>(current_scale_ * image1_.width()), transform_mode));
  }

  image_label_->adjustSize();
}

void BasicImageViewerWidget::ZoomIn() { ScaleImage(kZoomFactor); }

void BasicImageViewerWidget::ZoomOut() { ScaleImage(1.0 / kZoomFactor); }

void BasicImageViewerWidget::ShowOrHide() {
  if (switch_) {
    show_button_->setText(tr(std::string("Show " + switch_text_).c_str()));
  } else {
    show_button_->setText(tr(std::string("Hide " + switch_text_).c_str()));
  }
  switch_ = !switch_;
  ScaleImage(1.0);
}

MatchesImageViewerWidget::MatchesImageViewerWidget(QWidget* parent)
    : BasicImageViewerWidget(parent, "matches") {}

void MatchesImageViewerWidget::Show(const std::string& path1,
                                    const std::string& path2,
                                    const FeatureKeypoints& keypoints1,
                                    const FeatureKeypoints& keypoints2,
                                    const FeatureMatches& matches) {
  Bitmap bitmap1;
  Bitmap bitmap2;
  if (!bitmap1.Read(path1, true) || !bitmap2.Read(path2, true)) {
    std::cerr << "ERROR: Cannot read images at paths " << path1 << " and "
              << path2 << std::endl;
    return;
  }

  const auto image1 = QPixmap::fromImage(BitmapToQImageRGB(bitmap1));
  const auto image2 = QPixmap::fromImage(BitmapToQImageRGB(bitmap2));

  image1_ = ShowImagesSideBySide(image1, image2);
  image2_ = DrawMatches(image1, image2, keypoints1, keypoints2, matches);

  current_scale_ = 1.0;
  const double scale = (image_scroll_area_->height() - 5) /
                       static_cast<double>(image1_.height());
  ScaleImage(scale);
}

ImageViewerWidget::ImageViewerWidget(QWidget* parent,
                                     OpenGLWindow* opengl_window,
                                     OptionManager* options)
    : BasicImageViewerWidget(parent, "keypoints"),
      opengl_window_(opengl_window),
      options_(options) {
  setWindowTitle("Image information");

  table_widget_ = new QTableWidget(this);
  table_widget_->setColumnCount(2);
  table_widget_->setRowCount(11);

  QFont font;
  font.setPointSize(10);
  table_widget_->setFont(font);

  table_widget_->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);

  table_widget_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table_widget_->setSelectionMode(QAbstractItemView::SingleSelection);
  table_widget_->setShowGrid(true);

  table_widget_->horizontalHeader()->setStretchLastSection(true);
  table_widget_->horizontalHeader()->setVisible(false);
  table_widget_->verticalHeader()->setVisible(false);
  table_widget_->verticalHeader()->setDefaultSectionSize(18);

  int row = 0;

  table_widget_->setItem(row, 0, new QTableWidgetItem("image_id"));
  image_id_item_ = new QTableWidgetItem();
  table_widget_->setItem(row, 1, image_id_item_);
  row += 1;

  table_widget_->setItem(row, 0, new QTableWidgetItem("camera_id"));
  camera_id_item_ = new QTableWidgetItem();
  table_widget_->setItem(row, 1, camera_id_item_);
  row += 1;

  table_widget_->setItem(row, 0, new QTableWidgetItem("camera_model"));
  camera_model_item_ = new QTableWidgetItem();
  table_widget_->setItem(row, 1, camera_model_item_);
  row += 1;

  table_widget_->setItem(row, 0, new QTableWidgetItem("camera_params"));
  camera_params_item_ = new QTableWidgetItem();
  table_widget_->setItem(row, 1, camera_params_item_);
  row += 1;

  table_widget_->setItem(row, 0, new QTableWidgetItem("qw, qx, qy, qz"));
  qvec_item_ = new QTableWidgetItem();
  table_widget_->setItem(row, 1, qvec_item_);
  row += 1;

  table_widget_->setItem(row, 0, new QTableWidgetItem("tx, ty, ty"));
  tvec_item_ = new QTableWidgetItem();
  table_widget_->setItem(row, 1, tvec_item_);
  row += 1;

  table_widget_->setItem(row, 0, new QTableWidgetItem("dims"));
  dimensions_item_ = new QTableWidgetItem();
  table_widget_->setItem(row, 1, dimensions_item_);
  row += 1;

  table_widget_->setItem(row, 0, new QTableWidgetItem("num_points2D"));
  num_points2D_item_ = new QTableWidgetItem();
  num_points2D_item_->setForeground(Qt::red);
  table_widget_->setItem(row, 1, num_points2D_item_);
  row += 1;

  table_widget_->setItem(row, 0, new QTableWidgetItem("num_points3D"));
  num_points3D_item_ = new QTableWidgetItem();
  num_points3D_item_->setForeground(Qt::magenta);
  table_widget_->setItem(row, 1, num_points3D_item_);
  row += 1;

  table_widget_->setItem(row, 0, new QTableWidgetItem("num_observations"));
  num_obs_item_ = new QTableWidgetItem();
  table_widget_->setItem(row, 1, num_obs_item_);
  row += 1;

  table_widget_->setItem(row, 0, new QTableWidgetItem("name"));
  name_item_ = new QTableWidgetItem();
  table_widget_->setItem(row, 1, name_item_);
  row += 1;

  grid_->addWidget(table_widget_, 0, 0);

  delete_button_ = new QPushButton(tr("Delete"), this);
  delete_button_->setFont(font);
  button_layout_->addWidget(delete_button_);
  connect(delete_button_, &QPushButton::released, this,
          &ImageViewerWidget::DeleteImage);
}

void ImageViewerWidget::Show(const image_t image_id) {
  if (opengl_window_->images.count(image_id) == 0) {
    return;
  }

  image_id_ = image_id;

  const Image& image = opengl_window_->images.at(image_id);
  const Camera& camera = opengl_window_->cameras.at(image.CameraId());

  image_id_item_->setText(QString::number(image_id));
  camera_id_item_->setText(QString::number(image.CameraId()));
  camera_model_item_->setText(QString::fromStdString(camera.ModelName()));
  camera_params_item_->setText(QString::fromStdString(camera.ParamsToString()));
  qvec_item_->setText(QString::number(image.Qvec(0)) + ", " +
                      QString::number(image.Qvec(1)) + ", " +
                      QString::number(image.Qvec(2)) + ", " +
                      QString::number(image.Qvec(3)));
  tvec_item_->setText(QString::number(image.Tvec(0)) + ", " +
                      QString::number(image.Tvec(1)) + ", " +
                      QString::number(image.Tvec(2)));
  dimensions_item_->setText(QString::number(camera.Width()) + "x" +
                            QString::number(camera.Height()));
  num_points2D_item_->setText(QString::number(image.NumPoints2D()));

  std::vector<char> tri_mask(image.NumPoints2D());
  for (size_t i = 0; i < image.NumPoints2D(); ++i) {
    tri_mask[i] = image.Point2D(i).HasPoint3D();
  }

  num_points3D_item_->setText(QString::number(image.NumPoints3D()));
  num_obs_item_->setText(QString::number(image.NumObservations()));
  name_item_->setText(QString::fromStdString(image.Name()));

  ResizeTable();

  FeatureKeypoints keypoints(image.NumPoints2D());
  for (point2D_t i = 0; i < image.NumPoints2D(); ++i) {
    keypoints[i].x = static_cast<float>(image.Point2D(i).X());
    keypoints[i].y = static_cast<float>(image.Point2D(i).Y());
  }

  const std::string path =
      EnsureTrailingSlash(*options_->image_path) + image.Name();
  BasicImageViewerWidget::Show(path, keypoints, tri_mask);
}

void ImageViewerWidget::ResizeTable() {
  // Set fixed table dimensions.
  table_widget_->resizeColumnsToContents();
  int height = table_widget_->horizontalHeader()->height() +
               2 * table_widget_->frameWidth();
  for (int i = 0; i < table_widget_->rowCount(); i++) {
    height += table_widget_->rowHeight(i);
  }
  table_widget_->setFixedHeight(height);
}

void ImageViewerWidget::DeleteImage() {
  QMessageBox::StandardButton reply = QMessageBox::question(
      this, "", tr("Do you really want to delete this image?"),
      QMessageBox::Yes | QMessageBox::No);
  if (reply == QMessageBox::Yes) {
    if (opengl_window_->reconstruction->ExistsImage(image_id_)) {
      opengl_window_->reconstruction->DeRegisterImage(image_id_);
    }
    opengl_window_->Update();
  }
  hide();
}

}  // namespace colmap
