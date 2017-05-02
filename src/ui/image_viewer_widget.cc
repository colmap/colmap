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

#include "ui/image_viewer_widget.h"

#include "ui/opengl_window.h"
#include "util/misc.h"

namespace colmap {

const double ImageViewerWidget::kZoomFactor = 1.25;

ImageViewerWidget::ImageViewerWidget(QWidget* parent)
    : QWidget(parent), zoom_scale_(1.0) {
  setWindowFlags(Qt::Window);
  resize(parent->width() - 20, parent->height() - 20);

  QFont font;
  font.setPointSize(10);
  setFont(font);

  grid_layout_ = new QGridLayout(this);
  grid_layout_->setContentsMargins(5, 5, 5, 5);

  image_label_ = new QLabel(this);
  image_scroll_area_ = new QScrollArea(this);
  image_scroll_area_->setWidget(image_label_);
  image_scroll_area_->setSizePolicy(QSizePolicy::Expanding,
                                    QSizePolicy::Expanding);

  grid_layout_->addWidget(image_scroll_area_, 1, 0);

  button_layout_ = new QHBoxLayout();

  QPushButton* zoom_in_button = new QPushButton("+", this);
  zoom_in_button->setFont(font);
  zoom_in_button->setFixedWidth(50);
  button_layout_->addWidget(zoom_in_button);
  connect(zoom_in_button, &QPushButton::released, this,
          &ImageViewerWidget::ZoomIn);

  QPushButton* zoom_out_button = new QPushButton("-", this);
  zoom_out_button->setFont(font);
  zoom_out_button->setFixedWidth(50);
  button_layout_->addWidget(zoom_out_button);
  connect(zoom_out_button, &QPushButton::released, this,
          &ImageViewerWidget::ZoomOut);

  grid_layout_->addLayout(button_layout_, 2, 0, Qt::AlignRight);
}

void ImageViewerWidget::closeEvent(QCloseEvent* event) {
  pixmap_ = QPixmap();
  image_label_->clear();
}

void ImageViewerWidget::ShowBitmap(const Bitmap& bitmap, const bool rescale) {
  ShowPixmap(QPixmap::fromImage(BitmapToQImageRGB(bitmap)), rescale);
}

void ImageViewerWidget::ShowPixmap(const QPixmap& pixmap, const bool rescale) {
  pixmap_ = pixmap;

  show();
  raise();

  if (rescale) {
    zoom_scale_ = 1.0;

    const double kScrollbarMargin = 5;
    const double scale_x = (image_scroll_area_->width() - kScrollbarMargin) /
                           static_cast<double>(pixmap_.width());
    const double scale_y = (image_scroll_area_->height() - kScrollbarMargin) /
                           static_cast<double>(pixmap_.height());
    const double scale = std::min(scale_x, scale_y);

    Rescale(scale);
  } else {
    Rescale(1.0);
  }
}

void ImageViewerWidget::ReadAndShow(const std::string& path,
                                    const bool rescale) {
  Bitmap bitmap;
  if (!bitmap.Read(path, true)) {
    std::cerr << "ERROR: Cannot read image at path " << path << std::endl;
  }

  ShowBitmap(bitmap, rescale);
}

void ImageViewerWidget::Rescale(const double scale) {
  if (pixmap_.isNull()) {
    return;
  }

  zoom_scale_ *= scale;

  const Qt::TransformationMode transform_mode =
      zoom_scale_ > 1.0 ? Qt::FastTransformation : Qt::SmoothTransformation;
  const int scaled_width =
      static_cast<int>(std::round(zoom_scale_ * pixmap_.width()));
  image_label_->setPixmap(pixmap_.scaledToWidth(scaled_width, transform_mode));
  image_label_->adjustSize();
}

void ImageViewerWidget::ZoomIn() { Rescale(kZoomFactor); }

void ImageViewerWidget::ZoomOut() { Rescale(1.0 / kZoomFactor); }

FeatureImageViewerWidget::FeatureImageViewerWidget(
    QWidget* parent, const std::string& switch_text)
    : ImageViewerWidget(parent),
      switch_state_(true),
      switch_text_(switch_text) {
  switch_button_ = new QPushButton(tr(("Hide " + switch_text_).c_str()), this);
  switch_button_->setFont(font());
  button_layout_->addWidget(switch_button_);
  connect(switch_button_, &QPushButton::released, this,
          &FeatureImageViewerWidget::ShowOrHide);
}

void FeatureImageViewerWidget::ReadAndShowWithKeypoints(
    const std::string& path, const FeatureKeypoints& keypoints,
    const std::vector<char>& tri_mask) {
  Bitmap bitmap;
  if (!bitmap.Read(path, true)) {
    std::cerr << "ERROR: Cannot read image at path " << path << std::endl;
  }

  image1_ = QPixmap::fromImage(BitmapToQImageRGB(bitmap));
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

  switch_state_ = true;
  ShowPixmap(image2_, true);
}

void FeatureImageViewerWidget::ReadAndShowWithMatches(
    const std::string& path1, const std::string& path2,
    const FeatureKeypoints& keypoints1, const FeatureKeypoints& keypoints2,
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

  switch_state_ = true;
  ShowPixmap(image2_, true);
}

void FeatureImageViewerWidget::ShowOrHide() {
  if (switch_state_) {
    switch_button_->setText(std::string("Show " + switch_text_).c_str());
    ShowPixmap(image1_, false);
    switch_state_ = false;
  } else {
    switch_button_->setText(std::string("Hide " + switch_text_).c_str());
    ShowPixmap(image2_, false);
    switch_state_ = true;
  }
}

DatabaseImageViewerWidget::DatabaseImageViewerWidget(
    QWidget* parent, OpenGLWindow* opengl_window, OptionManager* options)
    : FeatureImageViewerWidget(parent, "keypoints"),
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

  table_widget_->setItem(row, 0, new QTableWidgetItem("tx, ty, tz"));
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

  grid_layout_->addWidget(table_widget_, 0, 0);

  delete_button_ = new QPushButton(tr("Delete"), this);
  delete_button_->setFont(font);
  button_layout_->addWidget(delete_button_);
  connect(delete_button_, &QPushButton::released, this,
          &DatabaseImageViewerWidget::DeleteImage);
}

void DatabaseImageViewerWidget::ShowImageWithId(const image_t image_id) {
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

  const std::string path = JoinPaths(*options_->image_path, image.Name());
  ReadAndShowWithKeypoints(path, keypoints, tri_mask);
}

void DatabaseImageViewerWidget::ResizeTable() {
  // Set fixed table dimensions.
  table_widget_->resizeColumnsToContents();
  int height = table_widget_->horizontalHeader()->height() +
               2 * table_widget_->frameWidth();
  for (int i = 0; i < table_widget_->rowCount(); i++) {
    height += table_widget_->rowHeight(i);
  }
  table_widget_->setFixedHeight(height);
}

void DatabaseImageViewerWidget::DeleteImage() {
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
