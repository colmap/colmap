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

#include "colmap/ui/point_viewer_widget.h"

#include "colmap/ui/model_viewer_widget.h"
#include "colmap/util/file.h"

namespace colmap {

PointViewerWidget::PointViewerWidget(QWidget* parent,
                                     ModelViewerWidget* model_viewer_widget,
                                     OptionManager* options)
    : QWidget(parent),
      model_viewer_widget_(model_viewer_widget),
      options_(options),
      point3D_id_(kInvalidPoint3DId),
      zoom_(250.0 / 1024.0) {
  setWindowFlags(Qt::Window);
  resize(parent->size().width() - 20, parent->size().height() - 20);

  QFont font;
  font.setPointSize(10);
  setFont(font);

  QGridLayout* grid = new QGridLayout(this);
  grid->setContentsMargins(5, 5, 5, 5);

  info_table_ = new QTableWidget(this);
  info_table_->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
  info_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  info_table_->setSelectionMode(QAbstractItemView::SingleSelection);
  info_table_->setShowGrid(true);
  info_table_->horizontalHeader()->setStretchLastSection(true);
  info_table_->horizontalHeader()->setVisible(false);
  info_table_->verticalHeader()->setVisible(false);
  info_table_->verticalHeader()->setDefaultSectionSize(18);

  info_table_->setColumnCount(2);
  info_table_->setRowCount(3);

  info_table_->setItem(0, 0, new QTableWidgetItem("position"));
  xyz_item_ = new QTableWidgetItem();
  info_table_->setItem(0, 1, xyz_item_);

  info_table_->setItem(1, 0, new QTableWidgetItem("color"));
  rgb_item_ = new QTableWidgetItem();
  info_table_->setItem(1, 1, rgb_item_);

  info_table_->setItem(2, 0, new QTableWidgetItem("error"));
  error_item_ = new QTableWidgetItem();
  info_table_->setItem(2, 1, error_item_);

  grid->addWidget(info_table_, 0, 0);

  location_table_ = new QTableWidget(this);
  location_table_->setColumnCount(4);
  QStringList table_header;
  table_header << "image_id"
               << "reproj_error"
               << "track_location"
               << "image_name";
  location_table_->setHorizontalHeaderLabels(table_header);
  location_table_->resizeColumnsToContents();
  location_table_->setShowGrid(true);
  location_table_->horizontalHeader()->setStretchLastSection(true);
  location_table_->verticalHeader()->setVisible(true);
  location_table_->setSelectionMode(QAbstractItemView::NoSelection);
  location_table_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  location_table_->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
  location_table_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);

  grid->addWidget(location_table_, 1, 0);

  QHBoxLayout* button_layout = new QHBoxLayout();

  zoom_in_button_ = new QPushButton(tr("+"), this);
  zoom_in_button_->setFont(font);
  zoom_in_button_->setFixedWidth(50);
  button_layout->addWidget(zoom_in_button_);
  connect(zoom_in_button_,
          &QPushButton::released,
          this,
          &PointViewerWidget::ZoomIn);

  zoom_out_button_ = new QPushButton(tr("-"), this);
  zoom_out_button_->setFont(font);
  zoom_out_button_->setFixedWidth(50);
  button_layout->addWidget(zoom_out_button_);
  connect(zoom_out_button_,
          &QPushButton::released,
          this,
          &PointViewerWidget::ZoomOut);

  delete_button_ = new QPushButton(tr("Delete"), this);
  button_layout->addWidget(delete_button_);
  connect(
      delete_button_, &QPushButton::released, this, &PointViewerWidget::Delete);

  grid->addLayout(button_layout, 2, 0, Qt::AlignRight);
}

void PointViewerWidget::Show(const point3D_t point3D_id) {
  location_pixmaps_.clear();
  image_ids_.clear();
  reproj_errors_.clear();
  image_names_.clear();

  if (model_viewer_widget_->points3D.count(point3D_id) == 0) {
    point3D_id_ = kInvalidPoint3DId;
    ClearLocations();
    return;
  }

  show();
  raise();

  point3D_id_ = point3D_id;

  // Show some general information about the point.

  setWindowTitle(QString::fromStdString("Point " + std::to_string(point3D_id)));

  const auto& point3D = model_viewer_widget_->points3D[point3D_id];

  xyz_item_->setText(QString::number(point3D.xyz(0)) + ", " +
                     QString::number(point3D.xyz(1)) + ", " +
                     QString::number(point3D.xyz(2)));
  rgb_item_->setText(QString::number(point3D.color(0)) + ", " +
                     QString::number(point3D.color(1)) + ", " +
                     QString::number(point3D.color(2)));
  error_item_->setText(QString::number(point3D.error));

  ResizeInfoTable();

  // Sort the track elements by the image names.

  std::vector<std::pair<TrackElement, std::string>> track_idx_image_name_pairs;
  track_idx_image_name_pairs.reserve(point3D.track.Length());
  for (const auto& track_el : point3D.track.Elements()) {
    const Image& image = model_viewer_widget_->images[track_el.image_id];
    track_idx_image_name_pairs.emplace_back(track_el, image.Name());
  }

  std::sort(track_idx_image_name_pairs.begin(),
            track_idx_image_name_pairs.end(),
            [](const std::pair<TrackElement, std::string>& track_el1,
               const std::pair<TrackElement, std::string>& track_el2) {
              return track_el1.second < track_el2.second;
            });

  // Paint features for each track element.

  for (const auto& track_el : track_idx_image_name_pairs) {
    const Image& image = model_viewer_widget_->images[track_el.first.image_id];
    const Camera& camera = model_viewer_widget_->cameras[image.CameraId()];
    const Point2D& point2D = image.Point2D(track_el.first.point2D_idx);
    const std::optional<Eigen::Vector2d> proj_point2D =
        camera.ImgFromCam(image.CamFromWorld() * point3D.xyz);
    if (!proj_point2D) {
      LOG(WARNING) << "Failed to project point into image " << image.Name();
      continue;
    }

    const double reproj_error = (point2D.xy - *proj_point2D).norm();

    Bitmap bitmap;
    const std::string path = JoinPaths(*options_->image_path, image.Name());
    if (!bitmap.Read(path, true)) {
      LOG(ERROR) << "Cannot read image at path " << path;
      continue;
    }

    QPixmap pixmap = QPixmap::fromImage(BitmapToQImageRGB(bitmap));

    // Paint feature in current image.
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);

    QPen pen;
    pen.setWidth(3);
    pen.setColor(Qt::green);
    painter.setPen(pen);

    const int kCrossSize = 15;
    const int x = static_cast<int>(std::round(point2D.xy(0)));
    const int y = static_cast<int>(std::round(point2D.xy(1)));
    painter.drawLine(
        x - kCrossSize, y - kCrossSize, x + kCrossSize, y + kCrossSize);
    painter.drawLine(
        x - kCrossSize, y + kCrossSize, x + kCrossSize, y - kCrossSize);

    pen.setColor(Qt::red);
    painter.setPen(pen);

    const int proj_x = static_cast<int>(std::round(proj_point2D->x()));
    const int proj_y = static_cast<int>(std::round(proj_point2D->y()));
    painter.drawEllipse(proj_x - 5, proj_y - 5, 10, 10);
    painter.drawEllipse(proj_x - 15, proj_y - 15, 30, 30);
    painter.drawEllipse(proj_x - 45, proj_y - 45, 90, 90);

    location_pixmaps_.push_back(pixmap);
    image_ids_.push_back(track_el.first.image_id);
    reproj_errors_.push_back(reproj_error);
    image_names_.push_back(image.Name());
  }

  UpdateImages();
}

void PointViewerWidget::closeEvent(QCloseEvent* event) {
  // Release the images, since zoomed in images can use a lot of memory.
  location_pixmaps_.clear();
  image_ids_.clear();
  reproj_errors_.clear();
  image_names_.clear();
  ClearLocations();
}

void PointViewerWidget::ResizeInfoTable() {
  // Set fixed table dimensions.
  info_table_->resizeColumnsToContents();
  int height =
      info_table_->horizontalHeader()->height() + 2 * info_table_->frameWidth();
  for (int i = 0; i < info_table_->rowCount(); i++) {
    height += info_table_->rowHeight(i);
  }
  info_table_->setFixedHeight(height);
}

void PointViewerWidget::ClearLocations() {
  while (location_table_->rowCount() > 0) {
    location_table_->removeRow(0);
  }
  for (auto location_label : location_labels_) {
    delete location_label;
  }
  location_labels_.clear();
}

void PointViewerWidget::UpdateImages() {
  ClearLocations();

  location_table_->setRowCount(static_cast<int>(location_pixmaps_.size()));

  for (size_t i = 0; i < location_pixmaps_.size(); ++i) {
    QLabel* image_id_label = new QLabel(QString::number(image_ids_[i]), this);
    image_id_label->setAlignment(Qt::AlignCenter);
    location_table_->setCellWidget(i, 0, image_id_label);
    location_labels_.push_back(image_id_label);

    QLabel* error_label = new QLabel(QString::number(reproj_errors_[i]), this);
    error_label->setAlignment(Qt::AlignCenter);
    location_table_->setCellWidget(i, 1, error_label);
    location_labels_.push_back(error_label);

    const QPixmap& pixmap = location_pixmaps_[i];
    QLabel* image_label = new QLabel(this);
    image_label->setPixmap(
        pixmap.scaledToWidth(zoom_ * pixmap.width(), Qt::FastTransformation));
    location_table_->setCellWidget(i, 2, image_label);
    location_table_->resizeRowToContents(i);
    location_labels_.push_back(image_label);

    QLabel* image_name_label = new QLabel(image_names_[i].c_str(), this);
    image_name_label->setAlignment(Qt::AlignCenter);
    location_table_->setCellWidget(i, 3, image_name_label);
    location_labels_.push_back(image_name_label);
  }
  location_table_->resizeColumnToContents(2);
}

void PointViewerWidget::ZoomIn() {
  zoom_ *= 1.33;
  UpdateImages();
}

void PointViewerWidget::ZoomOut() {
  zoom_ /= 1.3;
  UpdateImages();
}

void PointViewerWidget::Delete() {
  QMessageBox::StandardButton reply =
      QMessageBox::question(this,
                            "",
                            tr("Do you really want to delete this point?"),
                            QMessageBox::Yes | QMessageBox::No);
  if (reply == QMessageBox::Yes) {
    if (model_viewer_widget_->reconstruction->ExistsPoint3D(point3D_id_)) {
      model_viewer_widget_->reconstruction->DeletePoint3D(point3D_id_);
    }
    model_viewer_widget_->ReloadReconstruction();
  }
  hide();
}

}  // namespace colmap
