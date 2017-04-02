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

#include "ui/point_viewer_widget.h"

#include "ui/opengl_window.h"
#include "util/misc.h"

namespace colmap {

PointViewerWidget::PointViewerWidget(QWidget* parent,
                                     OpenGLWindow* opengl_window,
                                     OptionManager* options)
    : QWidget(parent),
      opengl_window_(opengl_window),
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

  location_table_ = new QTableWidget(this);
  location_table_->setColumnCount(3);
  QStringList table_header;
  table_header << "image_id"
               << "reproj_error"
               << "track_location";
  location_table_->setHorizontalHeaderLabels(table_header);
  location_table_->resizeColumnsToContents();
  location_table_->setShowGrid(true);
  location_table_->horizontalHeader()->setStretchLastSection(true);
  location_table_->verticalHeader()->setVisible(true);
  location_table_->setSelectionMode(QAbstractItemView::NoSelection);
  location_table_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
  location_table_->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
  location_table_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);

  grid->addWidget(location_table_, 0, 0);

  QHBoxLayout* button_layout = new QHBoxLayout();

  zoom_in_button_ = new QPushButton(tr("+"), this);
  zoom_in_button_->setFont(font);
  zoom_in_button_->setFixedWidth(50);
  button_layout->addWidget(zoom_in_button_);
  connect(zoom_in_button_, &QPushButton::released, this,
          &PointViewerWidget::ZoomIn);

  zoom_out_button_ = new QPushButton(tr("-"), this);
  zoom_out_button_->setFont(font);
  zoom_out_button_->setFixedWidth(50);
  button_layout->addWidget(zoom_out_button_);
  connect(zoom_out_button_, &QPushButton::released, this,
          &PointViewerWidget::ZoomOut);

  delete_button_ = new QPushButton(tr("Delete"), this);
  button_layout->addWidget(delete_button_);
  connect(delete_button_, &QPushButton::released, this,
          &PointViewerWidget::Delete);

  grid->addLayout(button_layout, 1, 0, Qt::AlignRight);
}

void PointViewerWidget::Show(const point3D_t point3D_id) {
  location_pixmaps_.clear();
  image_ids_.clear();
  reproj_errors_.clear();

  if (opengl_window_->points3D.count(point3D_id) == 0) {
    point3D_id_ = kInvalidPoint3DId;
    ClearLocations();
    return;
  }

  show();
  raise();

  point3D_id_ = point3D_id;

  setWindowTitle(QString::fromStdString("Point " + std::to_string(point3D_id)));

  const auto& point3D = opengl_window_->points3D[point3D_id];

  // Paint features for each track element.
  for (const auto& track_el : point3D.Track().Elements()) {
    const Image& image = opengl_window_->images[track_el.image_id];
    const Camera& camera = opengl_window_->cameras[image.CameraId()];
    const Point2D& point2D = image.Point2D(track_el.point2D_idx);

    const Eigen::Matrix3x4d proj_matrix = image.ProjectionMatrix();
    const double error = CalculateReprojectionError(point2D.XY(), point3D.XYZ(),
                                                    proj_matrix, camera);

    const std::string path = JoinPaths(*options_->image_path, image.Name());

    Bitmap bitmap;
    if (!bitmap.Read(path, true)) {
      std::cerr << "ERROR: Cannot read image at path " << path << std::endl;
      continue;
    }

    QPixmap pixmap = QPixmap::fromImage(BitmapToQImageRGB(bitmap));

    // Paint feature in current image.
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    QPen pen;
    pen.setWidth(3);
    pen.setColor(Qt::red);
    painter.setPen(pen);
    painter.drawEllipse(static_cast<int>(point2D.X() - 5),
                        static_cast<int>(point2D.Y() - 5), 10, 10);
    painter.drawEllipse(static_cast<int>(point2D.X() - 15),
                        static_cast<int>(point2D.Y() - 15), 30, 30);
    painter.drawEllipse(static_cast<int>(point2D.X() - 45),
                        static_cast<int>(point2D.Y() - 45), 90, 90);

    location_pixmaps_.push_back(pixmap);
    image_ids_.push_back(track_el.image_id);
    reproj_errors_.push_back(error);
  }

  UpdateImages();
}

void PointViewerWidget::closeEvent(QCloseEvent* event) {
  // Release the images, since zoomed in images can use a lot of memory.
  location_pixmaps_.clear();
  image_ids_.clear();
  reproj_errors_.clear();
  ClearLocations();
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
    location_table_->setCellWidget(i, 0, image_id_label);
    location_labels_.push_back(image_id_label);

    QLabel* error_label = new QLabel(QString::number(reproj_errors_[i]), this);
    location_table_->setCellWidget(i, 1, error_label);
    location_labels_.push_back(error_label);

    const QPixmap& pixmap = location_pixmaps_[i];
    QLabel* image_label = new QLabel(this);
    image_label->setPixmap(
        pixmap.scaledToWidth(zoom_ * pixmap.width(), Qt::FastTransformation));
    location_table_->setCellWidget(i, 2, image_label);
    location_table_->resizeRowToContents(i);
    location_labels_.push_back(image_label);
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
  QMessageBox::StandardButton reply = QMessageBox::question(
      this, "", tr("Do you really want to delete this point?"),
      QMessageBox::Yes | QMessageBox::No);
  if (reply == QMessageBox::Yes) {
    if (opengl_window_->reconstruction->ExistsPoint3D(point3D_id_)) {
      opengl_window_->reconstruction->DeletePoint3D(point3D_id_);
    }
    opengl_window_->Update();
  }
}

}  // namespace colmap
