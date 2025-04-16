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

#include "colmap/ui/movie_grabber_widget.h"

#include "colmap/geometry/pose.h"
#include "colmap/scene/projection.h"
#include "colmap/ui/model_viewer_widget.h"

namespace colmap {

MovieGrabberWidget::MovieGrabberWidget(QWidget* parent,
                                       ModelViewerWidget* model_viewer_widget)
    : QWidget(parent), model_viewer_widget_(model_viewer_widget) {
  setWindowFlags(Qt::Widget | Qt::WindowStaysOnTopHint | Qt::Tool);
  setWindowTitle("Grab movie");

  QGridLayout* grid = new QGridLayout(this);
  grid->setContentsMargins(0, 5, 0, 5);

  add_button_ = new QPushButton(tr("Add"), this);
  connect(add_button_, &QPushButton::released, this, &MovieGrabberWidget::Add);
  grid->addWidget(add_button_, 0, 0);

  delete_button_ = new QPushButton(tr("Delete"), this);
  connect(delete_button_,
          &QPushButton::released,
          this,
          &MovieGrabberWidget::Delete);
  grid->addWidget(delete_button_, 0, 1);

  clear_button_ = new QPushButton(tr("Clear"), this);
  connect(
      clear_button_, &QPushButton::released, this, &MovieGrabberWidget::Clear);
  grid->addWidget(clear_button_, 0, 2);

  table_ = new QTableWidget(this);
  table_->setColumnCount(1);
  QStringList table_header;
  table_header << "Time [seconds]";
  table_->setHorizontalHeaderLabels(table_header);
  table_->resizeColumnsToContents();
  table_->setShowGrid(true);
  table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table_->verticalHeader()->setVisible(true);
  table_->verticalHeader()->setDefaultSectionSize(18);
  table_->setSelectionMode(QAbstractItemView::SingleSelection);
  table_->setSelectionBehavior(QAbstractItemView::SelectRows);
  connect(table_,
          &QTableWidget::itemChanged,
          this,
          &MovieGrabberWidget::TimeChanged);
  connect(table_->selectionModel(),
          &QItemSelectionModel::selectionChanged,
          this,
          &MovieGrabberWidget::SelectionChanged);
  grid->addWidget(table_, 1, 0, 1, 3);

  grid->addWidget(new QLabel(tr("Frame rate"), this), 2, 1);
  frame_rate_sb_ = new QSpinBox(this);
  frame_rate_sb_->setMinimum(1);
  frame_rate_sb_->setMaximum(1000);
  frame_rate_sb_->setSingleStep(1);
  frame_rate_sb_->setValue(100);
  grid->addWidget(frame_rate_sb_, 2, 2);

  grid->addWidget(new QLabel(tr("Smooth transition"), this), 3, 1);
  smooth_cb_ = new QCheckBox(this);
  smooth_cb_->setChecked(true);
  grid->addWidget(smooth_cb_, 3, 2);

  grid->addWidget(new QLabel(tr("Smoothness"), this), 4, 1);
  smoothness_sb_ = new QDoubleSpinBox(this);
  smoothness_sb_->setMinimum(0);
  smoothness_sb_->setMaximum(1);
  smoothness_sb_->setSingleStep(0.01);
  smoothness_sb_->setValue(0.5);
  grid->addWidget(smoothness_sb_, 4, 2);

  assemble_button_ = new QPushButton(tr("Assemble movie"), this);
  connect(assemble_button_,
          &QPushButton::released,
          this,
          &MovieGrabberWidget::Assemble);
  grid->addWidget(assemble_button_, 5, 1, 1, 2);
}

void MovieGrabberWidget::Add() {
  const QMatrix4x4 matrix = model_viewer_widget_->ModelViewMatrix();

  double time = 0;
  if (table_->rowCount() > 0) {
    time = table_->item(table_->rowCount() - 1, 0)->text().toDouble() + 1;
  }

  QTableWidgetItem* item = new QTableWidgetItem();
  item->setData(Qt::DisplayRole, time);
  item->setFlags(Qt::NoItemFlags | Qt::ItemIsEnabled | Qt::ItemIsSelectable |
                 Qt::ItemIsEditable);
  item->setTextAlignment(Qt::AlignRight);

  // Save size state of current viewpoint.
  ViewData view_data;
  view_data.model_view_matrix = matrix;
  view_data.point_size = model_viewer_widget_->PointSize();
  view_data.image_size = model_viewer_widget_->ImageSize();
  view_data_.emplace(item, view_data);

  table_->insertRow(table_->rowCount());
  table_->setItem(table_->rowCount() - 1, 0, item);
  table_->selectRow(table_->rowCount() - 1);

  // Zoom out a little, so that we can see the newly added camera
  model_viewer_widget_->ChangeFocusDistance(-5);
}

void MovieGrabberWidget::Delete() {
  QModelIndexList selection = table_->selectionModel()->selectedIndexes();
  for (const auto& index : selection) {
    table_->removeRow(index.row());
  }
  UpdateViews();
  model_viewer_widget_->UpdateMovieGrabber();
}

void MovieGrabberWidget::Clear() {
  view_data_.clear();
  while (table_->rowCount() > 0) {
    table_->removeRow(0);
  }
  frames.clear();
  model_viewer_widget_->UpdateMovieGrabber();
}

void MovieGrabberWidget::Assemble() {
  if (table_->rowCount() < 2) {
    QMessageBox::critical(
        this, tr("Error"), tr("You must add at least two control frames."));
    return;
  }

  if (model_viewer_widget_->GetProjectionType() !=
      RenderOptions::ProjectionType::PERSPECTIVE) {
    QMessageBox::critical(
        this, tr("Error"), tr("You must use perspective projection."));
    return;
  }

  const QString path = QFileDialog::getExistingDirectory(
      this, tr("Choose destination..."), "", QFileDialog::ShowDirsOnly);

  // File dialog cancelled?
  if (path == "") {
    return;
  }

  const QDir dir = QDir(path);

  const QMatrix4x4 model_view_matrix_cached =
      model_viewer_widget_->ModelViewMatrix();
  const float point_size_cached = model_viewer_widget_->PointSize();
  const float image_size_cached = model_viewer_widget_->ImageSize();
  std::vector<Frame> cached_frames = frames;

  // Make sure we do not render movie grabber path.
  frames.clear();
  model_viewer_widget_->UpdateMovieGrabber();
  model_viewer_widget_->DisableCoordinateGrid();

  const float frame_rate = frame_rate_sb_->value();
  const float frame_time = 1.0f / frame_rate;
  size_t frame_number = 0;

  // Data of first view.
  const Eigen::Matrix4d prev_model_view_matrix =
      QMatrixToEigen(view_data_[table_->item(0, 0)].model_view_matrix)
          .cast<double>();
  Rigid3d prev_view_model = Inverse(
      Rigid3d(Eigen::Quaterniond(prev_model_view_matrix.topLeftCorner<3, 3>()),
              prev_model_view_matrix.topRightCorner<3, 1>()));

  for (int row = 1; row < table_->rowCount(); ++row) {
    const auto logical_idx = table_->verticalHeader()->logicalIndex(row);
    QTableWidgetItem* prev_table_item = table_->item(logical_idx - 1, 0);
    QTableWidgetItem* table_item = table_->item(logical_idx, 0);

    const ViewData& prev_view_data = view_data_.at(prev_table_item);
    const ViewData& view_data = view_data_.at(table_item);

    // Data of next view.
    const Eigen::Matrix4d curr_model_view_matrix =
        QMatrixToEigen(view_data.model_view_matrix).cast<double>();
    const Rigid3d curr_view_model = Inverse(Rigid3d(
        Eigen::Quaterniond(curr_model_view_matrix.topLeftCorner<3, 3>()),
        curr_model_view_matrix.topRightCorner<3, 1>()));

    // Time difference between previous and current view.
    const float dt = std::abs(table_item->text().toFloat() -
                              prev_table_item->text().toFloat());

    // Point size differences between previous and current view.
    const float dpoint_size = view_data.point_size - prev_view_data.point_size;
    const float dimage_size = view_data.image_size - prev_view_data.image_size;

    const auto num_frames = dt * frame_rate;
    for (size_t i = 0; i < num_frames; ++i) {
      const float t = i * frame_time;
      float tt = t / dt;

      if (smooth_cb_->isChecked()) {
        tt = ScaleSigmoid(tt, static_cast<float>(smoothness_sb_->value()));
      }

      const Rigid3d interp_view_model =
          InterpolateCameraPoses(prev_view_model, curr_view_model, tt);

      Eigen::Matrix4d frame_model_view_matrix = Eigen::Matrix4d::Identity();
      frame_model_view_matrix.topLeftCorner<3, 4>() =
          Inverse(interp_view_model).ToMatrix();

      model_viewer_widget_->SetModelViewMatrix(
          EigenToQMatrix(frame_model_view_matrix.cast<float>()));

      // Set point and image sizes.
      model_viewer_widget_->SetPointSize(prev_view_data.point_size +
                                         dpoint_size * tt);
      model_viewer_widget_->SetImageSize(prev_view_data.image_size +
                                         dimage_size * tt);

      QImage image = model_viewer_widget_->GrabImage();
      image.save(dir.filePath(
          "frame" + QString().asprintf("%06zu", frame_number) + ".png"));
      frame_number += 1;
    }

    prev_view_model = curr_view_model;
  }

  frames = std::move(cached_frames);
  model_viewer_widget_->SetPointSize(point_size_cached);
  model_viewer_widget_->SetImageSize(image_size_cached);
  model_viewer_widget_->UpdateMovieGrabber();
  model_viewer_widget_->EnableCoordinateGrid();
  model_viewer_widget_->SetModelViewMatrix(model_view_matrix_cached);
}

void MovieGrabberWidget::TimeChanged(QTableWidgetItem* item) {
  table_->sortItems(0, Qt::AscendingOrder);
  UpdateViews();
  model_viewer_widget_->UpdateMovieGrabber();
}

void MovieGrabberWidget::SelectionChanged(const QItemSelection& selected,
                                          const QItemSelection& deselected) {
  for (const auto& index : table_->selectionModel()->selectedIndexes()) {
    model_viewer_widget_->SelectMoviewGrabberView(index.row());
  }
}

void MovieGrabberWidget::UpdateViews() {
  frames.clear();
  for (int row = 0; row < table_->rowCount(); ++row) {
    const auto logical_idx = table_->verticalHeader()->logicalIndex(row);
    QTableWidgetItem* item = table_->item(logical_idx, 0);

    const Eigen::Matrix4d model_view_matrix =
        QMatrixToEigen(view_data_.at(item).model_view_matrix).cast<double>();
    Frame frame;
    frame.SetRigFromWorld(
        Rigid3d(Eigen::Quaterniond(model_view_matrix.topLeftCorner<3, 3>()),
                model_view_matrix.topRightCorner<3, 1>()));
    frames.push_back(frame);
  }
}

}  // namespace colmap
