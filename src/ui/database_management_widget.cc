// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#include "ui/database_management_widget.h"

#include <numeric>

#include "base/camera_models.h"
#include "ui/qt_utils.h"

namespace colmap {

MatchesTab::MatchesTab(QWidget* parent, OptionManager* options,
                       Database* database)
    : QWidget(parent),
      options_(options),
      database_(database),
      matches_viewer_(new MatchesImageViewerWidget(parent)) {}

void MatchesTab::Clear() {
  table_widget_->clearContents();
  matches_.clear();
  configs_.clear();
  sorted_matches_idxs_.clear();
}

void MatchesTab::InitializeTable(const QStringList& table_header) {
  QGridLayout* grid = new QGridLayout(this);

  info_label_ = new QLabel(this);
  grid->addWidget(info_label_, 0, 0);

  QPushButton* show_button = new QPushButton(tr("Show matches"), this);
  connect(show_button, &QPushButton::released, this, &MatchesTab::ShowMatches);
  grid->addWidget(show_button, 0, 1, Qt::AlignRight);

  table_widget_ = new QTableWidget(this);
  table_widget_->setColumnCount(table_header.size());
  table_widget_->setHorizontalHeaderLabels(table_header);

  table_widget_->setShowGrid(true);
  table_widget_->setSelectionBehavior(QAbstractItemView::SelectRows);
  table_widget_->setSelectionMode(QAbstractItemView::SingleSelection);
  table_widget_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table_widget_->horizontalHeader()->setStretchLastSection(true);
  table_widget_->verticalHeader()->setVisible(false);
  table_widget_->verticalHeader()->setDefaultSectionSize(20);

  grid->addWidget(table_widget_, 1, 0, 1, 2);
}

void MatchesTab::ShowMatches() {
  QItemSelectionModel* select = table_widget_->selectionModel();

  if (!select->hasSelection()) {
    QMessageBox::critical(this, "", tr("No image pair selected."));
    return;
  }

  if (select->selectedRows().size() > 1) {
    QMessageBox::critical(this, "", tr("Only one image pair may be selected."));
    return;
  }

  const size_t idx =
      sorted_matches_idxs_[select->selectedRows().begin()->row()];
  const auto& selection = matches_[idx];
  const std::string path1 =
      EnsureTrailingSlash(*options_->image_path) + image_->Name();
  const std::string path2 =
      EnsureTrailingSlash(*options_->image_path) + selection.first->Name();
  const auto keypoints1 = database_->ReadKeypoints(image_->ImageId());
  const auto keypoints2 = database_->ReadKeypoints(selection.first->ImageId());

  matches_viewer_->Show(path1, path2, keypoints1, keypoints2, selection.second);

  matches_viewer_->setWindowTitle(QString::fromStdString(
      "Matches for image pair " + std::to_string(image_->ImageId()) + " - " +
      std::to_string(selection.first->ImageId())));

  matches_viewer_->show();
  matches_viewer_->raise();
}

void MatchesTab::FillTable() {
  // Sort the matched pairs according to number of matches in descending order
  sorted_matches_idxs_.resize(matches_.size());
  std::iota(sorted_matches_idxs_.begin(), sorted_matches_idxs_.end(), 0);

  std::sort(sorted_matches_idxs_.begin(), sorted_matches_idxs_.end(),
            [&](const size_t idx1, const size_t idx2) {
              return matches_[idx1].second.size() >
                     matches_[idx2].second.size();
            });

  QString info;
  info += QString("Matched images: ") + QString::number(matches_.size());
  info_label_->setText(info);

  table_widget_->clearContents();
  table_widget_->setRowCount(matches_.size());

  for (size_t i = 0; i < sorted_matches_idxs_.size(); ++i) {
    const size_t idx = sorted_matches_idxs_[i];

    QTableWidgetItem* image_id_item =
        new QTableWidgetItem(QString::number(matches_[idx].first->ImageId()));
    table_widget_->setItem(i, 0, image_id_item);

    QTableWidgetItem* num_matches_item =
        new QTableWidgetItem(QString::number(matches_[idx].second.size()));
    table_widget_->setItem(i, 1, num_matches_item);

    // config for inlier matches tab
    if (table_widget_->columnCount() == 3) {
      QTableWidgetItem* config_item =
          new QTableWidgetItem(QString::number(configs_[idx]));
      table_widget_->setItem(i, 2, config_item);
    }
  }

  table_widget_->resizeColumnsToContents();
}

RawMatchesTab::RawMatchesTab(QWidget* parent, OptionManager* options,
                             Database* database)
    : MatchesTab(parent, options, database) {
  QStringList table_header;
  table_header << "image_id"
               << "num_matches";
  InitializeTable(table_header);
}

void RawMatchesTab::Update(const std::vector<Image>& images,
                           const image_t image_id) {
  matches_.clear();

  // Find all matched images

  for (const auto& image : images) {
    if (image.ImageId() == image_id) {
      image_ = &image;
      continue;
    }

    if (database_->ExistsMatches(image_id, image.ImageId())) {
      const auto matches = database_->ReadMatches(image_id, image.ImageId());

      if (matches.size() > 0) {
        matches_.emplace_back(&image, matches);
      }
    }
  }

  FillTable();
}

InlierMatchesTab::InlierMatchesTab(QWidget* parent, OptionManager* options,
                                   Database* database)
    : MatchesTab(parent, options, database) {
  QStringList table_header;
  table_header << "image_id"
               << "num_matches"
               << "config";
  InitializeTable(table_header);
}

void InlierMatchesTab::Update(const std::vector<Image>& images,
                              const image_t image_id) {
  matches_.clear();
  configs_.clear();

  // Find all matched images

  for (const auto& image : images) {
    if (image.ImageId() == image_id) {
      image_ = &image;
      continue;
    }

    if (database_->ExistsInlierMatches(image_id, image.ImageId())) {
      const auto two_view_geometry =
          database_->ReadInlierMatches(image_id, image.ImageId());

      if (two_view_geometry.inlier_matches.size() > 0) {
        matches_.emplace_back(&image, two_view_geometry.inlier_matches);
        configs_.push_back(two_view_geometry.config);
      }
    }
  }

  FillTable();
}

MatchesWidget::MatchesWidget(QWidget* parent, OptionManager* options,
                             Database* database)
    : parent_(parent), options_(options) {
  // Do not change flag, to make sure feature database is not accessed from
  // multiple threads
  setWindowFlags(Qt::Window);
  resize(parent->size().width() - 20, parent->size().height() - 20);

  QGridLayout* grid = new QGridLayout(this);

  tab_widget_ = new QTabWidget(this);

  raw_matches_tab_ = new RawMatchesTab(this, options_, database);
  tab_widget_->addTab(raw_matches_tab_, tr("Raw matches"));

  inlier_matches_tab_ = new InlierMatchesTab(this, options_, database);
  tab_widget_->addTab(inlier_matches_tab_, tr("Inlier matches"));

  grid->addWidget(tab_widget_, 0, 0);

  QPushButton* close_button = new QPushButton(tr("Close"), this);
  connect(close_button, &QPushButton::released, this, &MatchesWidget::close);
  grid->addWidget(close_button, 1, 0, Qt::AlignRight);
}

void MatchesWidget::ShowMatches(const std::vector<Image>& images,
                                const image_t image_id) {
  parent_->setDisabled(true);

  setWindowTitle(
      QString::fromStdString("Matches for image " + std::to_string(image_id)));

  raw_matches_tab_->Update(images, image_id);
  inlier_matches_tab_->Update(images, image_id);
}

void MatchesWidget::closeEvent(QCloseEvent* event) {
  raw_matches_tab_->Clear();
  inlier_matches_tab_->Clear();
  parent_->setEnabled(true);
}

ImageTab::ImageTab(QWidget* parent, OptionManager* options, Database* database)
    : QWidget(parent), options_(options), database_(database) {
  QGridLayout* grid = new QGridLayout(this);

  info_label_ = new QLabel(this);
  grid->addWidget(info_label_, 0, 0);

  QPushButton* set_camera_button = new QPushButton(tr("Set camera"), this);
  connect(set_camera_button, &QPushButton::released, this,
          &ImageTab::SetCamera);
  grid->addWidget(set_camera_button, 0, 1, Qt::AlignRight);

  QPushButton* show_image_button = new QPushButton(tr("Show image"), this);
  connect(show_image_button, &QPushButton::released, this,
          &ImageTab::ShowImage);
  grid->addWidget(show_image_button, 0, 2, Qt::AlignRight);

  QPushButton* show_matches_button = new QPushButton(tr("Show matches"), this);
  connect(show_matches_button, &QPushButton::released, this,
          &ImageTab::ShowMatches);
  grid->addWidget(show_matches_button, 0, 3, Qt::AlignRight);

  table_widget_ = new QTableWidget(this);
  table_widget_->setColumnCount(10);

  QStringList table_header;
  table_header << "image_id"
               << "name"
               << "camera_id"
               << "qw"
               << "qx"
               << "qy"
               << "qz"
               << "tx"
               << "ty"
               << "tz";
  table_widget_->setHorizontalHeaderLabels(table_header);

  table_widget_->setShowGrid(true);
  table_widget_->setSelectionBehavior(QAbstractItemView::SelectRows);
  table_widget_->horizontalHeader()->setStretchLastSection(true);
  table_widget_->verticalHeader()->setVisible(false);
  table_widget_->verticalHeader()->setDefaultSectionSize(20);

  connect(table_widget_, &QTableWidget::itemChanged, this,
          &ImageTab::itemChanged);

  grid->addWidget(table_widget_, 1, 0, 1, 4);

  grid->setColumnStretch(0, 2);

  image_viewer_ = new BasicImageViewerWidget(parent, "keypoints");
  matches_widget_ = new MatchesWidget(parent, options, database_);
}

void ImageTab::Update() {
  QString info;
  info += QString("Images: ") + QString::number(database_->NumImages());
  info += QString("\n");
  info += QString("Features: ") + QString::number(database_->NumKeypoints());
  info_label_->setText(info);

  images_ = database_->ReadAllImages();

  // Make sure, itemChanged is not invoked, while setting up the table
  table_widget_->blockSignals(true);

  table_widget_->clearContents();
  table_widget_->setRowCount(images_.size());

  for (size_t i = 0; i < images_.size(); ++i) {
    const auto& image = images_[i];
    QTableWidgetItem* id_item =
        new QTableWidgetItem(QString::number(image.ImageId()));
    id_item->setFlags(Qt::ItemIsSelectable);
    table_widget_->setItem(i, 0, id_item);
    table_widget_->setItem(
        i, 1, new QTableWidgetItem(QString::fromStdString(image.Name())));
    table_widget_->setItem(
        i, 2, new QTableWidgetItem(QString::number(image.CameraId())));
    table_widget_->setItem(
        i, 3, new QTableWidgetItem(QString::number(image.QvecPrior(0))));
    table_widget_->setItem(
        i, 4, new QTableWidgetItem(QString::number(image.QvecPrior(1))));
    table_widget_->setItem(
        i, 5, new QTableWidgetItem(QString::number(image.QvecPrior(2))));
    table_widget_->setItem(
        i, 6, new QTableWidgetItem(QString::number(image.QvecPrior(2))));
    table_widget_->setItem(
        i, 7, new QTableWidgetItem(QString::number(image.TvecPrior(0))));
    table_widget_->setItem(
        i, 8, new QTableWidgetItem(QString::number(image.TvecPrior(1))));
    table_widget_->setItem(
        i, 9, new QTableWidgetItem(QString::number(image.TvecPrior(2))));
  }
  table_widget_->resizeColumnsToContents();

  table_widget_->blockSignals(false);
}

void ImageTab::Save() {
  database_->BeginTransaction();
  for (const auto& image : images_) {
    database_->UpdateImage(image);
  }
  database_->EndTransaction();
}

void ImageTab::Clear() {
  images_.clear();
  table_widget_->clearContents();
}

void ImageTab::itemChanged(QTableWidgetItem* item) {
  camera_t camera_id = kInvalidCameraId;

  switch (item->column()) {
    // case 0: never change the image ID
    case 1:
      images_[item->row()].SetName(item->text().toUtf8().constData());
      break;
    case 2:
      camera_id = static_cast<camera_t>(item->data(Qt::DisplayRole).toInt());
      if (!database_->ExistsCamera(camera_id)) {
        QMessageBox::critical(this, "", tr("camera_id does not exist."));
        table_widget_->blockSignals(true);
        item->setText(QString::number(images_[item->row()].CameraId()));
        table_widget_->blockSignals(false);
      } else {
        images_[item->row()].SetCameraId(camera_id);
      }
      break;
    case 3:
      images_[item->row()].QvecPrior(0) = item->data(Qt::DisplayRole).toReal();
      break;
    case 4:
      images_[item->row()].QvecPrior(1) = item->data(Qt::DisplayRole).toReal();
      break;
    case 5:
      images_[item->row()].QvecPrior(2) = item->data(Qt::DisplayRole).toReal();
      break;
    case 6:
      images_[item->row()].QvecPrior(3) = item->data(Qt::DisplayRole).toReal();
      break;
    case 7:
      images_[item->row()].TvecPrior(0) = item->data(Qt::DisplayRole).toReal();
      break;
    case 8:
      images_[item->row()].TvecPrior(1) = item->data(Qt::DisplayRole).toReal();
      break;
    case 9:
      images_[item->row()].TvecPrior(2) = item->data(Qt::DisplayRole).toReal();
      break;
    default:
      break;
  }
}

void ImageTab::ShowImage() {
  QItemSelectionModel* select = table_widget_->selectionModel();

  if (!select->hasSelection()) {
    QMessageBox::critical(this, "", tr("No image selected."));
    return;
  }

  if (select->selectedRows().size() > 1) {
    QMessageBox::critical(this, "", tr("Only one image may be selected."));
    return;
  }

  const auto& image = images_[select->selectedRows().begin()->row()];

  const auto keypoints = database_->ReadKeypoints(image.ImageId());
  const std::vector<bool> tri_mask(keypoints.size(), false);

  image_viewer_->Show(EnsureTrailingSlash(*options_->image_path) + image.Name(),
                      keypoints, tri_mask);
  image_viewer_->setWindowTitle(
      QString::fromStdString("Image " + std::to_string(image.ImageId())));
  image_viewer_->show();
}

void ImageTab::ShowMatches() {
  QItemSelectionModel* select = table_widget_->selectionModel();

  if (!select->hasSelection()) {
    QMessageBox::critical(this, "", tr("No image selected."));
    return;
  }

  if (select->selectedRows().size() > 1) {
    QMessageBox::critical(this, "", tr("Only one image may be selected."));
    return;
  }

  const auto& image = images_[select->selectedRows().begin()->row()];

  matches_widget_->ShowMatches(images_, image.ImageId());
  matches_widget_->show();
  matches_widget_->raise();
}

void ImageTab::SetCamera() {
  QItemSelectionModel* select = table_widget_->selectionModel();

  if (!select->hasSelection()) {
    QMessageBox::critical(this, "", tr("No image selected."));
    return;
  }

  bool ok;
  const camera_t camera_id = static_cast<camera_t>(
      QInputDialog::getInt(this, "", tr("camera_id"), 0, 0, INT_MAX, 1, &ok));
  if (!ok) {
    return;
  }

  if (!database_->ExistsCamera(camera_id)) {
    QMessageBox::critical(this, "", tr("camera_id does not exist."));
    return;
  }

  // Make sure, itemChanged is not invoked, while updating up the table
  table_widget_->blockSignals(true);

  for (QModelIndex& index : select->selectedRows()) {
    table_widget_->setItem(index.row(), 2,
                           new QTableWidgetItem(QString::number(camera_id)));
    images_[index.row()].SetCameraId(camera_id);
  }

  table_widget_->blockSignals(false);
}

CameraTab::CameraTab(QWidget* parent, Database* database)
    : QWidget(parent), database_(database) {
  QGridLayout* grid = new QGridLayout(this);

  info_label_ = new QLabel(this);
  grid->addWidget(info_label_, 0, 0);

  QPushButton* add_camera_button = new QPushButton(tr("Add camera"), this);
  connect(add_camera_button, &QPushButton::released, this, &CameraTab::Add);
  grid->addWidget(add_camera_button, 0, 1, Qt::AlignRight);

  table_widget_ = new QTableWidget(this);
  table_widget_->setColumnCount(6);

  QStringList table_header;
  table_header << "camera_id"
               << "model"
               << "width"
               << "height"
               << "params"
               << "prior_focal_length";
  table_widget_->setHorizontalHeaderLabels(table_header);

  table_widget_->setShowGrid(true);
  table_widget_->setSelectionBehavior(QAbstractItemView::SelectRows);
  table_widget_->horizontalHeader()->setStretchLastSection(true);
  table_widget_->verticalHeader()->setVisible(false);
  table_widget_->verticalHeader()->setDefaultSectionSize(20);

  connect(table_widget_, &QTableWidget::itemChanged, this,
          &CameraTab::itemChanged);

  grid->addWidget(table_widget_, 1, 0, 1, 2);
}

void CameraTab::Update() {
  QString info;
  info += QString("Cameras: ") + QString::number(database_->NumCameras());
  info_label_->setText(info);

  cameras_ = database_->ReadAllCameras();

  // Make sure, itemChanged is not invoked, while setting up the table.
  table_widget_->blockSignals(true);

  table_widget_->clearContents();
  table_widget_->setRowCount(cameras_.size());

  std::sort(cameras_.begin(), cameras_.end(),
            [](const Camera& camera1, const Camera& camera2) {
              return camera1.CameraId() < camera2.CameraId();
            });

  for (size_t i = 0; i < cameras_.size(); ++i) {
    const Camera& camera = cameras_[i];
    QTableWidgetItem* id_item =
        new QTableWidgetItem(QString::number(camera.CameraId()));
    id_item->setFlags(Qt::ItemIsSelectable);
    table_widget_->setItem(i, 0, id_item);

    QTableWidgetItem* model_item =
        new QTableWidgetItem(QString::fromStdString(camera.ModelName()));
    model_item->setFlags(Qt::ItemIsSelectable);
    table_widget_->setItem(i, 1, model_item);

    table_widget_->setItem(
        i, 2, new QTableWidgetItem(QString::number(camera.Width())));
    table_widget_->setItem(
        i, 3, new QTableWidgetItem(QString::number(camera.Height())));

    table_widget_->setItem(i, 4, new QTableWidgetItem(QString::fromStdString(
                                     VectorToCSV(camera.Params()))));
    table_widget_->setItem(
        i, 5,
        new QTableWidgetItem(QString::number(camera.HasPriorFocalLength())));
  }
  table_widget_->resizeColumnsToContents();

  table_widget_->blockSignals(false);
}

void CameraTab::Save() {
  database_->BeginTransaction();
  for (const Camera& camera : cameras_) {
    database_->UpdateCamera(camera);
  }
  database_->EndTransaction();
}

void CameraTab::Clear() {
  cameras_.clear();
  table_widget_->clearContents();
}

void CameraTab::itemChanged(QTableWidgetItem* item) {
  Camera& camera = cameras_.at(item->row());
  const std::vector<double> prev_params = camera.Params();

  switch (item->column()) {
    // case 0: never change the camera ID
    // case 1: never change the camera model
    case 2:
      camera.SetWidth(static_cast<size_t>(item->data(Qt::DisplayRole).toInt()));
      break;
    case 3:
      camera.SetHeight(
          static_cast<size_t>(item->data(Qt::DisplayRole).toInt()));
      break;
    case 4:
      if (!camera.SetParamsFromString(item->text().toUtf8().constData())) {
        QMessageBox::critical(this, "", tr("Invalid camera parameters."));
        table_widget_->blockSignals(true);
        item->setText(QString::fromStdString(VectorToCSV(prev_params)));
        table_widget_->blockSignals(false);
      }
      break;
    case 5:
      camera.SetPriorFocalLength(
          static_cast<bool>(item->data(Qt::DisplayRole).toInt()));
      break;
    default:
      break;
  }
}

void CameraTab::Add() {
  QStringList camera_models;
#define CAMERA_MODEL_CASE(CameraModel) \
  << QString::fromStdString(CameraModelIdToName(CameraModel::model_id))
  camera_models CAMERA_MODEL_CASES;
#undef CAMERA_MODEL_CASE

  bool ok;
  const QString camera_model = QInputDialog::getItem(
      this, "", tr("Model:"), camera_models, 0, false, &ok);
  if (!ok) {
    return;
  }

  // Add new camera to feature database
  Camera camera;
  const double kDefaultFocalLength = 1.0;
  const size_t kDefaultWidth = 1;
  const size_t kDefaultHeight = 1;
  camera.InitializeWithName(camera_model.toUtf8().constData(),
                            kDefaultFocalLength, kDefaultWidth, kDefaultHeight);
  database_->WriteCamera(camera);

  // Reload all cameras
  Update();

  // Highlight new camera
  table_widget_->selectRow(cameras_.size() - 1);
}

DatabaseManagementWidget::DatabaseManagementWidget(QWidget* parent,
                                                   OptionManager* options)
    : parent_(parent), options_(options) {
  // Do not change flag, to make sure feature database is not accessed from
  // multiple threads
  setWindowFlags(Qt::Window);
  setWindowTitle("Database management");
  resize(parent->size().width() - 20, parent->size().height() - 20);

  QGridLayout* grid = new QGridLayout(this);

  tab_widget_ = new QTabWidget(this);

  image_tab_ = new ImageTab(this, options_, &database_);
  tab_widget_->addTab(image_tab_, tr("Images"));

  camera_tab_ = new CameraTab(this, &database_);
  tab_widget_->addTab(camera_tab_, tr("Cameras"));

  grid->addWidget(tab_widget_, 0, 0, 1, 2);

  QPushButton* save_button = new QPushButton(tr("Save"), this);
  connect(save_button, &QPushButton::released, this,
          &DatabaseManagementWidget::Save);
  grid->addWidget(save_button, 1, 0, Qt::AlignRight);

  QPushButton* cancel_button = new QPushButton(tr("Cancel"), this);
  connect(cancel_button, &QPushButton::released, this,
          &DatabaseManagementWidget::close);
  grid->addWidget(cancel_button, 1, 1, Qt::AlignRight);

  grid->setColumnStretch(0, 1);
}

void DatabaseManagementWidget::showEvent(QShowEvent* event) {
  parent_->setDisabled(true);

  database_.Open(*options_->database_path);

  image_tab_->Update();
  camera_tab_->Update();
}

void DatabaseManagementWidget::hideEvent(QHideEvent* event) {
  parent_->setEnabled(true);

  image_tab_->Clear();
  camera_tab_->Clear();

  database_.Close();
}

void DatabaseManagementWidget::Save() {
  image_tab_->Save();
  camera_tab_->Save();

  QMessageBox::information(this, "", tr("Saved changes"));
}

}  // namespace colmap
