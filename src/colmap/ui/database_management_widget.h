// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/scene/database.h"
#include "colmap/ui/image_viewer_widget.h"

#include <QtCore>
#include <QtWidgets>

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// Matches
////////////////////////////////////////////////////////////////////////////////

class TwoViewInfoTab : public QWidget {
 public:
  TwoViewInfoTab() {}
  TwoViewInfoTab(QWidget* parent, OptionManager* options);

  void Clear();

 protected:
  void InitializeTable(const QStringList& table_header);
  void ShowMatches();
  void FillTable();

  OptionManager* options_;
  std::shared_ptr<Database> database_;
  const Image* image_;
  std::vector<std::pair<const Image*, FeatureMatches>> matches_;
  std::vector<int> configs_;
  std::vector<size_t> sorted_matches_idxs_;

  QTableWidget* table_widget_;
  QLabel* info_label_;
  FeatureImageViewerWidget* matches_viewer_widget_;
};

class MatchesTab : public TwoViewInfoTab {
 public:
  MatchesTab(QWidget* parent, OptionManager* options);

  void Reload(const std::shared_ptr<Database>& database,
              const std::vector<Image>& images,
              image_t image_id);
};

class TwoViewGeometriesTab : public TwoViewInfoTab {
 public:
  TwoViewGeometriesTab(QWidget* parent, OptionManager* options);

  void Reload(const std::shared_ptr<Database>& database,
              const std::vector<Image>& images,
              image_t image_id);
};

class OverlappingImagesWidget : public QWidget {
 public:
  OverlappingImagesWidget(QWidget* parent, OptionManager* options);

  void ShowMatches(const std::shared_ptr<Database>& database,
                   const std::vector<Image>& images,
                   image_t image_id);

 private:
  void closeEvent(QCloseEvent* event);

  QWidget* parent_;

  QTabWidget* tab_widget_;
  MatchesTab* matches_tab_;
  TwoViewGeometriesTab* two_view_geometries_tab_;
};

////////////////////////////////////////////////////////////////////////////////
// Images, Cameras, PosePriors
////////////////////////////////////////////////////////////////////////////////

class CameraTab : public QWidget {
 public:
  explicit CameraTab(QWidget* parent);

  void Reload(const std::shared_ptr<Database>& database);
  void Clear();

 private:
  void itemChanged(QTableWidgetItem* item);
  void Add();
  void SetModel();

  std::shared_ptr<Database> database_;
  std::vector<Camera> cameras_;

  QTableWidget* table_widget_;
  QLabel* info_label_;
};

class ImageTab : public QWidget {
 public:
  ImageTab(QWidget* parent, CameraTab* camera_tab, OptionManager* options);

  void Reload(const std::shared_ptr<Database>& database);
  void Clear();

 private:
  void itemChanged(QTableWidgetItem* item);

  void ShowImage();
  void ShowMatches();
  void SetCamera();
  void SplitCamera();

  CameraTab* camera_tab_;

  OptionManager* options_;

  std::shared_ptr<Database> database_;
  std::vector<Image> images_;

  QTableWidget* table_widget_;
  QLabel* info_label_;

  OverlappingImagesWidget* overlapping_images_widget_;

  FeatureImageViewerWidget* image_viewer_widget_;
};

class PosePriorsTab : public QWidget {
 public:
  explicit PosePriorsTab(QWidget* parent);

  void Reload(const std::shared_ptr<Database>& database);
  void Clear();

 private:
  void itemChanged(QTableWidgetItem* item);

  std::shared_ptr<Database> database_;

  QTableWidget* table_widget_;
  QLabel* info_label_;
};

class DatabaseManagementWidget : public QWidget {
 public:
  DatabaseManagementWidget(QWidget* parent, OptionManager* options);

 private:
  void showEvent(QShowEvent* event);
  void hideEvent(QHideEvent* event);

  void ClearMatches();
  void ClearTwoViewGeometries();

  QWidget* parent_;

  OptionManager* options_;
  std::shared_ptr<Database> database_;

  QTabWidget* tab_widget_;
  ImageTab* image_tab_;
  CameraTab* camera_tab_;
  PosePriorsTab* pose_prior_tab_;
};

}  // namespace colmap
