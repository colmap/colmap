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

#ifndef COLMAP_SRC_UI_DATABASE_MANAGEMENT_WIDGET_H_
#define COLMAP_SRC_UI_DATABASE_MANAGEMENT_WIDGET_H_

#include <unordered_map>

#include <boost/filesystem.hpp>

#include <QtCore>
#include <QtWidgets>

#include "base/database.h"
#include "ui/image_viewer_widget.h"
#include "util/misc.h"
#include "util/option_manager.h"

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// Matches
////////////////////////////////////////////////////////////////////////////////

class MatchesTab : public QWidget {
 public:
  MatchesTab() {}
  MatchesTab(QWidget* parent, OptionManager* options, Database* database);

  void Clear();

 protected:
  void InitializeTable(const QStringList& table_header);
  void ShowMatches();
  void FillTable();

  OptionManager* options_;
  Database* database_;

  const Image* image_;
  std::vector<std::pair<const Image*, FeatureMatches>> matches_;
  std::vector<int> configs_;
  std::vector<size_t> sorted_matches_idxs_;

  QTableWidget* table_widget_;
  QLabel* info_label_;
  MatchesImageViewerWidget* matches_viewer_;
};

class RawMatchesTab : public MatchesTab {
 public:
  RawMatchesTab(QWidget* parent, OptionManager* options, Database* database);

  void Update(const std::vector<Image>& images, const image_t image_id);
};

class InlierMatchesTab : public MatchesTab {
 public:
  InlierMatchesTab(QWidget* parent, OptionManager* options, Database* database);

  void Update(const std::vector<Image>& images, const image_t image_id);
};

class MatchesWidget : public QWidget {
 public:
  MatchesWidget(QWidget* parent, OptionManager* options, Database* database);

  void ShowMatches(const std::vector<Image>& images, const image_t image_id);

 private:
  void closeEvent(QCloseEvent* event);

  QWidget* parent_;

  OptionManager* options_;

  QTabWidget* tab_widget_;
  RawMatchesTab* raw_matches_tab_;
  InlierMatchesTab* inlier_matches_tab_;
};

////////////////////////////////////////////////////////////////////////////////
// Images, Cameras
////////////////////////////////////////////////////////////////////////////////

class ImageTab : public QWidget {
 public:
  ImageTab(QWidget* parent, OptionManager* options, Database* database);

  void Update();
  void Save();
  void Clear();

 private:
  void itemChanged(QTableWidgetItem* item);

  void ShowImage();
  void ShowMatches();
  void SetCamera();

  OptionManager* options_;
  Database* database_;

  std::vector<Image> images_;

  QTableWidget* table_widget_;
  QLabel* info_label_;

  MatchesWidget* matches_widget_;

  BasicImageViewerWidget* image_viewer_;
};

class CameraTab : public QWidget {
 public:
  CameraTab(QWidget* parent, Database* database);

  void Update();
  void Save();
  void Clear();

 private:
  void itemChanged(QTableWidgetItem* item);
  void Add();

  Database* database_;

  std::vector<Camera> cameras_;

  QTableWidget* table_widget_;
  QLabel* info_label_;
};

class DatabaseManagementWidget : public QWidget {
 public:
  DatabaseManagementWidget(QWidget* parent, OptionManager* options);

 private:
  void showEvent(QShowEvent* event);
  void hideEvent(QHideEvent* event);

  void Save();

  QWidget* parent_;

  OptionManager* options_;
  Database database_;

  QTabWidget* tab_widget_;
  ImageTab* image_tab_;
  CameraTab* camera_tab_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_DATABASE_MANAGEMENT_WIDGET_H_
