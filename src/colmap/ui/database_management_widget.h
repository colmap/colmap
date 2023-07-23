// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include "colmap/scene/database.h"
#include "colmap/ui/image_viewer_widget.h"
#include "colmap/util/misc.h"
#include "colmap/util/option_manager.h"

#include <QtCore>
#include <QtWidgets>
#include <unordered_map>

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// Matches
////////////////////////////////////////////////////////////////////////////////

class TwoViewInfoTab : public QWidget {
 public:
  TwoViewInfoTab() {}
  TwoViewInfoTab(QWidget* parent, OptionManager* options, Database* database);

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
  FeatureImageViewerWidget* matches_viewer_widget_;
};

class MatchesTab : public TwoViewInfoTab {
 public:
  MatchesTab(QWidget* parent, OptionManager* options, Database* database);

  void Reload(const std::vector<Image>& images, image_t image_id);
};

class TwoViewGeometriesTab : public TwoViewInfoTab {
 public:
  TwoViewGeometriesTab(QWidget* parent,
                       OptionManager* options,
                       Database* database);

  void Reload(const std::vector<Image>& images, image_t image_id);
};

class OverlappingImagesWidget : public QWidget {
 public:
  OverlappingImagesWidget(QWidget* parent,
                          OptionManager* options,
                          Database* database);

  void ShowMatches(const std::vector<Image>& images, image_t image_id);

 private:
  void closeEvent(QCloseEvent* event);

  QWidget* parent_;

  OptionManager* options_;

  QTabWidget* tab_widget_;
  MatchesTab* matches_tab_;
  TwoViewGeometriesTab* two_view_geometries_tab_;
};

////////////////////////////////////////////////////////////////////////////////
// Images, Cameras
////////////////////////////////////////////////////////////////////////////////

class CameraTab : public QWidget {
 public:
  CameraTab(QWidget* parent, Database* database);

  void Reload();
  void Clear();

 private:
  void itemChanged(QTableWidgetItem* item);
  void Add();
  void SetModel();

  Database* database_;

  std::vector<Camera> cameras_;

  QTableWidget* table_widget_;
  QLabel* info_label_;
};

class ImageTab : public QWidget {
 public:
  ImageTab(QWidget* parent,
           CameraTab* camera_tab,
           OptionManager* options,
           Database* database);

  void Reload();
  void Clear();

 private:
  void itemChanged(QTableWidgetItem* item);

  void ShowImage();
  void ShowMatches();
  void SetCamera();
  void SplitCamera();

  CameraTab* camera_tab_;

  OptionManager* options_;
  Database* database_;

  std::vector<Image> images_;

  QTableWidget* table_widget_;
  QLabel* info_label_;

  OverlappingImagesWidget* overlapping_images_widget_;

  FeatureImageViewerWidget* image_viewer_widget_;
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
  Database database_;

  QTabWidget* tab_widget_;
  ImageTab* image_tab_;
  CameraTab* camera_tab_;
};

}  // namespace colmap
