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

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/scene/database.h"
#include "colmap/scene/projection.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/ui/qt_utils.h"

#include <QtCore>
#include <QtWidgets>

namespace colmap {

class ModelViewerWidget;

class ImageViewerGraphicsScene : public QGraphicsScene {
 public:
  ImageViewerGraphicsScene();

  QGraphicsPixmapItem* ImagePixmapItem() const;

 private:
  QGraphicsPixmapItem* image_pixmap_item_ = nullptr;
};

class ImageViewerWidget : public QWidget {
 public:
  explicit ImageViewerWidget(QWidget* parent);

  void ShowBitmap(const Bitmap& bitmap);
  void ShowPixmap(const QPixmap& pixmap);
  void ReadAndShow(const std::string& path);

 private:
  static const double kZoomFactor;

  ImageViewerGraphicsScene graphics_scene_;
  QGraphicsView* graphics_view_;

 protected:
  void resizeEvent(QResizeEvent* event);
  void closeEvent(QCloseEvent* event);
  void ZoomIn();
  void ZoomOut();
  void Save();

  QGridLayout* grid_layout_;
  QHBoxLayout* button_layout_;
};

class FeatureImageViewerWidget : public ImageViewerWidget {
 public:
  FeatureImageViewerWidget(QWidget* parent, const std::string& switch_text);

  void ReadAndShowWithKeypoints(const std::string& path,
                                const FeatureKeypoints& keypoints,
                                const std::vector<char>& tri_mask);

  void ReadAndShowWithMatches(const std::string& path1,
                              const std::string& path2,
                              const FeatureKeypoints& keypoints1,
                              const FeatureKeypoints& keypoints2,
                              const FeatureMatches& matches);

 protected:
  void ShowOrHide();

  QPixmap image1_;
  QPixmap image2_;
  bool switch_state_;
  QPushButton* switch_button_;
  const std::string switch_text_;
};

class DatabaseImageViewerWidget : public FeatureImageViewerWidget {
 public:
  DatabaseImageViewerWidget(QWidget* parent,
                            ModelViewerWidget* model_viewer_widget,
                            OptionManager* options);

  void ShowImageWithId(image_t image_id);

 private:
  void ResizeTable();
  void DeleteImage();

  ModelViewerWidget* model_viewer_widget_;

  OptionManager* options_;

  QPushButton* delete_button_;

  image_t image_id_;

  QTableWidget* table_widget_;
  QTableWidgetItem* image_id_item_;
  QTableWidgetItem* camera_id_item_;
  QTableWidgetItem* camera_model_item_;
  QTableWidgetItem* camera_params_item_;
  QTableWidgetItem* rotation_item_;
  QTableWidgetItem* translation_item_;
  QTableWidgetItem* dimensions_item_;
  QTableWidgetItem* num_points2D_item_;
  QTableWidgetItem* num_points3D_item_;
  QTableWidgetItem* name_item_;
};

}  // namespace colmap
