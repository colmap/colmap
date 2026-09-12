// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"
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
  void ReadAndShow(const std::filesystem::path& path);

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

  void ReadAndShowWithKeypoints(const std::filesystem::path& path,
                                const FeatureKeypoints& keypoints,
                                const std::vector<char>& tri_mask);

  void ReadAndShowWithMatches(const std::filesystem::path& path1,
                              const std::filesystem::path& path2,
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
  QTableWidgetItem* frame_id_item_;
  QTableWidgetItem* rig_id_item_;
  QTableWidgetItem* camera_item_;
  QTableWidgetItem* cam_from_world_item_;
  QTableWidgetItem* num_points2D_item_;
  QTableWidgetItem* num_points3D_item_;
  QTableWidgetItem* name_item_;
};

}  // namespace colmap
