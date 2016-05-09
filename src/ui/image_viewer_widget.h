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

#ifndef COLMAP_SRC_UI_IMAGE_VIEWER_WIDGET_H_
#define COLMAP_SRC_UI_IMAGE_VIEWER_WIDGET_H_

#include <Eigen/Core>

#include <QtCore>
#include <QtWidgets>

#include "base/database.h"
#include "base/projection.h"
#include "base/reconstruction.h"
#include "ui/qt_utils.h"
#include "util/option_manager.h"

namespace colmap {

class OpenGLWindow;

class BasicImageViewerWidget : public QWidget {
 public:
  BasicImageViewerWidget(QWidget* parent, const std::string& switch_text);

  void Show(const std::string& path, const FeatureKeypoints& keypoints,
            const std::vector<bool>& tri_mask);

 protected:
  static const double kZoomFactor;

  void closeEvent(QCloseEvent* event);

  void UpdateImage();
  void ZoomIn();
  void ZoomOut();
  void ShowOrHide();

  OpenGLWindow* opengl_window_;

  QGridLayout* grid_;
  QHBoxLayout* button_layout_;

  QPixmap image1_;
  QPixmap image2_;

  QPushButton* show_button_;
  QPushButton* zoom_in_button_;
  QPushButton* zoom_out_button_;
  QScrollArea* image_scroll_area_;
  QLabel* image_label_;

  int orig_width_;
  double zoom_;
  bool switch_;
  const std::string switch_text_;
};

class MatchesImageViewerWidget : public BasicImageViewerWidget {
 public:
  MatchesImageViewerWidget(QWidget* parent);

  void Show(const std::string& path1, const std::string& path2,
            const FeatureKeypoints& keypoints1,
            const FeatureKeypoints& keypoints2, const FeatureMatches& matches);
};

class ImageViewerWidget : public BasicImageViewerWidget {
 public:
  ImageViewerWidget(QWidget* parent, OpenGLWindow* opengl_window,
                    OptionManager* options);

  void Show(const image_t image_id);

 private:
  void Resize();
  void Delete();

  OpenGLWindow* opengl_window_;

  OptionManager* options_;

  QPushButton* delete_button_;

  image_t image_id_;

  QTableWidget* table_widget_;
  QTableWidgetItem* image_id_item_;
  QTableWidgetItem* camera_id_item_;
  QTableWidgetItem* camera_model_item_;
  QTableWidgetItem* camera_params_item_;
  QTableWidgetItem* qvec_item_;
  QTableWidgetItem* tvec_item_;
  QTableWidgetItem* dimensions_item_;
  QTableWidgetItem* num_points2D_item_;
  QTableWidgetItem* num_points3D_item_;
  QTableWidgetItem* num_obs_item_;
  QTableWidgetItem* name_item_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_IMAGE_VIEWER_WIDGET_H_
