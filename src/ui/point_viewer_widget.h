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

#ifndef COLMAP_SRC_UI_POINT_VIEWER_WIDGET_H_
#define COLMAP_SRC_UI_POINT_VIEWER_WIDGET_H_

#include <QtCore>
#include <QtWidgets>

#include "base/reconstruction.h"
#include "util/option_manager.h"

namespace colmap {

class OpenGLWindow;

class PointViewerWidget : public QWidget {
 public:
  PointViewerWidget(QWidget* parent, OpenGLWindow* opengl_window,
                    OptionManager* option);

  void Show(const point3D_t point3D_id);

 private:
  void closeEvent(QCloseEvent* event);

  void ClearLocations();
  void UpdateImages();
  void ZoomIn();
  void ZoomOut();
  void Delete();

  OpenGLWindow* opengl_window_;

  OptionManager* options_;

  QPushButton* delete_button_;

  point3D_t point3D_id_;

  QTableWidget* location_table_;
  std::vector<QPixmap> location_pixmaps_;
  std::vector<QLabel*> location_labels_;
  std::vector<image_t> image_ids_;
  std::vector<double> reproj_errors_;

  QPushButton* zoom_in_button_;
  QPushButton* zoom_out_button_;

  double zoom_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_POINT_VIEWER_WIDGET_H_
