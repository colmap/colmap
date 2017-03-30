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

#ifndef COLMAP_SRC_UI_MOVIE_GRABBER_WIDGET_H_
#define COLMAP_SRC_UI_MOVIE_GRABBER_WIDGET_H_

#include <unordered_map>

#include <QtCore>
#include <QtGui>
#include <QtWidgets>

#include "base/reconstruction.h"

namespace colmap {

class OpenGLWindow;

class MovieGrabberWidget : public QWidget {
 public:
  MovieGrabberWidget(QWidget* parent, OpenGLWindow* opengl_window);

  // List of views, used to visualize the movie grabber camera path.
  std::vector<Image> views;

  struct ViewData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    QMatrix4x4 model_view_matrix;
    float point_size = -1.0f;
    float image_size = -1.0f;
  };

 private:
  // Add, delete, clear viewpoints.
  void Add();
  void Delete();
  void Clear();

  // Assemble movie from current viewpoints.
  void Assemble();

  // Event slot for time modification.
  void TimeChanged(QTableWidgetItem* item);

  // Event slot for changed selection.
  void SelectionChanged(const QItemSelection& selected,
                        const QItemSelection& deselected);

  // Update state when viewpoints reordered.
  void UpdateViews();

  OpenGLWindow* opengl_window_;

  QPushButton* assemble_button_;
  QPushButton* add_button_;
  QPushButton* delete_button_;
  QPushButton* clear_button_;
  QTableWidget* table_;

  QSpinBox* frame_rate_sb_;
  QCheckBox* smooth_cb_;
  QDoubleSpinBox* smoothness_sb_;

  EIGEN_STL_UMAP(const QTableWidgetItem*, ViewData) view_data_;
};

}  // namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(
    colmap::MovieGrabberWidget::ViewData)

#endif  // COLMAP_SRC_UI_MOVIE_GRABBER_WIDGET_H_
