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

#include "colmap/scene/reconstruction.h"

#include <QtCore>
#include <QtGui>
#include <QtWidgets>
#include <unordered_map>

namespace colmap {

class ModelViewerWidget;

class MovieGrabberWidget : public QWidget {
 public:
  MovieGrabberWidget(QWidget* parent, ModelViewerWidget* model_viewer_widget);

  // List of views, used to visualize the movie grabber camera path.
  std::vector<Image> views;

  struct ViewData {
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

  ModelViewerWidget* model_viewer_widget_;

  QPushButton* assemble_button_;
  QPushButton* add_button_;
  QPushButton* delete_button_;
  QPushButton* clear_button_;
  QTableWidget* table_;

  QSpinBox* frame_rate_sb_;
  QCheckBox* smooth_cb_;
  QDoubleSpinBox* smoothness_sb_;

  std::unordered_map<const QTableWidgetItem*, ViewData> view_data_;
};

}  // namespace colmap
