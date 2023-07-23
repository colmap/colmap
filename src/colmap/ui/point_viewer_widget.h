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
#include "colmap/util/option_manager.h"

#include <QtCore>
#include <QtWidgets>

namespace colmap {

class ModelViewerWidget;

class PointViewerWidget : public QWidget {
 public:
  PointViewerWidget(QWidget* parent,
                    ModelViewerWidget* model_viewer_widget,
                    OptionManager* option);

  void Show(point3D_t point3D_id);

 private:
  void closeEvent(QCloseEvent* event);

  void ResizeInfoTable();
  void ClearLocations();
  void UpdateImages();
  void ZoomIn();
  void ZoomOut();
  void Delete();

  ModelViewerWidget* model_viewer_widget_;

  OptionManager* options_;

  QPushButton* delete_button_;

  point3D_t point3D_id_;

  QTableWidget* info_table_;
  QTableWidgetItem* xyz_item_;
  QTableWidgetItem* rgb_item_;
  QTableWidgetItem* error_item_;

  QTableWidget* location_table_;
  std::vector<QPixmap> location_pixmaps_;
  std::vector<QLabel*> location_labels_;
  std::vector<image_t> image_ids_;
  std::vector<double> reproj_errors_;
  std::vector<std::string> image_names_;

  QPushButton* zoom_in_button_;
  QPushButton* zoom_out_button_;

  double zoom_;
};

}  // namespace colmap
