// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"

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
