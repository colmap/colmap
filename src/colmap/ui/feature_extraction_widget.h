// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"
#include "colmap/util/misc.h"

#include <QtCore>
#include <QtWidgets>

namespace colmap {

class FeatureExtractionWidget : public QWidget {
 public:
  FeatureExtractionWidget(QWidget* parent, OptionManager* options);

 private:
  void showEvent(QShowEvent* event);
  void hideEvent(QHideEvent* event);

  void ReadOptions();
  void WriteOptions();

  QGroupBox* CreateCameraModelBox();

  void SelectCameraModel(int code);
  void Extract();

  QWidget* parent_;

  OptionManager* options_;

  QComboBox* camera_model_cb_;
  QCheckBox* single_camera_cb_;
  QCheckBox* single_camera_per_folder_cb_;
  QRadioButton* camera_params_exif_rb_;
  QRadioButton* camera_params_custom_rb_;
  QLabel* camera_params_info_;
  QLineEdit* camera_params_text_;

  std::vector<int> camera_model_ids_;

  QTabWidget* tab_widget_;
};

}  // namespace colmap
