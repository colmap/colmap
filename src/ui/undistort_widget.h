// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#ifndef COLMAP_SRC_UI_UNDISTORT_WIDGET_H_
#define COLMAP_SRC_UI_UNDISTORT_WIDGET_H_

#include <boost/filesystem.hpp>

#include <QtCore>
#include <QtWidgets>

#include "ui/opengl_window.h"
#include "util/misc.h"

namespace colmap {

class UndistortWidget : public QWidget {
 public:
  UndistortWidget(QWidget* parent, OptionManager* options);

  std::string GetOutputPath();

  bool IsValid();

  Reconstruction reconstruction;

 private:
  void Undistort();
  void SelectOutputPath();
  void ShowProgressBar();

  OptionManager* options_;

  QComboBox* combo_box_;
  QDoubleSpinBox* min_scale_sb_;
  QDoubleSpinBox* max_scale_sb_;
  QSpinBox* max_image_size_sb_;
  QDoubleSpinBox* blank_pixels_sb_;
  QLineEdit* output_path_text_;

  QProgressDialog* progress_bar_;
  QAction* destructor_;
  std::unique_ptr<Thread> undistorter_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_UNDISTORT_WIDGET_H_
