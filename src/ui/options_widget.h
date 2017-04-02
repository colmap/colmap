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

#ifndef COLMAP_SRC_UI_OPTIONS_WIDGET_H_
#define COLMAP_SRC_UI_OPTIONS_WIDGET_H_

#include <QtCore>
#include <QtWidgets>

namespace colmap {

class OptionsWidget : public QWidget {
 public:
  explicit OptionsWidget(QWidget* parent);

 protected:
  void showEvent(QShowEvent* event);
  void closeEvent(QCloseEvent* event);
  void hideEvent(QHideEvent* event);

  void AddOptionRow(const std::string& label_text, QWidget* widget);

  QSpinBox* AddOptionInt(int* option, const std::string& label_text,
                         const int min = 0,
                         const int max = static_cast<int>(1e7));
  QDoubleSpinBox* AddOptionDouble(double* option, const std::string& label_text,
                                  const double min = 0, const double max = 1e7,
                                  const double step = 0.01,
                                  const int decimals = 2);
  QDoubleSpinBox* AddOptionDoubleLog(
      double* option, const std::string& label_text, const double min = 0,
      const double max = 1e7, const double step = 0.01, const int decimals = 2);
  QCheckBox* AddOptionBool(bool* option, const std::string& label_text);
  QLineEdit* AddOptionText(std::string* option, const std::string& label_text);
  QLineEdit* AddOptionFilePath(std::string* option,
                               const std::string& label_text);
  QLineEdit* AddOptionDirPath(std::string* option,
                              const std::string& label_text);
  void AddSpacer();
  void AddSection(const std::string& label_text);

  void ReadOptions();
  void WriteOptions();

  QGridLayout* grid_layout_;

  std::vector<std::pair<QSpinBox*, int*>> options_int_;
  std::vector<std::pair<QDoubleSpinBox*, double*>> options_double_;
  std::vector<std::pair<QDoubleSpinBox*, double*>> options_double_log_;
  std::vector<std::pair<QCheckBox*, bool*>> options_bool_;
  std::vector<std::pair<QLineEdit*, std::string*>> options_text_;
  std::vector<std::pair<QLineEdit*, std::string*>> options_path_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_OPTIONS_WIDGET_H_
