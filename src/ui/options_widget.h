// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
// Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

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
