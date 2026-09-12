// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/hash_containers.h"

#include <QtCore>
#include <QtWidgets>
#include <filesystem>

namespace colmap {

class OptionsWidget : public QWidget {
 public:
  explicit OptionsWidget(QWidget* parent);

  void AddOptionRow(const std::string& label_text,
                    QWidget* widget,
                    void* option);
  void AddWidgetRow(const std::string& label_text, QWidget* widget);
  void AddLayoutRow(const std::string& label_text, QLayout* layout);

  QSpinBox* AddOptionInt(int* option,
                         const std::string& label_text,
                         int min = 0,
                         int max = static_cast<int>(1e7));
  // Like AddOptionInt, but for options whose "unlimited" sentinel
  // (std::numeric_limits<int>::max()) is shown and edited as -1, displayed as
  // "unlimited". A plain QSpinBox capped at INT_MAX overflows when stepped, so
  // the editable range is capped below INT_MAX.
  QSpinBox* AddOptionIntUnlimited(int* option,
                                  const std::string& label_text,
                                  int max = static_cast<int>(2e9));
  QDoubleSpinBox* AddOptionDouble(double* option,
                                  const std::string& label_text,
                                  double min = 0,
                                  double max = 1e7,
                                  double step = 0.01,
                                  int decimals = 2);
  QDoubleSpinBox* AddOptionDoubleLog(double* option,
                                     const std::string& label_text,
                                     double min = 0,
                                     double max = 1e7,
                                     double step = 0.01,
                                     int decimals = 2);
  QCheckBox* AddOptionBool(bool* option, const std::string& label_text);
  QLineEdit* AddOptionText(std::string* option, const std::string& label_text);
  QLineEdit* AddOptionFilePath(std::filesystem::path* option,
                               const std::string& label_text);
  QLineEdit* AddOptionDirPath(std::filesystem::path* option,
                              const std::string& label_text);

  void AddSpacer();
  void AddSection(const std::string& title);

  void ReadOptions();
  void WriteOptions();

 protected:
  void showEvent(QShowEvent* event);
  void closeEvent(QCloseEvent* event);
  void hideEvent(QHideEvent* event);

  void ShowOption(void* option);
  void HideOption(void* option);

  void ShowWidget(QWidget* option);
  void HideWidget(QWidget* option);

  void ShowLayout(QLayout* option);
  void HideLayout(QLayout* option);

  QGridLayout* grid_layout_;

  NodeHashMap<void*, std::pair<QLabel*, QWidget*>> option_rows_;
  NodeHashMap<QWidget*, std::pair<QLabel*, QWidget*>> widget_rows_;
  NodeHashMap<QLayout*, std::pair<QLabel*, QWidget*>> layout_rows_;

  std::vector<std::pair<QSpinBox*, int*>> options_int_;
  std::vector<std::pair<QSpinBox*, int*>> options_int_unlimited_;
  std::vector<std::pair<QDoubleSpinBox*, double*>> options_double_;
  std::vector<std::pair<QDoubleSpinBox*, double*>> options_double_log_;
  std::vector<std::pair<QCheckBox*, bool*>> options_bool_;
  std::vector<std::pair<QLineEdit*, std::string*>> options_text_;
  std::vector<std::pair<QLineEdit*, std::filesystem::path*>> options_path_;

 private:
  QLabel* CreateRowLabel(const std::string& label_text);
  QLineEdit* AddOptionPath(std::filesystem::path* option,
                           const std::string& label_text,
                           bool directory);

  template <typename Key>
  static void SetRowVisible(
      NodeHashMap<Key, std::pair<QLabel*, QWidget*>>& rows,
      const Key& key,
      bool visible) {
    auto& row = rows.at(key);
    row.first->setVisible(visible);
    row.second->setVisible(visible);
  }
};

}  // namespace colmap
