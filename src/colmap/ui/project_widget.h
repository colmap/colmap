// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/option_manager.h"

#include <QtCore>
#include <QtWidgets>

namespace colmap {

class ProjectWidget : public QWidget {
 public:
  ProjectWidget(QWidget* parent, OptionManager* options);

  bool IsValid() const;
  void Reset();

  std::filesystem::path GetDatabasePath() const;
  std::filesystem::path GetImagePath() const;
  void SetDatabasePath(const std::filesystem::path& path);
  void SetImagePath(const std::filesystem::path& path);

 private:
  void Save();
  void SelectNewDatabasePath();
  void SelectExistingDatabasePath();
  void SelectImagePath();
  QString DefaultDirectory();

  OptionManager* options_;

  // Whether file dialog was opened previously.
  bool prev_selected_;

  // Text boxes that hold the currently selected paths.
  QLineEdit* database_path_text_;
  QLineEdit* image_path_text_;
};

}  // namespace colmap
