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

#include "colmap/ui/project_widget.h"

#include "colmap/base/database.h"

namespace colmap {

ProjectWidget::ProjectWidget(QWidget* parent, OptionManager* options)
    : QWidget(parent), options_(options), prev_selected_(false) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Project");

  // Database path.
  QPushButton* databse_path_new = new QPushButton(tr("New"), this);
  connect(databse_path_new,
          &QPushButton::released,
          this,
          &ProjectWidget::SelectNewDatabasePath);
  QPushButton* databse_path_open = new QPushButton(tr("Open"), this);
  connect(databse_path_open,
          &QPushButton::released,
          this,
          &ProjectWidget::SelectExistingDatabasePath);
  database_path_text_ = new QLineEdit(this);
  database_path_text_->setText(
      QString::fromStdString(*options_->database_path));

  // Image path.
  QPushButton* image_path_select = new QPushButton(tr("Select"), this);
  connect(image_path_select,
          &QPushButton::released,
          this,
          &ProjectWidget::SelectImagePath);
  image_path_text_ = new QLineEdit(this);
  image_path_text_->setText(QString::fromStdString(*options_->image_path));

  // Save button.
  QPushButton* create_button = new QPushButton(tr("Save"), this);
  connect(create_button, &QPushButton::released, this, &ProjectWidget::Save);

  QGridLayout* grid = new QGridLayout(this);

  grid->addWidget(new QLabel(tr("Database"), this), 0, 0);
  grid->addWidget(database_path_text_, 0, 1);
  grid->addWidget(databse_path_new, 0, 2);
  grid->addWidget(databse_path_open, 0, 3);

  grid->addWidget(new QLabel(tr("Images"), this), 1, 0);
  grid->addWidget(image_path_text_, 1, 1);
  grid->addWidget(image_path_select, 1, 2);

  grid->addWidget(create_button, 2, 2);
}

bool ProjectWidget::IsValid() const {
  return ExistsDir(GetImagePath()) && !ExistsDir(GetDatabasePath()) &&
         ExistsDir(GetParentDir(GetDatabasePath()));
}

void ProjectWidget::Reset() {
  database_path_text_->clear();
  image_path_text_->clear();
}

std::string ProjectWidget::GetDatabasePath() const {
  return database_path_text_->text().toUtf8().constData();
}

std::string ProjectWidget::GetImagePath() const {
  return image_path_text_->text().toUtf8().constData();
}

void ProjectWidget::SetDatabasePath(const std::string& path) {
  database_path_text_->setText(QString::fromStdString(path));
}

void ProjectWidget::SetImagePath(const std::string& path) {
  image_path_text_->setText(QString::fromStdString(path));
}

void ProjectWidget::Save() {
  if (IsValid()) {
    *options_->database_path = GetDatabasePath();
    *options_->image_path = GetImagePath();

    // Save empty database file.
    Database database(*options_->database_path);

    hide();
  } else {
    QMessageBox::critical(this, "", tr("Invalid paths"));
  }
}

void ProjectWidget::SelectNewDatabasePath() {
  QString database_path =
      QFileDialog::getSaveFileName(this,
                                   tr("Select database file"),
                                   DefaultDirectory(),
                                   tr("SQLite3 database (*.db)"));
  if (database_path != "") {
    if (!HasFileExtension(database_path.toUtf8().constData(), ".db")) {
      database_path += ".db";
    }
    database_path_text_->setText(database_path);
  }
}

void ProjectWidget::SelectExistingDatabasePath() {
  const auto database_path =
      QFileDialog::getOpenFileName(this,
                                   tr("Select database file"),
                                   DefaultDirectory(),
                                   tr("SQLite3 database (*.db)"));
  if (database_path != "") {
    database_path_text_->setText(database_path);
  }
}

void ProjectWidget::SelectImagePath() {
  const auto image_path =
      QFileDialog::getExistingDirectory(this,
                                        tr("Select image path..."),
                                        DefaultDirectory(),
                                        QFileDialog::ShowDirsOnly);
  if (image_path != "") {
    image_path_text_->setText(image_path);
  }
}

QString ProjectWidget::DefaultDirectory() {
  if (prev_selected_) {
    return "";
  }

  prev_selected_ = true;

  if (!options_->project_path->empty()) {
    const auto parent_path = GetParentDir(*options_->project_path);
    if (ExistsDir(parent_path)) {
      return QString::fromStdString(parent_path);
    }
  }

  if (!database_path_text_->text().isEmpty()) {
    const auto parent_path =
        GetParentDir(database_path_text_->text().toUtf8().constData());
    if (ExistsDir(parent_path)) {
      return QString::fromStdString(parent_path);
    }
  }

  if (!image_path_text_->text().isEmpty()) {
    return image_path_text_->text();
  }

  return "";
}

}  // namespace colmap
