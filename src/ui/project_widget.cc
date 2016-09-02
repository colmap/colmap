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

#include "ui/project_widget.h"

#include "sfm/controllers.h"

namespace colmap {

ProjectWidget::ProjectWidget(QWidget* parent, OptionManager* options)
    : QWidget(parent), options_(options), prev_selected_(false) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Project");

  // Database path.
  QPushButton* databse_path_new = new QPushButton(tr("New"), this);
  connect(databse_path_new, &QPushButton::released, this,
          &ProjectWidget::SelectNewDatabasePath);
  QPushButton* databse_path_open = new QPushButton(tr("Open"), this);
  connect(databse_path_open, &QPushButton::released, this,
          &ProjectWidget::SelectExistingDatabasePath);
  database_path_text_ = new QLineEdit(this);
  database_path_text_->setText(
      QString::fromStdString(*options_->database_path));

  // Image path.
  QPushButton* image_path_select = new QPushButton(tr("Select"), this);
  connect(image_path_select, &QPushButton::released, this,
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
  return boost::filesystem::is_directory(ImagePath()) &&
         boost::filesystem::is_directory(
             boost::filesystem::path(DatabasePath()).parent_path());
}

void ProjectWidget::Reset() {
  database_path_text_->clear();
  image_path_text_->clear();
}

std::string ProjectWidget::DatabasePath() const {
  return database_path_text_->text().toUtf8().constData();
}

std::string ProjectWidget::ImagePath() const {
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
    *options_->database_path = DatabasePath();
    *options_->image_path = ImagePath();

    // Save empty database file.
    Database database(*options_->database_path);

    hide();
  } else {
    QMessageBox::critical(this, "", tr("Invalid paths."));
  }
}

void ProjectWidget::SelectNewDatabasePath() {
  QString database_path = QFileDialog::getSaveFileName(
      this, tr("Select database file"), DefaultDirectory(),
      tr("SQLite3 database (*.db)"));
  if (database_path != "" &&
      !HasFileExtension(database_path.toUtf8().constData(), ".db")) {
    database_path += ".db";
  }
  database_path_text_->setText(database_path);
}

void ProjectWidget::SelectExistingDatabasePath() {
  database_path_text_->setText(QFileDialog::getOpenFileName(
      this, tr("Select database file"), DefaultDirectory(),
      tr("SQLite3 database (*.db)")));
}

void ProjectWidget::SelectImagePath() {
  image_path_text_->setText(QFileDialog::getExistingDirectory(
      this, tr("Select image path..."), DefaultDirectory(),
      QFileDialog::ShowDirsOnly));
}

QString ProjectWidget::DefaultDirectory() {
  std::string directory_path = "";
  if (!prev_selected_ && !options_->project_path->empty()) {
    const boost::filesystem::path parent_path =
        boost::filesystem::path(*options_->project_path).parent_path();
    if (boost::filesystem::is_directory(parent_path)) {
      directory_path = parent_path.string();
    }
  }
  prev_selected_ = true;
  return QString::fromStdString(directory_path);
}

}  // namespace colmap
