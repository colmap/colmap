// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/ui/project_widget.h"

#include "colmap/scene/database.h"
#include "colmap/util/file.h"

namespace colmap {

ProjectWidget::ProjectWidget(QWidget* parent, OptionManager* options)
    : QWidget(parent), options_(options), prev_selected_(false) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Project");

  // Database path.
  QPushButton* database_path_new = new QPushButton(tr("New"), this);
  connect(database_path_new,
          &QPushButton::released,
          this,
          &ProjectWidget::SelectNewDatabasePath);
  QPushButton* database_path_open = new QPushButton(tr("Open"), this);
  connect(database_path_open,
          &QPushButton::released,
          this,
          &ProjectWidget::SelectExistingDatabasePath);
  database_path_text_ = new QLineEdit(this);
  database_path_text_->setText(
      QString::fromStdString(options_->database_path->string()));

  // Image path.
  QPushButton* image_path_select = new QPushButton(tr("Select"), this);
  connect(image_path_select,
          &QPushButton::released,
          this,
          &ProjectWidget::SelectImagePath);
  image_path_text_ = new QLineEdit(this);
  image_path_text_->setText(
      QString::fromStdString(options_->image_path->string()));

  // Save button.
  QPushButton* create_button = new QPushButton(tr("Save"), this);
  connect(create_button, &QPushButton::released, this, &ProjectWidget::Save);

  QGridLayout* grid = new QGridLayout(this);

  grid->addWidget(new QLabel(tr("Database"), this), 0, 0);
  grid->addWidget(database_path_text_, 0, 1);
  grid->addWidget(database_path_new, 0, 2);
  grid->addWidget(database_path_open, 0, 3);

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

std::filesystem::path ProjectWidget::GetDatabasePath() const {
  return database_path_text_->text().toUtf8().constData();
}

std::filesystem::path ProjectWidget::GetImagePath() const {
  return image_path_text_->text().toUtf8().constData();
}

void ProjectWidget::SetDatabasePath(const std::filesystem::path& path) {
  database_path_text_->setText(QString::fromStdString(path.string()));
}

void ProjectWidget::SetImagePath(const std::filesystem::path& path) {
  image_path_text_->setText(QString::fromStdString(path.string()));
}

void ProjectWidget::Save() {
  if (IsValid()) {
    *options_->database_path = GetDatabasePath();
    *options_->image_path = GetImagePath();

    // Save empty database file.
    auto database = Database::Open(*options_->database_path);

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
      return QString::fromStdString(parent_path.string());
    }
  }

  if (!database_path_text_->text().isEmpty()) {
    const auto parent_path =
        GetParentDir(database_path_text_->text().toUtf8().constData());
    if (ExistsDir(parent_path)) {
      return QString::fromStdString(parent_path.string());
    }
  }

  if (!image_path_text_->text().isEmpty()) {
    return image_path_text_->text();
  }

  return "";
}

}  // namespace colmap
