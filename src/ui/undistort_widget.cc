// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#include "ui/undistort_widget.h"

#include "base/undistortion.h"

namespace colmap {

UndistortWidget::UndistortWidget(QWidget* parent, OptionManager* options)
    : QWidget(parent), options_(options) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Undistort images");

  QGridLayout* grid = new QGridLayout(this);

  grid->addWidget(new QLabel(tr("Format"), this), grid->rowCount(), 0);
  combo_box_ = new QComboBox(this);
  combo_box_->addItem("Default");
  combo_box_->addItem("PMVS");
  combo_box_->addItem("CMP-MVS");
  grid->addWidget(combo_box_, grid->rowCount() - 1, 1);

  UndistortCameraOptions default_options;

  grid->addWidget(new QLabel(tr("min_scale"), this), grid->rowCount(), 0);
  min_scale_sb_ = new QDoubleSpinBox(this);
  min_scale_sb_->setMinimum(0);
  min_scale_sb_->setSingleStep(0.01);
  min_scale_sb_->setDecimals(2);
  min_scale_sb_->setValue(default_options.min_scale);
  grid->addWidget(min_scale_sb_, grid->rowCount() - 1, 1);

  grid->addWidget(new QLabel(tr("max_scale"), this), grid->rowCount(), 0);
  max_scale_sb_ = new QDoubleSpinBox(this);
  max_scale_sb_->setMinimum(0);
  max_scale_sb_->setSingleStep(0.01);
  max_scale_sb_->setDecimals(2);
  max_scale_sb_->setValue(default_options.max_scale);
  grid->addWidget(max_scale_sb_, grid->rowCount() - 1, 1);

  grid->addWidget(new QLabel(tr("max_image_size"), this), grid->rowCount(), 0);
  max_image_size_sb_ = new QSpinBox(this);
  max_image_size_sb_->setMinimum(-1);
  max_image_size_sb_->setMaximum(1e6);
  max_image_size_sb_->setValue(default_options.max_image_size);
  grid->addWidget(max_image_size_sb_, grid->rowCount() - 1, 1);

  grid->addWidget(new QLabel(tr("blank_pixels"), this), grid->rowCount(), 0);
  blank_pixels_sb_ = new QDoubleSpinBox(this);
  blank_pixels_sb_->setMinimum(0);
  blank_pixels_sb_->setMaximum(1);
  blank_pixels_sb_->setSingleStep(0.01);
  blank_pixels_sb_->setDecimals(2);
  blank_pixels_sb_->setValue(default_options.blank_pixels);
  grid->addWidget(blank_pixels_sb_, grid->rowCount() - 1, 1);

  grid->addWidget(new QLabel(tr("Path"), this), grid->rowCount(), 0);
  output_path_text_ = new QLineEdit(this);
  grid->addWidget(output_path_text_, grid->rowCount() - 1, 1);

  QPushButton* output_path_select = new QPushButton(tr("Select"), this);
  connect(output_path_select, &QPushButton::released, this,
          &UndistortWidget::SelectOutputPath);
  grid->addWidget(output_path_select, grid->rowCount() - 1, 2);

  QPushButton* undistort_button = new QPushButton(tr("Undistort"), this);
  connect(undistort_button, &QPushButton::released, this,
          &UndistortWidget::Undistort);
  grid->addWidget(undistort_button, grid->rowCount(), 2);

  progress_bar_ = new QProgressDialog(this);
  progress_bar_->setWindowModality(Qt::ApplicationModal);
  progress_bar_->setLabel(new QLabel(tr("Undistorting..."), this));
  progress_bar_->setMaximum(0);
  progress_bar_->setMinimum(0);
  progress_bar_->setValue(0);
  progress_bar_->hide();
  progress_bar_->close();
}

bool UndistortWidget::IsValid() {
  return boost::filesystem::is_directory(GetOutputPath());
}

std::string UndistortWidget::GetOutputPath() {
  return EnsureTrailingSlash(output_path_text_->text().toUtf8().constData());
}

void UndistortWidget::SelectOutputPath() {
  output_path_text_->setText(QFileDialog::getExistingDirectory(
      this, tr("Select output path..."), "", QFileDialog::ShowDirsOnly));
}

void UndistortWidget::Undistort() {
  if (!IsValid()) {
    QMessageBox::critical(this, "", tr("Invalid output path"));
  } else {
    progress_bar_->show();
    progress_bar_->raise();

    UndistortCameraOptions options;
    options.min_scale = min_scale_sb_->value();
    options.max_scale = max_scale_sb_->value();
    options.blank_pixels = blank_pixels_sb_->value();
    options.max_image_size = max_image_size_sb_->value();

    ImageUndistorter* undistorter = nullptr;

    if (combo_box_->currentIndex() == 0) {
      undistorter = new ImageUndistorter(
          options, reconstruction, *options_->image_path, GetOutputPath());
    } else if (combo_box_->currentIndex() == 1) {
      undistorter = new PMVSUndistorter(options, reconstruction,
                                        *options_->image_path, GetOutputPath());
    } else if (combo_box_->currentIndex() == 2) {
      undistorter = new CMPMVSUndistorter(
          options, reconstruction, *options_->image_path, GetOutputPath());
    } else {
      QMessageBox::critical(this, "", tr("Invalid output format"));
      return;
    }

    // Execute after finished.
    connect(undistorter, &QThread::finished, progress_bar_,
            [this, undistorter]() {
              this->progress_bar_->hide();
              QMessageBox::information(this, "", tr("Undistortion finished"));
              undistorter->deleteLater();
            });

    connect(progress_bar_, &QProgressDialog::canceled, [undistorter]() {
      undistorter->Stop();
      undistorter->wait();
    });

    undistorter->start();
  }
}

}  // namespace colmap
