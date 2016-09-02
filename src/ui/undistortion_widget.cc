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

#include "ui/undistortion_widget.h"

namespace colmap {

UndistortionWidget::UndistortionWidget(QWidget* parent,
                                       const OptionManager* options)
    : OptionsWidget(parent),
      options_(options),
      reconstruction_(nullptr),
      progress_bar_(nullptr) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Undistortion");

  output_format_ = new QComboBox(this);
  output_format_->addItem("COLMAP");
  output_format_->addItem("PMVS");
  output_format_->addItem("CMP-MVS");
  output_format_->setFont(font());

  AddOptionRow("format", output_format_);

  UndistortCameraOptions default_options;

  AddOptionDouble(&undistortion_options_.min_scale, "min_scale", 0);
  AddOptionDouble(&undistortion_options_.max_scale, "max_scale", 0);
  AddOptionInt(&undistortion_options_.max_image_size, "max_image_size", -1);
  AddOptionDouble(&undistortion_options_.blank_pixels, "blank_pixels", 0);
  AddOptionDirPath(&output_path_, "output_path");

  AddSpacer();

  QPushButton* undistort_button = new QPushButton(tr("Undistort"), this);
  connect(undistort_button, &QPushButton::released, this,
          &UndistortionWidget::Undistort);
  grid_layout_->addWidget(undistort_button, grid_layout_->rowCount(), 1);

  destructor_ = new QAction(this);
  connect(destructor_, &QAction::triggered, this, [this]() {
    if (undistorter_) {
      undistorter_->Stop();
      undistorter_->Wait();
      undistorter_.reset();
    }
    progress_bar_->hide();
  });
}

void UndistortionWidget::Show(const Reconstruction& reconstruction) {
  reconstruction_ = &reconstruction;
  show();
  raise();
}

bool UndistortionWidget::IsValid() const {
  return boost::filesystem::is_directory(output_path_);
}

void UndistortionWidget::ShowProgressBar() {
  if (progress_bar_ == nullptr) {
    progress_bar_ = new QProgressDialog(this);
    progress_bar_->setWindowModality(Qt::ApplicationModal);
    progress_bar_->setLabel(new QLabel(tr("Undistorting..."), this));
    progress_bar_->setMaximum(0);
    progress_bar_->setMinimum(0);
    progress_bar_->setValue(0);
    connect(progress_bar_, &QProgressDialog::canceled,
            [this]() { destructor_->trigger(); });
  }
  progress_bar_->show();
  progress_bar_->raise();
}

void UndistortionWidget::Undistort() {
  CHECK_NOTNULL(reconstruction_);

  WriteOptions();

  if (!IsValid()) {
    QMessageBox::critical(this, "", tr("Invalid output path"));
  } else {
    if (output_format_->currentIndex() == 0) {
      undistorter_.reset(
          new COLMAPUndistorter(undistortion_options_, *reconstruction_,
                                *options_->image_path, output_path_));
      undistorter_->SetCallback(COLMAPUndistorter::FINISHED_CALLBACK,
                                [this]() { destructor_->trigger(); });
    } else if (output_format_->currentIndex() == 1) {
      undistorter_.reset(
          new PMVSUndistorter(undistortion_options_, *reconstruction_,
                              *options_->image_path, output_path_));
      undistorter_->SetCallback(PMVSUndistorter::FINISHED_CALLBACK,
                                [this]() { destructor_->trigger(); });
    } else if (output_format_->currentIndex() == 2) {
      undistorter_.reset(
          new CMPMVSUndistorter(undistortion_options_, *reconstruction_,
                                *options_->image_path, output_path_));
      undistorter_->SetCallback(CMPMVSUndistorter::FINISHED_CALLBACK,
                                [this]() { destructor_->trigger(); });
    } else {
      QMessageBox::critical(this, "", tr("Invalid output format"));
      return;
    }

    undistorter_->Start();

    ShowProgressBar();
  }
}

}  // namespace colmap
