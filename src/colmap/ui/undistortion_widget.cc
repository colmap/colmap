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

#include "colmap/ui/undistortion_widget.h"

namespace colmap {

UndistortionWidget::UndistortionWidget(QWidget* parent,
                                       const OptionManager* options)
    : OptionsWidget(parent),
      options_(options),
      reconstruction_(nullptr),
      thread_control_widget_(new ThreadControlWidget(this)) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Undistortion");

  output_format_ = new QComboBox(this);
  output_format_->addItem("COLMAP");
  output_format_->addItem("PMVS");
  output_format_->addItem("CMP-MVS");
  output_format_->setFont(font());
  AddWidgetRow("format", output_format_);

  AddOptionDouble(&undistortion_options_.min_scale, "min_scale", 0);
  AddOptionDouble(&undistortion_options_.max_scale, "max_scale", 0);
  AddOptionInt(&undistortion_options_.max_image_size, "max_image_size", -1);
  AddOptionDouble(&undistortion_options_.blank_pixels, "blank_pixels", 0);
  AddOptionDouble(&undistortion_options_.roi_min_x, "roi_min_x", 0.0, 1.0);
  AddOptionDouble(&undistortion_options_.roi_min_y, "roi_min_y", 0.0, 1.0);
  AddOptionDouble(&undistortion_options_.roi_max_x, "roi_max_x", 0.0, 1.0);
  AddOptionDouble(&undistortion_options_.roi_max_y, "roi_max_y", 0.0, 1.0);
  AddOptionDirPath(&output_path_, "output_path");

  AddSpacer();

  QPushButton* undistort_button = new QPushButton(tr("Undistort"), this);
  connect(undistort_button,
          &QPushButton::released,
          this,
          &UndistortionWidget::Undistort);
  grid_layout_->addWidget(undistort_button, grid_layout_->rowCount(), 1);
}

void UndistortionWidget::Show(
    std::shared_ptr<const Reconstruction> reconstruction) {
  reconstruction_ = std::move(reconstruction);
  show();
  raise();
}

bool UndistortionWidget::IsValid() const { return ExistsDir(output_path_); }

void UndistortionWidget::Undistort() {
  THROW_CHECK_NOTNULL(reconstruction_);

  WriteOptions();

  if (IsValid()) {
    std::unique_ptr<Thread> undistorter;

    if (output_format_->currentIndex() == 0) {
      undistorter = std::make_unique<COLMAPUndistorter>(undistortion_options_,
                                                        *reconstruction_,
                                                        *options_->image_path,
                                                        output_path_);
    } else if (output_format_->currentIndex() == 1) {
      undistorter = std::make_unique<PMVSUndistorter>(undistortion_options_,
                                                      *reconstruction_,
                                                      *options_->image_path,
                                                      output_path_);
    } else if (output_format_->currentIndex() == 2) {
      undistorter = std::make_unique<CMPMVSUndistorter>(undistortion_options_,
                                                        *reconstruction_,
                                                        *options_->image_path,
                                                        output_path_);
    } else {
      QMessageBox::critical(this, "", tr("Invalid output format"));
      return;
    }

    thread_control_widget_->StartThread(
        "Undistorting...", true, std::move(undistorter));
  } else {
    QMessageBox::critical(this, "", tr("Invalid output path"));
  }
}

}  // namespace colmap
