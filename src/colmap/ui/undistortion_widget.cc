// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/ui/undistortion_widget.h"

#include "colmap/util/controller_thread.h"

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

  AddOptionDouble(&camera_options_.min_scale, "min_scale", 0);
  AddOptionDouble(&camera_options_.max_scale, "max_scale", 0);
  AddOptionInt(&camera_options_.max_image_size, "max_image_size", -1);
  AddOptionDouble(&camera_options_.blank_pixels, "blank_pixels", 0);
  AddOptionDouble(&camera_options_.roi_min_x, "roi_min_x", 0.0, 1.0);
  AddOptionDouble(&camera_options_.roi_min_y, "roi_min_y", 0.0, 1.0);
  AddOptionDouble(&camera_options_.roi_max_x, "roi_max_x", 0.0, 1.0);
  AddOptionDouble(&camera_options_.roi_max_y, "roi_max_y", 0.0, 1.0);
  AddOptionInt(&colmap_options_.jpeg_quality, "jpeg_quality", -1);
  AddOptionInt(&num_threads_, "num_threads", -1);
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
      colmap_options_.num_threads = num_threads_;
      undistorter = std::make_unique<ControllerThread<COLMAPUndistorter>>(
          std::make_shared<COLMAPUndistorter>(colmap_options_,
                                              camera_options_,
                                              *reconstruction_,
                                              *options_->image_path,
                                              output_path_));
    } else if (output_format_->currentIndex() == 1) {
      PMVSUndistorter::Options pmvs_options;
      pmvs_options.jpeg_quality = colmap_options_.jpeg_quality;
      pmvs_options.num_threads = num_threads_;
      undistorter = std::make_unique<ControllerThread<PMVSUndistorter>>(
          std::make_shared<PMVSUndistorter>(pmvs_options,
                                            camera_options_,
                                            *reconstruction_,
                                            *options_->image_path,
                                            output_path_));
    } else if (output_format_->currentIndex() == 2) {
      CMPMVSUndistorter::Options cmpmvs_options;
      cmpmvs_options.jpeg_quality = colmap_options_.jpeg_quality;
      cmpmvs_options.num_threads = num_threads_;
      undistorter = std::make_unique<ControllerThread<CMPMVSUndistorter>>(
          std::make_shared<CMPMVSUndistorter>(cmpmvs_options,
                                              camera_options_,
                                              *reconstruction_,
                                              *options_->image_path,
                                              output_path_));
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
