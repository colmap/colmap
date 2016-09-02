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

#include "ui/bundle_adjustment_widget.h"

namespace colmap {

BundleAdjustmentWidget::BundleAdjustmentWidget(QWidget* parent,
                                               OptionManager* options)
    : OptionsWidget(parent),
      options_(options),
      reconstruction_(nullptr),
      progress_bar_(nullptr) {
  setWindowTitle("Bundle adjustment");

  AddOptionInt(&options->ba_options->max_num_iterations, "max_num_iterations");
  AddOptionInt(&options->ba_options->max_linear_solver_iterations,
               "max_linear_solver_iterations");

  AddOptionDoubleLog(&options->ba_options->function_tolerance,
                     "function_tolerance [10eX]", -1000, 1000);
  AddOptionDoubleLog(&options->ba_options->gradient_tolerance,
                     "gradient_tolerance [10eX]", -1000, 1000);
  AddOptionDoubleLog(&options->ba_options->parameter_tolerance,
                     "parameter_tolerance [10eX]", -1000, 1000);

  AddOptionBool(&options->ba_options->refine_focal_length,
                "refine_focal_length");
  AddOptionBool(&options->ba_options->refine_principal_point,
                "refine_principal_point");
  AddOptionBool(&options->ba_options->refine_extra_params,
                "refine_extra_params");

  QPushButton* run_button = new QPushButton(tr("Run"), this);
  grid_layout_->addWidget(run_button, grid_layout_->rowCount(), 1);
  connect(run_button, &QPushButton::released, this,
          &BundleAdjustmentWidget::Run);

  destructor_ = new QAction(this);
  connect(destructor_, &QAction::triggered, this, [this]() {
    if (ba_controller_) {
      ba_controller_->Stop();
      ba_controller_->Wait();
      ba_controller_.reset();
    }
    progress_bar_->hide();
  });
}

void BundleAdjustmentWidget::Show(Reconstruction* reconstruction) {
  reconstruction_ = reconstruction;
  show();
  raise();
}

void BundleAdjustmentWidget::ShowProgressBar() {
  if (progress_bar_ == nullptr) {
    progress_bar_ = new QProgressDialog(this);
    progress_bar_->setWindowModality(Qt::ApplicationModal);
    progress_bar_->setLabel(new QLabel(tr("Bundle adjusting..."), this));
    progress_bar_->setMaximum(0);
    progress_bar_->setMinimum(0);
    progress_bar_->setValue(0);
    connect(progress_bar_, &QProgressDialog::canceled,
            [this]() { destructor_->trigger(); });
  }
  progress_bar_->show();
  progress_bar_->raise();
}

void BundleAdjustmentWidget::Run() {
  CHECK_NOTNULL(reconstruction_);

  WriteOptions();

  ba_controller_.reset(
      new BundleAdjustmentController(*options_, reconstruction_));
  ba_controller_->SetCallback(BundleAdjustmentController::FINISHED_CALLBACK,
                              [this]() { destructor_->trigger(); });
  ba_controller_->Start();

  ShowProgressBar();
}

}  // namespace colmap
