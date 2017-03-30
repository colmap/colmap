// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "sfm/controllers.h"

namespace colmap {

BundleAdjustmentWidget::BundleAdjustmentWidget(QWidget* parent,
                                               OptionManager* options)
    : OptionsWidget(parent),
      options_(options),
      reconstruction_(nullptr),
      action_render_now_(nullptr),
      thread_control_widget_(new ThreadControlWidget(this)) {
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
}

void BundleAdjustmentWidget::Show(Reconstruction* reconstruction,
                                  QAction* action_render_now) {
  reconstruction_ = reconstruction;
  action_render_now_ = action_render_now;
  show();
  raise();
}

void BundleAdjustmentWidget::Run() {
  CHECK_NOTNULL(reconstruction_);

  WriteOptions();

  Thread* thread = new BundleAdjustmentController(*options_, reconstruction_);
  thread->AddCallback(Thread::FINISHED_CALLBACK,
                      [this]() { action_render_now_->trigger(); });

  thread_control_widget_->StartThread("Bundle adjusting...", true, thread);
}

}  // namespace colmap
