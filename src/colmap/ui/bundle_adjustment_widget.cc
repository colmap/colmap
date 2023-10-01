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

#include "colmap/ui/bundle_adjustment_widget.h"

#include "colmap/controllers/bundle_adjustment.h"
#include "colmap/ui/main_window.h"

namespace colmap {

BundleAdjustmentWidget::BundleAdjustmentWidget(MainWindow* main_window,
                                               OptionManager* options)
    : OptionsWidget(main_window),
      main_window_(main_window),
      options_(options),
      reconstruction_(nullptr),
      thread_control_widget_(new ThreadControlWidget(this)) {
  setWindowTitle("Bundle adjustment");

  AddOptionInt(&options->bundle_adjustment->solver_options.max_num_iterations,
               "max_num_iterations");
  AddOptionInt(
      &options->bundle_adjustment->solver_options.max_linear_solver_iterations,
      "max_linear_solver_iterations");

  AddOptionDoubleLog(
      &options->bundle_adjustment->solver_options.function_tolerance,
      "function_tolerance [10eX]",
      -1000,
      1000);
  AddOptionDoubleLog(
      &options->bundle_adjustment->solver_options.gradient_tolerance,
      "gradient_tolerance [10eX]",
      -1000,
      1000);
  AddOptionDoubleLog(
      &options->bundle_adjustment->solver_options.parameter_tolerance,
      "parameter_tolerance [10eX]",
      -1000,
      1000);

  AddOptionBool(&options->bundle_adjustment->refine_focal_length,
                "refine_focal_length");
  AddOptionBool(&options->bundle_adjustment->refine_principal_point,
                "refine_principal_point");
  AddOptionBool(&options->bundle_adjustment->refine_extra_params,
                "refine_extra_params");
  AddOptionBool(&options->bundle_adjustment->refine_extrinsics,
                "refine_extrinsics");

  QPushButton* run_button = new QPushButton(tr("Run"), this);
  grid_layout_->addWidget(run_button, grid_layout_->rowCount(), 1);
  connect(
      run_button, &QPushButton::released, this, &BundleAdjustmentWidget::Run);

  render_action_ = new QAction(this);
  connect(render_action_,
          &QAction::triggered,
          this,
          &BundleAdjustmentWidget::Render,
          Qt::QueuedConnection);
}

void BundleAdjustmentWidget::Show(
    std::shared_ptr<Reconstruction> reconstruction) {
  reconstruction_ = std::move(reconstruction);
  show();
  raise();
}

void BundleAdjustmentWidget::Run() {
  CHECK_NOTNULL(reconstruction_);

  WriteOptions();

  auto thread =
      std::make_unique<BundleAdjustmentController>(*options_, reconstruction_);
  thread->AddCallback(Thread::FINISHED_CALLBACK,
                      [this]() { render_action_->trigger(); });

  // Normalize scene for numerical stability and
  // to avoid large scale changes in viewer.
  reconstruction_->Normalize();

  thread_control_widget_->StartThread(
      "Bundle adjusting...", true, std::move(thread));
}

void BundleAdjustmentWidget::Render() { main_window_->RenderNow(); }

}  // namespace colmap
