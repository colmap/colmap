// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/ui/bundle_adjustment_widget.h"

#include "colmap/controllers/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/ui/main_window.h"
#include "colmap/util/controller_thread.h"
#include "colmap/util/file.h"

namespace colmap {
namespace {

class BundleAdjustmentGeneralOptionsWidget : public OptionsWidget {
 public:
  BundleAdjustmentGeneralOptionsWidget(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionInt(
        &options->bundle_adjustment->ceres->solver_options.max_num_iterations,
        "max_num_iterations");
    AddOptionInt(&options->bundle_adjustment->ceres->solver_options
                      .max_linear_solver_iterations,
                 "max_linear_solver_iterations");

    AddOptionDoubleLog(
        &options->bundle_adjustment->ceres->solver_options.function_tolerance,
        "function_tolerance [10eX]",
        -1000,
        1000);
    AddOptionDoubleLog(
        &options->bundle_adjustment->ceres->solver_options.gradient_tolerance,
        "gradient_tolerance [10eX]",
        -1000,
        1000);
    AddOptionDoubleLog(
        &options->bundle_adjustment->ceres->solver_options.parameter_tolerance,
        "parameter_tolerance [10eX]",
        -1000,
        1000);

    AddOptionBool(&options->bundle_adjustment->refine_focal_length,
                  "refine_focal_length");
    AddOptionBool(&options->bundle_adjustment->refine_principal_point,
                  "refine_principal_point");
    AddOptionBool(&options->bundle_adjustment->refine_extra_params,
                  "refine_extra_params");
    AddOptionBool(&options->bundle_adjustment->refine_rig_from_world,
                  "refine_rig_from_world");
    AddOptionBool(&options->bundle_adjustment->refine_sensor_from_rig,
                  "refine_sensor_from_rig");
    AddOptionBool(&options->bundle_adjustment->refine_points3D,
                  "refine_points3D");
  }
};

#ifdef CASPAR_ENABLED
// Solver backend selection and Caspar-specific options.
class BundleAdjustmentBackendOptionsWidget : public OptionsWidget {
 public:
  BundleAdjustmentBackendOptionsWidget(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    auto* backend_combo = new QComboBox(this);
    backend_combo->addItem(
        "CERES");  // index 0 == BundleAdjustmentBackend::CERES
    backend_combo->addItem(
        "CASPAR");  // index 1 == BundleAdjustmentBackend::CASPAR
    backend_combo->setCurrentIndex(
        static_cast<int>(options->bundle_adjustment->backend));
    connect(backend_combo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            [options](int idx) {
              options->bundle_adjustment->backend =
                  static_cast<BundleAdjustmentBackend>(idx);
            });
    AddWidgetRow("backend", backend_combo);

    AddSection("Caspar Options");
    AddOptionText(&options->bundle_adjustment->caspar->gpu_index,
                  "gpu_index (-1 = auto)");

    const bool caspar_active =
        options->bundle_adjustment->backend == BundleAdjustmentBackend::CASPAR;
    if (!caspar_active) {
      HideOption(&options->bundle_adjustment->caspar->gpu_index);
    }

    connect(backend_combo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            [this, options](int idx) {
              const bool is_caspar =
                  static_cast<BundleAdjustmentBackend>(idx) ==
                  BundleAdjustmentBackend::CASPAR;
              if (is_caspar) {
                ShowOption(&options->bundle_adjustment->caspar->gpu_index);
              } else {
                HideOption(&options->bundle_adjustment->caspar->gpu_index);
              }
            });
  }
};
#endif

// Same options as in the "Priors" tab of the incremental mapper.
class BundleAdjustmentPriorsOptionsWidget : public OptionsWidget {
 public:
  BundleAdjustmentPriorsOptionsWidget(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionBool(&options->bundle_adjustment_pose_prior->use_prior_position,
                  "use_prior_position");
    AddOptionBool(&options->bundle_adjustment_pose_prior
                       ->use_robust_loss_on_prior_position,
                  "use_robust_loss_on_prior_position");
    AddOptionDouble(
        &options->bundle_adjustment_pose_prior->prior_position_loss_scale,
        "prior_position_loss_scale");
  }
};

}  // namespace

BundleAdjustmentWidget::BundleAdjustmentWidget(MainWindow* main_window,
                                               OptionManager* options)
    : OptionsWidget(main_window),
      main_window_(main_window),
      options_(options),
      reconstruction_(nullptr),
      thread_control_widget_(new ThreadControlWidget(this)) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Bundle adjustment");

  tab_widget_ = new QTabWidget(this);
  tab_widget_->setElideMode(Qt::TextElideMode::ElideRight);
  tab_widget_->addTab(new BundleAdjustmentGeneralOptionsWidget(this, options),
                      tr("General"));
#ifdef CASPAR_ENABLED
  tab_widget_->addTab(new BundleAdjustmentBackendOptionsWidget(this, options),
                      tr("Backend"));
#endif
  tab_widget_->addTab(new BundleAdjustmentPriorsOptionsWidget(this, options),
                      tr("Priors"));
  grid_layout_->addWidget(tab_widget_, grid_layout_->rowCount(), 0, 1, 2);

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
  THROW_CHECK_NOTNULL(reconstruction_);

  // The tabs write their options when hidden, but the current tab is not
  // hidden when clicking the run button. Only write the current tab, since the
  // other tabs may hold stale values, e.g., after the options were reloaded.
  static_cast<OptionsWidget*>(tab_widget_->currentWidget())->WriteOptions();

  // Validate the pose prior options here, since errors in the controller
  // thread are only logged.
  if (options_->bundle_adjustment_pose_prior->use_prior_position) {
    if (!ExistsFile(*options_->database_path)) {
      QMessageBox::critical(
          this, "", tr("Pose priors require a valid database in the project."));
      return;
    }
    if (options_->bundle_adjustment->backend ==
        BundleAdjustmentBackend::CASPAR) {
      QMessageBox::critical(
          this, "", tr("The Caspar backend does not support pose priors."));
      return;
    }
    if (!options_->bundle_adjustment_pose_prior->Check()) {
      QMessageBox::critical(this, "", tr("Invalid pose prior options."));
      return;
    }
  }

  auto thread = std::make_unique<ControllerThread<BundleAdjustmentController>>(
      std::make_shared<BundleAdjustmentController>(*options_, reconstruction_));
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
