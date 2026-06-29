// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/ui/reconstruction_options_widget.h"

#include "colmap/controllers/global_pipeline.h"
#include "colmap/controllers/hierarchical_pipeline.h"
#include "colmap/controllers/incremental_pipeline.h"

namespace colmap {
namespace {

class IncrementalMapperGeneralOptionsWidget : public OptionsWidget {
 public:
  IncrementalMapperGeneralOptionsWidget(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionBool(&options->mapper->multiple_models, "multiple_models");
    AddOptionInt(&options->mapper->max_num_models, "max_num_models");
    AddOptionInt(&options->mapper->max_model_overlap, "max_model_overlap");
    AddOptionInt(&options->mapper->min_model_size, "min_model_size");
    AddOptionBool(&options->mapper->extract_colors, "extract_colors");
    AddOptionInt(&options->mapper->num_threads, "num_threads", -1);
    AddOptionInt(&options->mapper->random_seed, "random_seed", -1);
    AddOptionInt(&options->mapper->min_num_matches, "min_num_matches");
    AddOptionBool(&options->mapper->ignore_watermarks, "ignore_watermarks");
    AddOptionDirPath(&options->mapper->snapshot_path, "snapshot_path");
    AddOptionInt(
        &options->mapper->snapshot_frames_freq, "snapshot_frames_freq", 0);
  }
};

class IncrementalMapperTriangulationOptionsWidget : public OptionsWidget {
 public:
  IncrementalMapperTriangulationOptionsWidget(QWidget* parent,
                                              OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionInt(&options->mapper->triangulation.max_transitivity,
                 "max_transitivity");
    AddOptionDouble(&options->mapper->triangulation.create_max_angle_error,
                    "create_max_angle_error [deg]");
    AddOptionDouble(&options->mapper->triangulation.continue_max_angle_error,
                    "continue_max_angle_error [deg]");
    AddOptionDouble(&options->mapper->triangulation.merge_max_reproj_error,
                    "merge_max_reproj_error [px]");
    AddOptionDouble(&options->mapper->triangulation.re_max_angle_error,
                    "re_max_angle_error [deg]");
    AddOptionDouble(&options->mapper->triangulation.re_min_ratio,
                    "re_min_ratio");
    AddOptionInt(&options->mapper->triangulation.re_max_trials,
                 "re_max_trials");
    AddOptionDouble(&options->mapper->triangulation.complete_max_reproj_error,
                    "complete_max_reproj_error [px]");
    AddOptionInt(&options->mapper->triangulation.complete_max_transitivity,
                 "complete_max_transitivity");
    AddOptionDouble(
        &options->mapper->triangulation.min_angle, "min_angle [deg]", 0, 180);
    AddOptionBool(&options->mapper->triangulation.ignore_two_view_tracks,
                  "ignore_two_view_tracks");
  }
};

class IncrementalMapperRegistrationOptionsWidget : public OptionsWidget {
 public:
  IncrementalMapperRegistrationOptionsWidget(QWidget* parent,
                                             OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionDouble(&options->mapper->mapper.abs_pose_max_error,
                    "abs_pose_max_error [px]");
    AddOptionInt(&options->mapper->mapper.abs_pose_min_num_inliers,
                 "abs_pose_min_num_inliers");
    AddOptionDouble(&options->mapper->mapper.abs_pose_min_inlier_ratio,
                    "abs_pose_min_inlier_ratio");
    AddOptionInt(&options->mapper->mapper.max_reg_trials, "max_reg_trials", 1);
    AddOptionBool(&options->mapper->structure_less_registration_fallback,
                  "structure_less_registration_fallback");
    AddOptionBool(&options->mapper->structure_less_registration_only,
                  "structure_less_registration_only");
  }
};

class IncrementalMapperInitializationOptionsWidget : public OptionsWidget {
 public:
  IncrementalMapperInitializationOptionsWidget(QWidget* parent,
                                               OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionInt(&options->mapper->init_image_id1,
                 "init_image_id1",
                 -1,
                 static_cast<int>(2e9));
    AddOptionInt(&options->mapper->init_image_id2,
                 "init_image_id2",
                 -1,
                 static_cast<int>(2e9));
    AddOptionInt(&options->mapper->init_num_trials, "init_num_trials");
    AddOptionInt(&options->mapper->mapper.init_min_num_inliers,
                 "init_min_num_inliers");
    AddOptionDouble(&options->mapper->mapper.init_max_error, "init_max_error");
    AddOptionDouble(&options->mapper->mapper.init_max_forward_motion,
                    "init_max_forward_motion");
    AddOptionDouble(&options->mapper->mapper.init_min_tri_angle,
                    "init_min_tri_angle [deg]");
    AddOptionInt(
        &options->mapper->mapper.init_max_reg_trials, "init_max_reg_trials", 1);
  }
};

class IncrementalMapperBundleAdjustmentOptionsWidget : public OptionsWidget {
 public:
  IncrementalMapperBundleAdjustmentOptionsWidget(QWidget* parent,
                                                 OptionManager* options)
      : OptionsWidget(parent) {
    AddSection("Rig/Camera parameters");
    AddOptionBool(&options->mapper->ba_refine_focal_length,
                  "refine_focal_length");
    AddOptionBool(&options->mapper->ba_refine_principal_point,
                  "refine_principal_point");
    AddOptionBool(&options->mapper->ba_refine_extra_params,
                  "refine_extra_params");
    AddOptionBool(&options->mapper->ba_refine_sensor_from_rig,
                  "refine_sensor_from_rig");

    AddSpacer();

    AddSection("Local Bundle Adjustment");
    AddOptionInt(&options->mapper->mapper.ba_local_num_images, "num_images");
    AddOptionInt(&options->mapper->ba_local_max_num_iterations,
                 "max_num_iterations");
    AddOptionInt(
        &options->mapper->ba_local_max_refinements, "max_refinements", 1);
    AddOptionDouble(&options->mapper->ba_local_max_refinement_change,
                    "max_refinement_change",
                    0,
                    1,
                    1e-6,
                    6);

#ifdef CASPAR_ENABLED
    {
      auto* backend_combo = new QComboBox(this);
      backend_combo->addItem("CERES");
      backend_combo->addItem("CASPAR");
      backend_combo->setCurrentIndex(
          static_cast<int>(options->mapper->ba_local_backend));
      connect(backend_combo,
              QOverload<int>::of(&QComboBox::currentIndexChanged),
              [options](int idx) {
                options->mapper->ba_local_backend =
                    static_cast<BundleAdjustmentBackend>(idx);
              });
      AddWidgetRow("local_backend", backend_combo);
    }
#endif

    AddSpacer();

#ifdef CASPAR_ENABLED
    AddSection("Global Bundle Adjustment Backend");
    {
      auto* backend_combo = new QComboBox(this);
      backend_combo->addItem("CERES");
      backend_combo->addItem("CASPAR");
      backend_combo->setCurrentIndex(
          static_cast<int>(options->mapper->ba_global_backend));
      connect(backend_combo,
              QOverload<int>::of(&QComboBox::currentIndexChanged),
              [options](int idx) {
                options->mapper->ba_global_backend =
                    static_cast<BundleAdjustmentBackend>(idx);
              });
      AddWidgetRow("global_backend", backend_combo);
    }
#endif

    AddSpacer();

    AddSection("Global Bundle Adjustment");
    AddOptionDouble(&options->mapper->ba_global_frames_ratio, "images_ratio");
    AddOptionInt(&options->mapper->ba_global_frames_freq, "images_freq");
    AddOptionDouble(&options->mapper->ba_global_points_ratio, "points_ratio");
    AddOptionInt(&options->mapper->ba_global_points_freq, "points_freq");
    AddOptionInt(&options->mapper->ba_global_max_num_iterations,
                 "max_num_iterations");
    AddOptionInt(
        &options->mapper->ba_global_max_refinements, "max_refinements", 1);
    AddOptionDouble(&options->mapper->ba_global_max_refinement_change,
                    "max_refinement_change",
                    0,
                    1,
                    1e-6,
                    6);
  }
};

class IncrementalMapperFilteringOptionsWidget : public OptionsWidget {
 public:
  IncrementalMapperFilteringOptionsWidget(QWidget* parent,
                                          OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionDouble(&options->mapper->min_focal_length_ratio,
                    "min_focal_length_ratio");
    AddOptionDouble(&options->mapper->max_focal_length_ratio,
                    "max_focal_length_ratio");
    AddOptionDouble(&options->mapper->max_extra_param, "max_extra_param");

    AddOptionDouble(&options->mapper->mapper.filter_max_reproj_error,
                    "filter_max_reproj_error [px]");
    AddOptionDouble(&options->mapper->mapper.filter_min_tri_angle,
                    "filter_min_tri_angle [deg]");
  }
};

class IncrementalMapperPriorsOptionsWidget : public OptionsWidget {
 public:
  IncrementalMapperPriorsOptionsWidget(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionBool(&options->mapper->use_prior_position, "use_prior_position");
    AddOptionBool(&options->mapper->use_robust_loss_on_prior_position,
                  "use_robust_loss_on_prior_position");
    AddOptionDouble(&options->mapper->prior_position_loss_scale,
                    "prior_position_loss_scale");
  }
};

// Hierarchical-specific options. The per-cluster reconstruction itself is
// configured through the shared incremental mapper options.
class HierarchicalMapperOptionsWidget : public OptionsWidget {
 public:
  HierarchicalMapperOptionsWidget(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    auto& hierarchical = *options->hierarchical_mapper;
    AddOptionInt(&hierarchical.num_threads, "num_threads", -1);
    AddOptionInt(&hierarchical.num_workers, "num_workers", -1);
    AddOptionInt(&hierarchical.init_num_trials, "init_num_trials");

    AddSpacer();

    AddSection("Clustering");
    AddOptionBool(&hierarchical.clustering_options.is_hierarchical,
                  "is_hierarchical");
    AddOptionInt(&hierarchical.clustering_options.branching, "branching");
    AddOptionInt(&hierarchical.clustering_options.image_overlap,
                 "image_overlap");
    AddOptionInt(&hierarchical.clustering_options.num_image_matches,
                 "num_image_matches");
    AddOptionInt(&hierarchical.clustering_options.leaf_max_num_images,
                 "leaf_max_num_images");
  }
};

class GlobalMapperGeneralOptionsWidget : public OptionsWidget {
 public:
  GlobalMapperGeneralOptionsWidget(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionInt(&options->global_mapper->min_num_matches, "min_num_matches");
    AddOptionBool(&options->global_mapper->ignore_watermarks,
                  "ignore_watermarks");
    AddOptionInt(&options->global_mapper->num_threads, "num_threads", -1);
    AddOptionInt(&options->global_mapper->random_seed, "random_seed", -1);
    AddOptionBool(&options->global_mapper->decompose_relative_pose,
                  "decompose_relative_pose");
    AddOptionBool(&options->global_mapper->mapper.refine_sensor_from_rig,
                  "refine_sensor_from_rig");
    AddOptionInt(&options->global_mapper->mapper.ba_num_iterations,
                 "ba_num_iterations");
    AddOptionDouble(
        &options->global_mapper->mapper.max_angular_reproj_error_deg,
        "max_angular_reproj_error [deg]");
    AddOptionDouble(&options->global_mapper->mapper.max_normalized_reproj_error,
                    "max_normalized_reproj_error",
                    0,
                    1e7,
                    1e-4,
                    6);
    AddOptionDouble(&options->global_mapper->mapper.min_tri_angle_deg,
                    "min_tri_angle [deg]",
                    0,
                    180);

    AddSpacer();

    AddSection("Stages");
    AddOptionBool(&options->global_mapper->mapper.skip_rotation_averaging,
                  "skip_rotation_averaging");
    AddOptionBool(&options->global_mapper->mapper.skip_track_establishment,
                  "skip_track_establishment");
    AddOptionBool(&options->global_mapper->mapper.skip_global_positioning,
                  "skip_global_positioning");
    AddOptionBool(&options->global_mapper->mapper.skip_bundle_adjustment,
                  "skip_bundle_adjustment");
    AddOptionBool(&options->global_mapper->mapper.skip_retriangulation,
                  "skip_retriangulation");
  }
};

class GlobalMapperTrackOptionsWidget : public OptionsWidget {
 public:
  GlobalMapperTrackOptionsWidget(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionDouble(
        &options->global_mapper->mapper.track_intra_image_consistency_threshold,
        "intra_image_consistency_threshold [px]");
    // These options default to INT_MAX ("no limit"), which is shown and edited
    // as -1 ("unlimited").
    AddOptionIntUnlimited(
        &options->global_mapper->mapper.track_required_tracks_per_view,
        "required_tracks_per_view");
    AddOptionInt(&options->global_mapper->mapper.track_min_num_views_per_track,
                 "min_num_views_per_track");
    AddOptionIntUnlimited(&options->global_mapper->mapper.keep_max_num_tracks,
                          "keep_max_num_tracks");
  }
};

class GlobalMapperRotationAveragingOptionsWidget : public OptionsWidget {
 public:
  GlobalMapperRotationAveragingOptionsWidget(QWidget* parent,
                                             OptionManager* options)
      : OptionsWidget(parent) {
    AddOptionBool(
        &options->global_mapper->mapper.rotation_averaging.use_gravity,
        "use_gravity");
    AddOptionBool(
        &options->global_mapper->mapper.rotation_averaging.use_stratified,
        "use_stratified");
    AddOptionDouble(&options->global_mapper->mapper.rotation_averaging
                         .max_rotation_error_deg,
                    "max_rotation_error [deg]");
  }
};

class GlobalMapperPositioningOptionsWidget : public OptionsWidget {
 public:
  GlobalMapperPositioningOptionsWidget(QWidget* parent, OptionManager* options)
      : OptionsWidget(parent) {
    auto& global_positioning =
        options->global_mapper->mapper.global_positioning;
    AddOptionBool(&global_positioning.use_gpu, "use_gpu");
    AddOptionText(&global_positioning.gpu_index, "gpu_index");
    AddOptionBool(&global_positioning.optimize_positions, "optimize_positions");
    AddOptionBool(&global_positioning.optimize_points, "optimize_points");
    AddOptionBool(&global_positioning.optimize_scales, "optimize_scales");
    AddOptionDouble(&global_positioning.loss_function_scale,
                    "loss_function_scale");
    AddOptionInt(&global_positioning.solver_options.max_num_iterations,
                 "max_num_iterations");
  }
};

class GlobalMapperBundleAdjustmentOptionsWidget : public OptionsWidget {
 public:
  GlobalMapperBundleAdjustmentOptionsWidget(QWidget* parent,
                                            OptionManager* options)
      : OptionsWidget(parent) {
    auto& bundle_adjustment = options->global_mapper->mapper.bundle_adjustment;
    AddOptionBool(&bundle_adjustment.refine_focal_length,
                  "refine_focal_length");
    AddOptionBool(&bundle_adjustment.refine_principal_point,
                  "refine_principal_point");
    AddOptionBool(&bundle_adjustment.refine_extra_params,
                  "refine_extra_params");
    AddOptionBool(&bundle_adjustment.refine_rig_from_world,
                  "refine_rig_from_world");
    AddOptionBool(&bundle_adjustment.refine_points3D, "refine_points3D");
    AddOptionInt(&bundle_adjustment.min_track_length, "min_track_length");
    AddOptionBool(&options->global_mapper->mapper.ba_skip_fixed_rotation_stage,
                  "skip_fixed_rotation_stage");
    AddOptionBool(
        &options->global_mapper->mapper.ba_skip_joint_optimization_stage,
        "skip_joint_optimization_stage");

    if (bundle_adjustment.ceres) {
      AddSpacer();
      AddSection("Ceres");
      AddOptionBool(&bundle_adjustment.ceres->use_gpu, "use_gpu");
      AddOptionText(&bundle_adjustment.ceres->gpu_index, "gpu_index");
      AddOptionDouble(&bundle_adjustment.ceres->loss_function_scale,
                      "loss_function_scale");
      AddOptionInt(&bundle_adjustment.ceres->solver_options.max_num_iterations,
                   "max_num_iterations");
    }
  }
};

class GlobalMapperTriangulationOptionsWidget : public OptionsWidget {
 public:
  GlobalMapperTriangulationOptionsWidget(QWidget* parent,
                                         OptionManager* options)
      : OptionsWidget(parent) {
    auto& retriangulation = options->global_mapper->mapper.retriangulation;
    AddOptionDouble(&retriangulation.complete_max_reproj_error,
                    "complete_max_reproj_error [px]");
    AddOptionDouble(&retriangulation.merge_max_reproj_error,
                    "merge_max_reproj_error [px]");
    AddOptionDouble(&retriangulation.min_angle, "min_angle [deg]", 0, 180);
  }
};

}  // namespace

ReconstructionOptionsWidget::ReconstructionOptionsWidget(
    QWidget* parent,
    OptionManager* options,
    MapperType* mapper_type,
    std::function<void()> on_mapper_type_changed)
    : QWidget(parent) {
  setWindowFlags(Qt::Dialog);
  setWindowModality(Qt::ApplicationModal);
  setWindowTitle("Reconstruction options");

  QGridLayout* grid = new QGridLayout(this);

  // Mapper selection drop-down, kept above the option tabs so it stays visible
  // regardless of which mapper's options are shown.
  auto* mapper_combo = new QComboBox(this);
  mapper_combo->addItem("incremental");
  mapper_combo->addItem("hierarchical");
  mapper_combo->addItem("global");
  mapper_combo->setCurrentIndex(static_cast<int>(*mapper_type));
  auto* mapper_layout = new QHBoxLayout();
  mapper_layout->addStretch(1);
  mapper_layout->addWidget(new QLabel(tr("mapper"), this));
  mapper_layout->addWidget(mapper_combo);
  mapper_layout->addStretch(1);
  grid->addLayout(mapper_layout, 0, 0);

  // Builds the incremental mapper option tabs. These drive both the incremental
  // mapper and, per cluster, the hierarchical mapper, so a separate set is
  // built for each page (all bound to the same shared options).
  const auto make_incremental_tabs = [this, options]() {
    QTabWidget* tabs = new QTabWidget(this);
    tabs->setElideMode(Qt::TextElideMode::ElideRight);
    tabs->addTab(new IncrementalMapperGeneralOptionsWidget(this, options),
                 tr("General"));
    tabs->addTab(
        new IncrementalMapperInitializationOptionsWidget(this, options),
        tr("Init"));
    tabs->addTab(new IncrementalMapperRegistrationOptionsWidget(this, options),
                 tr("Registration"));
    tabs->addTab(new IncrementalMapperTriangulationOptionsWidget(this, options),
                 tr("Triangulation"));
    tabs->addTab(
        new IncrementalMapperBundleAdjustmentOptionsWidget(this, options),
        tr("Bundle"));
    tabs->addTab(new IncrementalMapperFilteringOptionsWidget(this, options),
                 tr("Filter"));
    tabs->addTab(new IncrementalMapperPriorsOptionsWidget(this, options),
                 tr("Priors"));
    return tabs;
  };

  // Incremental mapper option tabs.
  QTabWidget* incremental_tabs = make_incremental_tabs();

  // Hierarchical mapper option tabs: the shared incremental tabs plus a tab for
  // the hierarchical-specific clustering and worker options.
  QTabWidget* hierarchical_tabs = make_incremental_tabs();
  hierarchical_tabs->addTab(new HierarchicalMapperOptionsWidget(this, options),
                            tr("Hierarchical"));

  // Global mapper option tabs.
  QTabWidget* global_tabs = new QTabWidget(this);
  global_tabs->setElideMode(Qt::TextElideMode::ElideRight);
  global_tabs->addTab(new GlobalMapperGeneralOptionsWidget(this, options),
                      tr("General"));
  global_tabs->addTab(new GlobalMapperTrackOptionsWidget(this, options),
                      tr("Tracks"));
  global_tabs->addTab(
      new GlobalMapperRotationAveragingOptionsWidget(this, options),
      tr("Rotation"));
  global_tabs->addTab(new GlobalMapperPositioningOptionsWidget(this, options),
                      tr("Positioning"));
  global_tabs->addTab(
      new GlobalMapperBundleAdjustmentOptionsWidget(this, options),
      tr("Bundle"));
  global_tabs->addTab(new GlobalMapperTriangulationOptionsWidget(this, options),
                      tr("Triangulation"));

  // Show the option tabs for the selected mapper. The pages are added in the
  // same order as the MapperType enum (and the combo items above), so the stack
  // index matches the enum value.
  static_assert(static_cast<int>(MapperType::INCREMENTAL) == 0);
  static_assert(static_cast<int>(MapperType::HIERARCHICAL) == 1);
  static_assert(static_cast<int>(MapperType::GLOBAL) == 2);
  auto* stack = new QStackedWidget(this);
  stack->addWidget(incremental_tabs);
  stack->addWidget(hierarchical_tabs);
  stack->addWidget(global_tabs);
  stack->setCurrentIndex(static_cast<int>(*mapper_type));
  grid->addWidget(stack, 1, 0);

  connect(
      mapper_combo,
      QOverload<int>::of(&QComboBox::currentIndexChanged),
      [mapper_type,
       stack,
       on_mapper_type_changed = std::move(on_mapper_type_changed)](int idx) {
        *mapper_type = static_cast<MapperType>(idx);
        stack->setCurrentIndex(idx);
        if (on_mapper_type_changed) {
          on_mapper_type_changed();
        }
      });
}

}  // namespace colmap
