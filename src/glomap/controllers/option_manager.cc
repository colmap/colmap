#include "glomap/controllers/option_manager.h"

#include "glomap/estimators/global_positioning.h"
#include "glomap/estimators/gravity_refinement.h"
#include "glomap/sfm/global_mapper.h"

namespace glomap {

OptionManager::OptionManager(bool add_project_options)
    : colmap::BaseOptionManager(add_project_options) {
  mapper = std::make_shared<GlobalMapperOptions>();
  gravity_refiner = std::make_shared<GravityRefinerOptions>();
}

void OptionManager::AddAllOptions() {
  colmap::BaseOptionManager::AddAllOptions();
  AddGlobalMapperOptions();
  AddGravityRefinerOptions();
}

void OptionManager::AddGlobalMapperOptions() {
  if (added_global_mapper_options_) {
    return;
  }
  added_global_mapper_options_ = true;

  // Global mapper options
  AddAndRegisterDefaultOption("Mapper.num_threads", &mapper->num_threads);
  AddAndRegisterDefaultOption("Mapper.random_seed", &mapper->random_seed);
  AddAndRegisterDefaultOption("Mapper.num_iterations_ba",
                              &mapper->num_iterations_ba);
  AddAndRegisterDefaultOption("Mapper.skip_view_graph_calibration",
                              &mapper->skip_view_graph_calibration);
  AddAndRegisterDefaultOption("Mapper.skip_rotation_averaging",
                              &mapper->skip_rotation_averaging);
  AddAndRegisterDefaultOption("Mapper.skip_global_positioning",
                              &mapper->skip_global_positioning);
  AddAndRegisterDefaultOption("Mapper.skip_bundle_adjustment",
                              &mapper->skip_bundle_adjustment);
  AddAndRegisterDefaultOption("Mapper.skip_retriangulation",
                              &mapper->skip_retriangulation);
  AddAndRegisterDefaultOption("Mapper.skip_pruning", &mapper->skip_pruning);

  // Rotation averaging options
  AddAndRegisterDefaultOption(
      "RotationAveraging.max_rotation_error_deg",
      &mapper->rotation_averaging.max_rotation_error_deg);

  // View graph calibration options
  AddAndRegisterDefaultOption(
      "ViewGraphCalib.min_focal_length_ratio",
      &mapper->view_graph_calibration.min_focal_length_ratio);
  AddAndRegisterDefaultOption(
      "ViewGraphCalib.max_focal_length_ratio",
      &mapper->view_graph_calibration.max_focal_length_ratio);
  AddAndRegisterDefaultOption(
      "ViewGraphCalib.max_calibration_error",
      &mapper->view_graph_calibration.max_calibration_error);

  // Track establishment options
  AddAndRegisterDefaultOption(
      "TrackEstablishment.min_num_tracks_per_view",
      &mapper->track_establishment.min_num_tracks_per_view);
  AddAndRegisterDefaultOption(
      "TrackEstablishment.min_num_view_per_track",
      &mapper->track_establishment.min_num_view_per_track);
  AddAndRegisterDefaultOption(
      "TrackEstablishment.max_num_view_per_track",
      &mapper->track_establishment.max_num_view_per_track);
  AddAndRegisterDefaultOption("TrackEstablishment.max_num_tracks",
                              &mapper->track_establishment.max_num_tracks);

  // Global positioning options
  AddAndRegisterDefaultOption("GlobalPositioning.use_gpu",
                              &mapper->global_positioning.use_gpu);
  AddAndRegisterDefaultOption("GlobalPositioning.gpu_index",
                              &mapper->global_positioning.gpu_index);
  AddAndRegisterDefaultOption("GlobalPositioning.optimize_positions",
                              &mapper->global_positioning.optimize_positions);
  AddAndRegisterDefaultOption("GlobalPositioning.optimize_points",
                              &mapper->global_positioning.optimize_points);
  AddAndRegisterDefaultOption("GlobalPositioning.optimize_scales",
                              &mapper->global_positioning.optimize_scales);
  AddAndRegisterDefaultOption("GlobalPositioning.loss_function_scale",
                              &mapper->global_positioning.loss_function_scale);
  AddAndRegisterDefaultOption(
      "GlobalPositioning.max_num_iterations",
      &mapper->global_positioning.solver_options.max_num_iterations);
  AddAndRegisterDefaultEnumOption(
      "GlobalPositioning.constraint_type",
      &mapper->global_positioning.constraint_type,
      GlobalPositioningConstraintTypeToString,
      GlobalPositioningConstraintTypeFromString,
      "{ONLY_POINTS, ONLY_CAMERAS, POINTS_AND_CAMERAS_BALANCED, "
      "POINTS_AND_CAMERAS}");

  // Bundle adjustment options
  AddAndRegisterDefaultOption("BundleAdjustment.use_gpu",
                              &mapper->bundle_adjustment.use_gpu);
  AddAndRegisterDefaultOption("BundleAdjustment.gpu_index",
                              &mapper->bundle_adjustment.gpu_index);
  AddAndRegisterDefaultOption("BundleAdjustment.optimize_rig_poses",
                              &mapper->bundle_adjustment.optimize_rig_poses);
  AddAndRegisterDefaultOption("BundleAdjustment.optimize_rotations",
                              &mapper->bundle_adjustment.optimize_rotations);
  AddAndRegisterDefaultOption("BundleAdjustment.optimize_translation",
                              &mapper->bundle_adjustment.optimize_translation);
  AddAndRegisterDefaultOption("BundleAdjustment.optimize_intrinsics",
                              &mapper->bundle_adjustment.optimize_intrinsics);
  AddAndRegisterDefaultOption(
      "BundleAdjustment.optimize_principal_point",
      &mapper->bundle_adjustment.optimize_principal_point);
  AddAndRegisterDefaultOption("BundleAdjustment.optimize_points",
                              &mapper->bundle_adjustment.optimize_points);
  AddAndRegisterDefaultOption("BundleAdjustment.loss_function_scale",
                              &mapper->bundle_adjustment.loss_function_scale);
  AddAndRegisterDefaultOption(
      "BundleAdjustment.max_num_iterations",
      &mapper->bundle_adjustment.solver_options.max_num_iterations);

  // Retriangulation options
  AddAndRegisterDefaultOption(
      "Retriangulation.complete_max_reproj_error",
      &mapper->retriangulation.complete_max_reproj_error);
  AddAndRegisterDefaultOption("Retriangulation.merge_max_reproj_error",
                              &mapper->retriangulation.merge_max_reproj_error);
  AddAndRegisterDefaultOption("Retriangulation.min_angle",
                              &mapper->retriangulation.min_angle);

  // Threshold options
  AddAndRegisterDefaultOption("Mapper.max_angular_reproj_error_deg",
                              &mapper->max_angular_reproj_error_deg);
  AddAndRegisterDefaultOption("Mapper.max_normalized_reproj_error",
                              &mapper->max_normalized_reproj_error);
  AddAndRegisterDefaultOption("Mapper.min_tri_angle_deg",
                              &mapper->min_tri_angle_deg);
}

void OptionManager::AddGravityRefinerOptions() {
  if (added_gravity_refiner_options_) {
    return;
  }
  added_gravity_refiner_options_ = true;

  AddAndRegisterDefaultOption("GravityRefiner.max_outlier_ratio",
                              &gravity_refiner->max_outlier_ratio);
  AddAndRegisterDefaultOption("GravityRefiner.max_gravity_error",
                              &gravity_refiner->max_gravity_error);
  AddAndRegisterDefaultOption("GravityRefiner.min_num_neighbors",
                              &gravity_refiner->min_num_neighbors);
}

void OptionManager::Reset() {
  colmap::BaseOptionManager::Reset();

  added_global_mapper_options_ = false;
  added_gravity_refiner_options_ = false;
}

void OptionManager::ResetOptions(const bool reset_paths) {
  colmap::BaseOptionManager::ResetOptions(reset_paths);
  *mapper = GlobalMapperOptions();
  *gravity_refiner = GravityRefinerOptions();
}

}  // namespace glomap
