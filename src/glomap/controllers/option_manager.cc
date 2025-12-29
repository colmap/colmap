#include "glomap/controllers/option_manager.h"

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
  AddAndRegisterDefaultOption("Mapper.num_iterations_retriangulation",
                              &mapper->num_iterations_retriangulation);
  AddAndRegisterDefaultOption("Mapper.skip_preprocessing",
                              &mapper->skip_preprocessing);
  AddAndRegisterDefaultOption("Mapper.skip_view_graph_calibration",
                              &mapper->skip_view_graph_calibration);
  AddAndRegisterDefaultOption("Mapper.skip_relative_pose_estimation",
                              &mapper->skip_relative_pose_estimation);
  AddAndRegisterDefaultOption("Mapper.skip_rotation_averaging",
                              &mapper->skip_rotation_averaging);
  AddAndRegisterDefaultOption("Mapper.skip_global_positioning",
                              &mapper->skip_global_positioning);
  AddAndRegisterDefaultOption("Mapper.skip_bundle_adjustment",
                              &mapper->skip_bundle_adjustment);
  AddAndRegisterDefaultOption("Mapper.skip_retriangulation",
                              &mapper->skip_retriangulation);
  AddAndRegisterDefaultOption("Mapper.skip_pruning", &mapper->skip_pruning);

  // View graph calibration options
  AddAndRegisterDefaultOption(
      "ViewGraphCalib.thres_lower_ratio",
      &mapper->view_graph_calibration.thres_lower_ratio);
  AddAndRegisterDefaultOption(
      "ViewGraphCalib.thres_higher_ratio",
      &mapper->view_graph_calibration.thres_higher_ratio);
  AddAndRegisterDefaultOption(
      "ViewGraphCalib.thres_two_view_error",
      &mapper->view_graph_calibration.thres_two_view_error);

  // Relative pose estimation options
  AddAndRegisterDefaultOption(
      "RelPoseEstimation.max_epipolar_error",
      &mapper->relative_pose_estimation.ransac_options.max_epipolar_error);

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

  // Triangulation options
  AddAndRegisterDefaultOption(
      "Triangulation.complete_max_reproj_error",
      &mapper->retriangulation.tri_complete_max_reproj_error);
  AddAndRegisterDefaultOption(
      "Triangulation.merge_max_reproj_error",
      &mapper->retriangulation.tri_merge_max_reproj_error);
  AddAndRegisterDefaultOption("Triangulation.min_angle",
                              &mapper->retriangulation.tri_min_angle);
  AddAndRegisterDefaultOption("Triangulation.min_num_matches",
                              &mapper->retriangulation.min_num_matches);

  // Inlier threshold options
  AddAndRegisterDefaultOption("Thresholds.max_angle_error",
                              &mapper->inlier_thresholds.max_angle_error);
  AddAndRegisterDefaultOption(
      "Thresholds.max_reprojection_error",
      &mapper->inlier_thresholds.max_reprojection_error);
  AddAndRegisterDefaultOption(
      "Thresholds.min_triangulation_angle",
      &mapper->inlier_thresholds.min_triangulation_angle);
  AddAndRegisterDefaultOption("Thresholds.max_epipolar_error_E",
                              &mapper->inlier_thresholds.max_epipolar_error_E);
  AddAndRegisterDefaultOption("Thresholds.max_epipolar_error_F",
                              &mapper->inlier_thresholds.max_epipolar_error_F);
  AddAndRegisterDefaultOption("Thresholds.max_epipolar_error_H",
                              &mapper->inlier_thresholds.max_epipolar_error_H);
  AddAndRegisterDefaultOption("Thresholds.min_inlier_num",
                              &mapper->inlier_thresholds.min_inlier_num);
  AddAndRegisterDefaultOption("Thresholds.min_inlier_ratio",
                              &mapper->inlier_thresholds.min_inlier_ratio);
  AddAndRegisterDefaultOption("Thresholds.max_rotation_error",
                              &mapper->inlier_thresholds.max_rotation_error);
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

bool OptionManager::Check() { return colmap::BaseOptionManager::Check(); }

void OptionManager::ResetOptions(const bool reset_paths) {
  colmap::BaseOptionManager::ResetOptions(reset_paths);
  *mapper = GlobalMapperOptions();
  *gravity_refiner = GravityRefinerOptions();
}

}  // namespace glomap
