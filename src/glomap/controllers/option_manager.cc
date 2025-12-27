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
  AddAndRegisterDefaultOption("ViewGraphCalib.thres_lower_ratio",
                              &mapper->opt_vgcalib.thres_lower_ratio);
  AddAndRegisterDefaultOption("ViewGraphCalib.thres_higher_ratio",
                              &mapper->opt_vgcalib.thres_higher_ratio);
  AddAndRegisterDefaultOption("ViewGraphCalib.thres_two_view_error",
                              &mapper->opt_vgcalib.thres_two_view_error);

  // Relative pose estimation options
  AddAndRegisterDefaultOption(
      "RelPoseEstimation.max_epipolar_error",
      &mapper->opt_relpose.ransac_options.max_epipolar_error);

  // Track establishment options
  AddAndRegisterDefaultOption("TrackEstablishment.min_num_tracks_per_view",
                              &mapper->opt_track.min_num_tracks_per_view);
  AddAndRegisterDefaultOption("TrackEstablishment.min_num_view_per_track",
                              &mapper->opt_track.min_num_view_per_track);
  AddAndRegisterDefaultOption("TrackEstablishment.max_num_view_per_track",
                              &mapper->opt_track.max_num_view_per_track);
  AddAndRegisterDefaultOption("TrackEstablishment.max_num_tracks",
                              &mapper->opt_track.max_num_tracks);

  // Global positioning options
  AddAndRegisterDefaultOption("GlobalPositioning.use_gpu",
                              &mapper->opt_gp.use_gpu);
  AddAndRegisterDefaultOption("GlobalPositioning.gpu_index",
                              &mapper->opt_gp.gpu_index);
  AddAndRegisterDefaultOption("GlobalPositioning.optimize_positions",
                              &mapper->opt_gp.optimize_positions);
  AddAndRegisterDefaultOption("GlobalPositioning.optimize_points",
                              &mapper->opt_gp.optimize_points);
  AddAndRegisterDefaultOption("GlobalPositioning.optimize_scales",
                              &mapper->opt_gp.optimize_scales);
  AddAndRegisterDefaultOption("GlobalPositioning.thres_loss_function",
                              &mapper->opt_gp.thres_loss_function);
  AddAndRegisterDefaultOption(
      "GlobalPositioning.max_num_iterations",
      &mapper->opt_gp.solver_options.max_num_iterations);

  // Bundle adjustment options
  AddAndRegisterDefaultOption("BundleAdjustment.use_gpu",
                              &mapper->opt_ba.use_gpu);
  AddAndRegisterDefaultOption("BundleAdjustment.gpu_index",
                              &mapper->opt_ba.gpu_index);
  AddAndRegisterDefaultOption("BundleAdjustment.optimize_rig_poses",
                              &mapper->opt_ba.optimize_rig_poses);
  AddAndRegisterDefaultOption("BundleAdjustment.optimize_rotations",
                              &mapper->opt_ba.optimize_rotations);
  AddAndRegisterDefaultOption("BundleAdjustment.optimize_translation",
                              &mapper->opt_ba.optimize_translation);
  AddAndRegisterDefaultOption("BundleAdjustment.optimize_intrinsics",
                              &mapper->opt_ba.optimize_intrinsics);
  AddAndRegisterDefaultOption("BundleAdjustment.optimize_principal_point",
                              &mapper->opt_ba.optimize_principal_point);
  AddAndRegisterDefaultOption("BundleAdjustment.optimize_points",
                              &mapper->opt_ba.optimize_points);
  AddAndRegisterDefaultOption("BundleAdjustment.thres_loss_function",
                              &mapper->opt_ba.thres_loss_function);
  AddAndRegisterDefaultOption(
      "BundleAdjustment.max_num_iterations",
      &mapper->opt_ba.solver_options.max_num_iterations);

  // Triangulation options
  AddAndRegisterDefaultOption(
      "Triangulation.complete_max_reproj_error",
      &mapper->opt_triangulator.tri_complete_max_reproj_error);
  AddAndRegisterDefaultOption(
      "Triangulation.merge_max_reproj_error",
      &mapper->opt_triangulator.tri_merge_max_reproj_error);
  AddAndRegisterDefaultOption("Triangulation.min_angle",
                              &mapper->opt_triangulator.tri_min_angle);
  AddAndRegisterDefaultOption("Triangulation.min_num_matches",
                              &mapper->opt_triangulator.min_num_matches);

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
