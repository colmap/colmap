#include "glomap/estimators/bundle_adjustment.h"

#include "colmap/util/logging.h"

namespace glomap {

colmap::BundleAdjustmentOptions BundleAdjusterOptions::ToColmapOptions() const {
  colmap::BundleAdjustmentOptions options;
  options.loss_function_type =
      colmap::BundleAdjustmentOptions::LossFunctionType::HUBER;
  options.loss_function_scale = loss_function_scale;
  options.refine_focal_length = optimize_intrinsics;
  options.refine_principal_point = optimize_principal_point;
  options.refine_extra_params = optimize_intrinsics;
  options.refine_points3D = optimize_points;
  options.min_track_length = min_num_view_per_track;
  options.refine_sensor_from_rig = optimize_rig_poses;
  options.refine_rig_from_world = optimize_translation;
  options.constant_rig_from_world_rotation = !optimize_rotations;
  options.use_gpu = use_gpu;
  options.gpu_index = gpu_index;
  options.min_num_images_gpu_solver = min_num_images_gpu_solver;
  // Disable auto-selection to use explicit solver settings from glomap
  options.auto_select_solver_type = false;
  options.solver_options = solver_options;
  return options;
}

bool RunBundleAdjustment(const BundleAdjusterOptions& options,
                         bool constant_rotation,
                         colmap::Reconstruction& reconstruction) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Number of images = " << reconstruction.NumImages();
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Number of tracks = " << reconstruction.NumPoints3D();
    return false;
  }

  // Convert to colmap options
  colmap::BundleAdjustmentOptions ba_options = options.ToColmapOptions();
  ba_options.constant_rig_from_world_rotation = constant_rotation;
  ba_options.print_summary = false;

  // Add all images with valid poses.
  colmap::BundleAdjustmentConfig ba_config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (image.HasPose()) {
      ba_config.AddImage(image_id);
    }
  }
  // Use TWO_CAMS_FROM_WORLD for deterministic gauge fixing.
  ba_config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  auto ba = colmap::CreateDefaultBundleAdjuster(
      ba_options, ba_config, reconstruction);

  ceres::Solver::Summary summary = ba->Solve();

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  return summary.IsSolutionUsable();
}

}  // namespace glomap
