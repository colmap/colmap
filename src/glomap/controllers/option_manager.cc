#include "glomap/controllers/option_manager.h"

#include "glomap/estimators/gravity_refinement.h"
#include "glomap/sfm/global_mapper.h"

#include <boost/property_tree/ini_parser.hpp>

namespace config = boost::program_options;

namespace glomap {

OptionManager::OptionManager(bool add_project_options) {
  database_path = std::make_shared<std::string>();
  image_path = std::make_shared<std::string>();

  mapper = std::make_shared<GlobalMapperOptions>();
  gravity_refiner = std::make_shared<GravityRefinerOptions>();
  Reset();

  desc_->add_options()("help,h", "");

  AddAndRegisterDefaultOption("log_to_stderr", &FLAGS_logtostderr);
  AddAndRegisterDefaultOption("log_level", &FLAGS_v);
}

void OptionManager::AddAllOptions() {
  AddDatabaseOptions();
  AddImageOptions();
  AddGlobalMapperOptions();
  AddInlierThresholdOptions();
  AddViewGraphCalibrationOptions();
  AddRelativePoseEstimationOptions();
  AddRotationEstimatorOptions();
  AddTrackEstablishmentOptions();
  AddGlobalPositionerOptions();
  AddBundleAdjusterOptions();
  AddTriangulatorOptions();
}

void OptionManager::AddDatabaseOptions() {
  if (added_database_options_) {
    return;
  }
  added_database_options_ = true;

  AddAndRegisterRequiredOption("database_path", database_path.get());
}

void OptionManager::AddImageOptions() {
  if (added_image_options_) {
    return;
  }
  added_image_options_ = true;

  AddAndRegisterRequiredOption("image_path", image_path.get());
}

void OptionManager::AddGlobalMapperOptions() {
  if (added_mapper_options_) {
    return;
  }
  added_mapper_options_ = true;

  AddAndRegisterDefaultOption("ba_iteration_num",
                              &mapper->num_iteration_bundle_adjustment);
  AddAndRegisterDefaultOption("retriangulation_iteration_num",
                              &mapper->num_iteration_retriangulation);
  AddAndRegisterDefaultOption("skip_preprocessing",
                              &mapper->skip_preprocessing);
  AddAndRegisterDefaultOption("skip_view_graph_calibration",
                              &mapper->skip_view_graph_calibration);
  AddAndRegisterDefaultOption("skip_relative_pose_estimation",
                              &mapper->skip_relative_pose_estimation);
  AddAndRegisterDefaultOption("skip_rotation_averaging",
                              &mapper->skip_rotation_averaging);
  AddAndRegisterDefaultOption("skip_global_positioning",
                              &mapper->skip_global_positioning);
  AddAndRegisterDefaultOption("skip_bundle_adjustment",
                              &mapper->skip_bundle_adjustment);
  AddAndRegisterDefaultOption("skip_retriangulation",
                              &mapper->skip_retriangulation);
  AddAndRegisterDefaultOption("skip_pruning", &mapper->skip_pruning);
}

void OptionManager::AddGlobalMapperFullOptions() {
  AddGlobalMapperOptions();

  AddViewGraphCalibrationOptions();
  AddRelativePoseEstimationOptions();
  AddRotationEstimatorOptions();
  AddTrackEstablishmentOptions();
  AddGlobalPositionerOptions();
  AddBundleAdjusterOptions();
  AddTriangulatorOptions();
  AddInlierThresholdOptions();
}

void OptionManager::AddViewGraphCalibrationOptions() {
  if (added_view_graph_calibration_options_) {
    return;
  }
  added_view_graph_calibration_options_ = true;
  AddAndRegisterDefaultOption("ViewGraphCalib.thres_lower_ratio",
                              &mapper->opt_vgcalib.thres_lower_ratio);
  AddAndRegisterDefaultOption("ViewGraphCalib.thres_higher_ratio",
                              &mapper->opt_vgcalib.thres_higher_ratio);
  AddAndRegisterDefaultOption("ViewGraphCalib.thres_two_view_error",
                              &mapper->opt_vgcalib.thres_two_view_error);
}

void OptionManager::AddRelativePoseEstimationOptions() {
  if (added_relative_pose_options_) {
    return;
  }
  added_relative_pose_options_ = true;
  AddAndRegisterDefaultOption(
      "RelPoseEstimation.max_epipolar_error",
      &mapper->opt_relpose.ransac_options.max_epipolar_error);
}

void OptionManager::AddRotationEstimatorOptions() {
  if (added_rotation_averaging_options_) {
    return;
  }
  added_rotation_averaging_options_ = true;
  // TODO: maybe add options for rotation averaging
}

void OptionManager::AddTrackEstablishmentOptions() {
  if (added_track_establishment_options_) {
    return;
  }
  added_track_establishment_options_ = true;
  AddAndRegisterDefaultOption("TrackEstablishment.min_num_tracks_per_view",
                              &mapper->opt_track.min_num_tracks_per_view);
  AddAndRegisterDefaultOption("TrackEstablishment.min_num_view_per_track",
                              &mapper->opt_track.min_num_view_per_track);
  AddAndRegisterDefaultOption("TrackEstablishment.max_num_view_per_track",
                              &mapper->opt_track.max_num_view_per_track);
  AddAndRegisterDefaultOption("TrackEstablishment.max_num_tracks",
                              &mapper->opt_track.max_num_tracks);
}

void OptionManager::AddGlobalPositionerOptions() {
  if (added_global_positioning_options_) {
    return;
  }
  added_global_positioning_options_ = true;
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

  // TODO: move the constrain type selection here
}
void OptionManager::AddBundleAdjusterOptions() {
  if (added_bundle_adjustment_options_) {
    return;
  }
  added_bundle_adjustment_options_ = true;
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
}
void OptionManager::AddTriangulatorOptions() {
  if (added_triangulation_options_) {
    return;
  }
  added_triangulation_options_ = true;
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
}
void OptionManager::AddInlierThresholdOptions() {
  if (added_inliers_options_) {
    return;
  }
  added_inliers_options_ = true;
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
  FLAGS_logtostderr = true;

  const bool kResetPaths = true;
  ResetOptions(kResetPaths);

  desc_ = std::make_shared<boost::program_options::options_description>();

  options_bool_.clear();
  options_int_.clear();
  options_double_.clear();
  options_string_.clear();

  added_mapper_options_ = false;
  added_view_graph_calibration_options_ = false;
  added_relative_pose_options_ = false;
  added_rotation_averaging_options_ = false;
  added_track_establishment_options_ = false;
  added_global_positioning_options_ = false;
  added_bundle_adjustment_options_ = false;
  added_triangulation_options_ = false;
  added_inliers_options_ = false;
}

void OptionManager::ResetOptions(const bool reset_paths) {
  if (reset_paths) {
    *database_path = "";
    *image_path = "";
  }
  *mapper = GlobalMapperOptions();
  *gravity_refiner = GravityRefinerOptions();
}

void OptionManager::Parse(const int argc, char** argv) {
  config::variables_map vmap;

  try {
    config::store(config::parse_command_line(argc, argv, *desc_), vmap);

    if (vmap.count("help")) {
      LOG(INFO) << "The following options can be specified via command-line:\n"
                << *desc_;
      // NOLINTNEXTLINE(concurrency-mt-unsafe)
      exit(EXIT_SUCCESS);
    }

    vmap.notify();

  } catch (std::exception& exc) {
    LOG(ERROR) << "Failed to parse options - " << exc.what() << ".";
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    exit(EXIT_FAILURE);
  } catch (...) {
    LOG(ERROR) << "Failed to parse options for unknown reason.";
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    exit(EXIT_FAILURE);
  }
}

}  // namespace glomap
