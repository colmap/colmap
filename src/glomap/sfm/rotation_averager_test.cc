#include "glomap/sfm/rotation_averager.h"

#include "colmap/math/random.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include "glomap/estimators/gravity_refinement.h"
#include "glomap/io/colmap_converter.h"
#include "glomap/math/gravity.h"
#include "glomap/sfm/global_mapper.h"

#include <gtest/gtest.h>

namespace glomap {
namespace {

void SynthesizeGravityOutliers(std::vector<colmap::PosePrior>& pose_priors,
                               double outlier_ratio = 0.0) {
  for (auto& pose_prior : pose_priors) {
    if (pose_prior.HasGravity() &&
        colmap::RandomUniformReal<double>(0, 1) < outlier_ratio) {
      pose_prior.gravity = Eigen::Vector3d::Random().normalized();
    }
  }
}

GlobalMapperOptions CreateMapperTestOptions() {
  GlobalMapperOptions options;
  options.skip_view_graph_calibration = false;
  options.skip_relative_pose_estimation = false;
  options.skip_rotation_averaging = true;
  options.skip_track_establishment = true;
  options.skip_global_positioning = true;
  options.skip_bundle_adjustment = true;
  options.skip_retriangulation = true;
  return options;
}

RotationAveragerOptions CreateRATestOptions(bool use_gravity = false) {
  RotationAveragerOptions options;
  options.skip_initialization = false;
  options.use_gravity = use_gravity;
  options.use_stratified = true;
  return options;
}

void ExpectEqualRotations(const colmap::Reconstruction& gt,
                          const colmap::Reconstruction& computed,
                          const double max_rotation_error_deg) {
  const double max_rotation_error_rad =
      colmap::DegToRad(max_rotation_error_deg);
  const std::vector<image_t> reg_image_ids = gt.RegImageIds();
  for (size_t i = 0; i < reg_image_ids.size(); i++) {
    const image_t image_id1 = reg_image_ids[i];
    for (size_t j = 0; j < i; j++) {
      const image_t image_id2 = reg_image_ids[j];
      const Eigen::Quaterniond cam2_from_cam1 =
          computed.Image(image_id2).CamFromWorld().rotation *
          computed.Image(image_id1).CamFromWorld().rotation.inverse();
      const Eigen::Quaterniond cam2_from_cam1_gt =
          gt.Image(image_id2).CamFromWorld().rotation *
          gt.Image(image_id1).CamFromWorld().rotation.inverse();
      EXPECT_LT(cam2_from_cam1.angularDistance(cam2_from_cam1_gt),
                max_rotation_error_rad);
    }
  }
}

void ExpectEqualGravity(const Eigen::Vector3d& gravity_in_world,
                        const colmap::Reconstruction& gt,
                        const std::vector<colmap::PosePrior>& pose_priors,
                        const double max_gravity_error_deg) {
  std::unordered_map<image_t, const colmap::PosePrior*> image_to_pose_prior;
  for (const auto& pose_prior : pose_priors) {
    if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
      image_to_pose_prior.emplace(pose_prior.corr_data_id.id, &pose_prior);
    }
  }
  for (const auto& image_id : gt.RegImageIds()) {
    const auto& image = gt.Image(image_id);
    if (!image.IsRefInFrame()) {
      continue;
    }
    const Eigen::Vector3d gravity_gt =
        gt.Image(image_id).CamFromWorld().rotation * gravity_in_world;
    const Eigen::Vector3d gravity_computed =
        image_to_pose_prior.at(image_id)->gravity;
    const double gravity_error_deg = CalcAngle(gravity_gt, gravity_computed);
    EXPECT_LT(gravity_error_deg, max_gravity_error_deg);
  }
}

TEST(RotationEstimator, WithoutNoise) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<point3D_t, Point3D> tracks;
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(*database,
                      view_graph,
                      rigs,
                      cameras,
                      frames,
                      images,
                      tracks,
                      pose_priors);

  // TODO: The current 1-dof rotation averaging sometimes fails to pick the
  // right solution (e.g., 180 deg flipped).
  for (const bool use_gravity : {false}) {
    SolveRotationAveraging(view_graph,
                           rigs,
                           frames,
                           images,
                           pose_priors,
                           CreateRATestOptions(use_gravity));

    colmap::Reconstruction reconstruction;
    ConvertGlomapToColmap(
        rigs, cameras, frames, images, tracks, reconstruction);
    ExpectEqualRotations(
        gt_reconstruction, reconstruction, /*max_rotation_error_deg=*/1e-2);
  }
}

TEST(RotationEstimator, WithoutNoiseWithNonTrivialKnownRig) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<point3D_t, Point3D> tracks;
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(*database,
                      view_graph,
                      rigs,
                      cameras,
                      frames,
                      images,
                      tracks,
                      pose_priors);

  for (const bool use_gravity : {true, false}) {
    SolveRotationAveraging(view_graph,
                           rigs,
                           frames,
                           images,
                           pose_priors,
                           CreateRATestOptions(use_gravity));

    colmap::Reconstruction reconstruction;
    ConvertGlomapToColmap(
        rigs, cameras, frames, images, tracks, reconstruction);
    ExpectEqualRotations(
        gt_reconstruction, reconstruction, /*max_rotation_error_deg=*/1e-2);
  }
}

TEST(RotationEstimator, WithoutNoiseWithNonTrivialUnknownRig) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  synthetic_dataset_options.prior_gravity = true;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<point3D_t, Point3D> tracks;
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  for (auto& [rig_id, rig] : rigs) {
    for (auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        rig.ResetSensorFromRig(sensor_id);
      }
    }
  }

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(*database,
                      view_graph,
                      rigs,
                      cameras,
                      frames,
                      images,
                      tracks,
                      pose_priors);

  // For unknown rigs, it is not supported to use gravity.
  for (const bool use_gravity : {false}) {
    SolveRotationAveraging(view_graph,
                           rigs,
                           frames,
                           images,
                           pose_priors,
                           CreateRATestOptions(use_gravity));

    colmap::Reconstruction reconstruction;
    ConvertGlomapToColmap(
        rigs, cameras, frames, images, tracks, reconstruction);
    ExpectEqualRotations(
        gt_reconstruction, reconstruction, /*max_rotation_error_deg=*/1e-2);
  }
}

TEST(RotationEstimator, WithNoiseAndOutliers) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.prior_gravity = true;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  colmap::SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  synthetic_noise_options.prior_gravity_stddev = 3e-1;
  colmap::SynthesizeNoise(
      synthetic_noise_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<point3D_t, Point3D> tracks;
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();
  SynthesizeGravityOutliers(pose_priors, /*outlier_ratio=*/0.3);

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(*database,
                      view_graph,
                      rigs,
                      cameras,
                      frames,
                      images,
                      tracks,
                      pose_priors);

  // TODO: The current 1-dof rotation averaging sometimes fails to pick the
  // right solution (e.g., 180 deg flipped).
  for (const bool use_gravity : {false}) {
    SolveRotationAveraging(view_graph,
                           rigs,
                           frames,
                           images,
                           pose_priors,
                           CreateRATestOptions(use_gravity));

    colmap::Reconstruction reconstruction;
    ConvertGlomapToColmap(
        rigs, cameras, frames, images, tracks, reconstruction);
    ExpectEqualRotations(
        gt_reconstruction, reconstruction, /*max_rotation_error_deg=*/3);
  }
}

TEST(RotationEstimator, WithNoiseAndOutliersWithNonTrivialKnownRigs) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.prior_gravity = true;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  colmap::SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  synthetic_noise_options.prior_gravity_stddev = 3e-1;
  colmap::SynthesizeNoise(
      synthetic_noise_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<point3D_t, Point3D> tracks;
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();
  SynthesizeGravityOutliers(pose_priors, /*outlier_ratio=*/0.3);

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(*database,
                      view_graph,
                      rigs,
                      cameras,
                      frames,
                      images,
                      tracks,
                      pose_priors);

  // TODO: The current 1-dof rotation averaging sometimes fails to pick the
  // right solution (e.g., 180 deg flipped).
  for (const bool use_gravity : {false}) {
    SolveRotationAveraging(view_graph,
                           rigs,
                           frames,
                           images,
                           pose_priors,
                           CreateRATestOptions(use_gravity));

    colmap::Reconstruction reconstruction;
    ConvertGlomapToColmap(
        rigs, cameras, frames, images, tracks, reconstruction);
    ExpectEqualRotations(
        gt_reconstruction, reconstruction, /*max_rotation_error_deg=*/2.);
  }
}

TEST(RotationEstimator, RefineGravity) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 25;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_gravity = true;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<point3D_t, Point3D> tracks;
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();
  SynthesizeGravityOutliers(pose_priors, /*outlier_ratio=*/0.3);

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(*database,
                      view_graph,
                      rigs,
                      cameras,
                      frames,
                      images,
                      tracks,
                      pose_priors);

  GravityRefinerOptions opt_grav_refine;
  GravityRefiner grav_refiner(opt_grav_refine);
  grav_refiner.RefineGravity(view_graph, frames, images, pose_priors);

  ExpectEqualGravity(synthetic_dataset_options.prior_gravity_in_world,
                     gt_reconstruction,
                     pose_priors,
                     /*max_gravity_error_deg=*/1e-2);
}

TEST(RotationEstimator, RefineGravityWithNonTrivialRigs) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 25;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_gravity = true;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<point3D_t, Point3D> tracks;
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();
  SynthesizeGravityOutliers(pose_priors, /*outlier_ratio=*/0.3);

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(*database,
                      view_graph,
                      rigs,
                      cameras,
                      frames,
                      images,
                      tracks,
                      pose_priors);

  GravityRefinerOptions opt_grav_refine;
  GravityRefiner grav_refiner(opt_grav_refine);
  grav_refiner.RefineGravity(view_graph, frames, images, pose_priors);

  ExpectEqualGravity(synthetic_dataset_options.prior_gravity_in_world,
                     gt_reconstruction,
                     pose_priors,
                     /*max_gravity_error_deg=*/1e-2);
}

}  // namespace
}  // namespace glomap
