#include "glomap/estimators/rotation_averaging.h"

#include "colmap/geometry/triangulation.h"
#include "colmap/math/random.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include "glomap/estimators/gravity_refinement.h"
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
  options.skip_rotation_averaging = true;
  options.skip_track_establishment = true;
  options.skip_global_positioning = true;
  options.skip_bundle_adjustment = true;
  options.skip_retriangulation = true;
  return options;
}

RotationEstimatorOptions CreateRATestOptions(bool use_gravity = false) {
  RotationEstimatorOptions options;
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
  const double max_gravity_error_rad = colmap::DegToRad(max_gravity_error_deg);
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
    const double gravity_error_rad =
        colmap::CalculateAngleBetweenVectors(gravity_gt, gravity_computed);
    EXPECT_LT(gravity_error_rad, max_gravity_error_rad);
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

  auto reconstruction = std::make_shared<colmap::Reconstruction>();
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();

  GlobalMapper global_mapper(database);
  global_mapper.BeginReconstruction(reconstruction);

  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(CreateMapperTestOptions(), cluster_ids);

  // TODO: This is a misuse of frame registration. Frames should only be
  // registered when their poses are actually computed, not with arbitrary
  // identity poses. The rotation averaging code should be updated to work
  // with unregistered frames.
  // Same applies to all tests below.
  for (const auto& [frame_id, frame] : reconstruction->Frames()) {
    if (!frame.HasPose()) {
      reconstruction->Frame(frame_id).SetRigFromWorld(Rigid3d());
      reconstruction->RegisterFrame(frame_id);
    }
  }

  // TODO: The current 1-dof rotation averaging sometimes fails to pick the
  // right solution (e.g., 180 deg flipped).
  for (const bool use_gravity : {false}) {
    // Make a copy for this iteration
    colmap::Reconstruction reconstruction_copy = *reconstruction;
    SolveRotationAveraging(CreateRATestOptions(use_gravity),
                           *global_mapper.ViewGraph(),
                           reconstruction_copy,
                           pose_priors);

    ExpectEqualRotations(gt_reconstruction,
                         reconstruction_copy,
                         /*max_rotation_error_deg=*/1e-2);
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

  auto reconstruction = std::make_shared<colmap::Reconstruction>();
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();

  GlobalMapper global_mapper(database);
  global_mapper.BeginReconstruction(reconstruction);

  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(CreateMapperTestOptions(), cluster_ids);

  for (const auto& [frame_id, frame] : reconstruction->Frames()) {
    if (!frame.HasPose()) {
      reconstruction->Frame(frame_id).SetRigFromWorld(Rigid3d());
      reconstruction->RegisterFrame(frame_id);
    }
  }

  for (const bool use_gravity : {true, false}) {
    // Make a copy for this iteration
    colmap::Reconstruction reconstruction_copy = *reconstruction;
    SolveRotationAveraging(CreateRATestOptions(use_gravity),
                           *global_mapper.ViewGraph(),
                           reconstruction_copy,
                           pose_priors);

    ExpectEqualRotations(gt_reconstruction,
                         reconstruction_copy,
                         /*max_rotation_error_deg=*/1e-2);
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

  auto reconstruction = std::make_shared<colmap::Reconstruction>();
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();

  GlobalMapper global_mapper(database);
  global_mapper.BeginReconstruction(reconstruction);

  for (const auto& [rig_id, rig] : reconstruction->Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction->Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }

  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(CreateMapperTestOptions(), cluster_ids);

  for (const auto& [frame_id, frame] : reconstruction->Frames()) {
    if (!frame.HasPose()) {
      reconstruction->Frame(frame_id).SetRigFromWorld(Rigid3d());
      reconstruction->RegisterFrame(frame_id);
    }
  }

  // For unknown rigs, it is not supported to use gravity.
  for (const bool use_gravity : {false}) {
    // Make a copy for this iteration
    colmap::Reconstruction reconstruction_copy = *reconstruction;
    SolveRotationAveraging(CreateRATestOptions(use_gravity),
                           *global_mapper.ViewGraph(),
                           reconstruction_copy,
                           pose_priors);

    ExpectEqualRotations(gt_reconstruction,
                         reconstruction_copy,
                         /*max_rotation_error_deg=*/1e-2);
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

  auto reconstruction = std::make_shared<colmap::Reconstruction>();
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();
  SynthesizeGravityOutliers(pose_priors, /*outlier_ratio=*/0.3);

  GlobalMapper global_mapper(database);
  global_mapper.BeginReconstruction(reconstruction);

  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(CreateMapperTestOptions(), cluster_ids);

  for (const auto& [frame_id, frame] : reconstruction->Frames()) {
    if (!frame.HasPose()) {
      reconstruction->Frame(frame_id).SetRigFromWorld(Rigid3d());
      reconstruction->RegisterFrame(frame_id);
    }
  }

  // TODO: The current 1-dof rotation averaging sometimes fails to pick the
  // right solution (e.g., 180 deg flipped).
  for (const bool use_gravity : {false}) {
    // Make a copy for this iteration
    colmap::Reconstruction reconstruction_copy = *reconstruction;
    SolveRotationAveraging(CreateRATestOptions(use_gravity),
                           *global_mapper.ViewGraph(),
                           reconstruction_copy,
                           pose_priors);

    ExpectEqualRotations(
        gt_reconstruction, reconstruction_copy, /*max_rotation_error_deg=*/3);
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

  auto reconstruction = std::make_shared<colmap::Reconstruction>();
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();
  SynthesizeGravityOutliers(pose_priors, /*outlier_ratio=*/0.3);

  GlobalMapper global_mapper(database);
  global_mapper.BeginReconstruction(reconstruction);

  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(CreateMapperTestOptions(), cluster_ids);

  for (const auto& [frame_id, frame] : reconstruction->Frames()) {
    if (!frame.HasPose()) {
      reconstruction->Frame(frame_id).SetRigFromWorld(Rigid3d());
      reconstruction->RegisterFrame(frame_id);
    }
  }

  // TODO: The current 1-dof rotation averaging sometimes fails to pick the
  // right solution (e.g., 180 deg flipped).
  for (const bool use_gravity : {false}) {
    // Make a copy for this iteration
    colmap::Reconstruction reconstruction_copy = *reconstruction;
    SolveRotationAveraging(CreateRATestOptions(use_gravity),
                           *global_mapper.ViewGraph(),
                           reconstruction_copy,
                           pose_priors);

    ExpectEqualRotations(
        gt_reconstruction, reconstruction_copy, /*max_rotation_error_deg=*/2.);
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

  auto reconstruction = std::make_shared<colmap::Reconstruction>();
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();
  SynthesizeGravityOutliers(pose_priors, /*outlier_ratio=*/0.3);

  GlobalMapper global_mapper(database);
  global_mapper.BeginReconstruction(reconstruction);

  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(CreateMapperTestOptions(), cluster_ids);

  for (const auto& [frame_id, frame] : reconstruction->Frames()) {
    if (!frame.HasPose()) {
      reconstruction->Frame(frame_id).SetRigFromWorld(Rigid3d());
      reconstruction->RegisterFrame(frame_id);
    }
  }

  GravityRefinerOptions opt_grav_refine;
  GravityRefiner grav_refiner(opt_grav_refine);
  grav_refiner.RefineGravity(
      *global_mapper.ViewGraph(), *reconstruction, pose_priors);

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

  auto reconstruction = std::make_shared<colmap::Reconstruction>();
  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();
  SynthesizeGravityOutliers(pose_priors, /*outlier_ratio=*/0.3);

  GlobalMapper global_mapper(database);
  global_mapper.BeginReconstruction(reconstruction);

  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(CreateMapperTestOptions(), cluster_ids);

  for (const auto& [frame_id, frame] : reconstruction->Frames()) {
    if (!frame.HasPose()) {
      reconstruction->Frame(frame_id).SetRigFromWorld(Rigid3d());
      reconstruction->RegisterFrame(frame_id);
    }
  }

  GravityRefinerOptions opt_grav_refine;
  GravityRefiner grav_refiner(opt_grav_refine);
  grav_refiner.RefineGravity(
      *global_mapper.ViewGraph(), *reconstruction, pose_priors);

  ExpectEqualGravity(synthetic_dataset_options.prior_gravity_in_world,
                     gt_reconstruction,
                     pose_priors,
                     /*max_gravity_error_deg=*/1e-2);
}

}  // namespace
}  // namespace glomap
