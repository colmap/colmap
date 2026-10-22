// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/bundle_adjustment.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/geometry/sim3.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(BundleAdjustmentController, EmptyReconstruction) {
  auto reconstruction = std::make_shared<Reconstruction>();

  OptionManager options;
  BundleAdjustmentController controller(options, reconstruction);
  EXPECT_NO_THROW(controller.Run());

  EXPECT_EQ(reconstruction->NumRegImages(), 0);
  EXPECT_EQ(reconstruction->NumPoints3D(), 0);
}

TEST(BundleAdjustmentController, StopsBeforeOptimization) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 50;
  SynthesizeDataset(synthetic_options, &gt_reconstruction);

  auto reconstruction = std::make_shared<Reconstruction>(gt_reconstruction);
  OptionManager options;
  BundleAdjustmentController controller(options, reconstruction);
  bool stop_checked = false;
  controller.SetCheckIfStoppedFunc([&stop_checked]() {
    stop_checked = true;
    return true;
  });
  controller.Run();

  EXPECT_TRUE(stop_checked);
  EXPECT_THAT(*reconstruction, ReconstructionEq(gt_reconstruction));
}

TEST(BundleAdjustmentController, Reconstruction) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 2;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 100;
  SynthesizeDataset(synthetic_options, &gt_reconstruction);

  auto reconstruction = std::make_shared<Reconstruction>(gt_reconstruction);

  SyntheticNoiseOptions noise_options;
  noise_options.point2D_stddev = 0.1;
  noise_options.point3D_stddev = 0.1;
  noise_options.rig_from_world_rotation_stddev = 0.1;
  noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(noise_options, reconstruction.get());

  OptionManager options;
  BundleAdjustmentController controller(options, reconstruction);
  controller.Run();

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

// Loads the priors from the database.
std::vector<PosePrior> LoadPosePriors(const Database& database) {
  DatabaseCache::Options database_cache_options;
  database_cache_options.convert_pose_priors_to_enu = true;
  return DatabaseCache::Create(database, database_cache_options)->PosePriors();
}

struct PosePriorTestData {
  std::shared_ptr<Database> database;
  Reconstruction gt_reconstruction;
  // Noisy reconstruction in an arbitrary frame, such that only the priors can
  // recover the metric frame of the ground truth.
  Reconstruction noisy_reconstruction;
  SyntheticNoiseOptions noise_options;
};

PosePriorTestData CreatePosePriorTestData(
    const PosePrior::CoordinateSystem prior_coordinate_system) {
  PosePriorTestData data;
  data.database = Database::Open(CreateTestDir() / "database.db");

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_position = true;
  synthetic_dataset_options.prior_position_coordinate_system =
      prior_coordinate_system;
  SynthesizeDataset(synthetic_dataset_options,
                    &data.gt_reconstruction,
                    data.database.get());

  // Perturb the initial state of the optimization and the priors in the
  // database, while keeping the ground truth clean.
  data.noisy_reconstruction = data.gt_reconstruction;
  data.noise_options.point2D_stddev = 0.5;
  data.noise_options.point3D_stddev = 0.1;
  data.noise_options.rig_from_world_rotation_stddev = 0.1;
  data.noise_options.rig_from_world_translation_stddev = 0.1;
  data.noise_options.prior_position_stddev = 1.5;
  SynthesizeNoise(
      data.noise_options, &data.noisy_reconstruction, data.database.get());
  data.noisy_reconstruction.Transform(Sim3d(
      2.0, Eigen::Quaterniond::Identity(), Eigen::Vector3d(10.0, 20.0, 30.0)));
  return data;
}

void TestPosePriorsWithNoise(
    const PosePrior::CoordinateSystem prior_coordinate_system) {
  PosePriorTestData data = CreatePosePriorTestData(prior_coordinate_system);
  auto reconstruction =
      std::make_shared<Reconstruction>(std::move(data.noisy_reconstruction));

  PosePriorBundleAdjustmentOptions prior_options;
  prior_options.ceres->prior_position_loss_function_type =
      CeresLossFunctionType::CAUCHY;

  // Pose priors loaded as ENU or Cartesian coordinates depending on their
  // coordinate system.
  const std::vector<PosePrior> pose_priors = LoadPosePriors(*data.database);

  OptionManager options;
  BundleAdjustmentController controller(
      options,
      reconstruction,
      std::make_unique<BundleAdjustmentController::PosePriorConfig>(
          BundleAdjustmentController::PosePriorConfig{
              pose_priors, std::move(prior_options)}));
  controller.Run();

  // The comparison with ReconstructionNear() wrt. the ground truth below aligns
  // the reconstructions, so also check that the priors pulled the
  // reconstruction into their frame.
  double mean_distance_to_priors = 0;
  for (const PosePrior& pose_prior : pose_priors) {
    mean_distance_to_priors +=
        (reconstruction->Image(pose_prior.corr_data_id.id).ProjectionCenter() -
         pose_prior.position)
            .norm();
  }
  mean_distance_to_priors /= pose_priors.size();
  EXPECT_LT(mean_distance_to_priors, 2 * data.noise_options.prior_position_stddev);

  EXPECT_THAT(data.gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(BundleAdjustmentController, PosePriorsWithNoise) {
  TestPosePriorsWithNoise(PosePrior::CoordinateSystem::CARTESIAN);
}

TEST(BundleAdjustmentController, GPSPosePriorsWithNoise) {
  TestPosePriorsWithNoise(PosePrior::CoordinateSystem::WGS84);
}

TEST(BundleAdjustmentController, PosePriorsFewerThanThree) {
  PosePriorTestData data =
      CreatePosePriorTestData(PosePrior::CoordinateSystem::CARTESIAN);
  auto reconstruction_with_priors =
      std::make_shared<Reconstruction>(data.noisy_reconstruction);
  auto reconstruction_without_priors =
      std::make_shared<Reconstruction>(data.noisy_reconstruction);

  // With fewer than three priors, the frame of the reconstruction cannot be
  // constrained and the pose prior bundle adjustment must fall back to fixing
  // the gauge, i.e., end up on the same solution as plain bundle adjustment.
  std::vector<PosePrior> pose_priors = LoadPosePriors(*data.database);
  ASSERT_GT(pose_priors.size(), 2);
  pose_priors.resize(2);

  OptionManager options;
  BundleAdjustmentController controller_with_priors(
      options,
      reconstruction_with_priors,
      std::make_unique<BundleAdjustmentController::PosePriorConfig>(
          BundleAdjustmentController::PosePriorConfig{
              std::move(pose_priors), PosePriorBundleAdjustmentOptions()}));
  EXPECT_NO_THROW(controller_with_priors.Run());

  BundleAdjustmentController controller_without_priors(
      options, reconstruction_without_priors);
  controller_without_priors.Run();

  EXPECT_THAT(*reconstruction_without_priors,
              ReconstructionNear(*reconstruction_with_priors,
                                 /*max_rotation_error_deg=*/1e-6,
                                 /*max_proj_center_error=*/1e-6,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0,
                                 /*align=*/false));
}

}  // namespace
}  // namespace colmap
