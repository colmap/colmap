// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/bundle_adjustment.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/scene/database_sqlite.h"
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

TEST(BundleAdjustmentController, WithDatabasePositionPriors) {
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 6;
  synthetic_options.num_points3D = 80;
  synthetic_options.prior_position = true;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  // Store bogus covariances in the database and verify that
  // prior_position_overwrite_covariance overrides them in memory without
  // mutating the database on disk.
  const Eigen::Matrix3d orig_db_cov = 1e4 * Eigen::Matrix3d::Identity();
  for (PosePrior prior : database->ReadAllPosePriors()) {
    prior.position_covariance = orig_db_cov;
    database->UpdatePosePrior(prior);
  }
  database.reset();

  auto reconstruction = std::make_shared<Reconstruction>(gt_reconstruction);
  SyntheticNoiseOptions noise_options;
  noise_options.point2D_stddev = 0.2;
  noise_options.point3D_stddev = 0.05;
  noise_options.rig_from_world_rotation_stddev = 0.5;
  noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(noise_options, reconstruction.get());
  reconstruction->Transform(
      Sim3d(2.5, Eigen::Quaterniond::Identity(), Eigen::Vector3d(5, -3, 2)));

  OptionManager options;
  *options.database_path = database_path;
  options.bundle_adjustment->use_prior_position = true;
  options.bundle_adjustment->prior_position_overwrite_covariance = true;
  options.bundle_adjustment->prior_position_stddev =
      Eigen::Vector3d::Constant(0.01);
  options.bundle_adjustment->prior_position_alignment_ransac_options
      .random_seed = 0;

  BundleAdjustmentController controller(options, reconstruction);
  controller.Run();

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/0.2,
                                 /*max_proj_center_error=*/0.05,
                                 /*max_scale_error=*/1e-2,
                                 /*num_obs_tolerance=*/0.0,
                                 /*align=*/false));

  const auto reopened_db = Database::Open(database_path);
  for (const PosePrior& prior : reopened_db->ReadAllPosePriors()) {
    EXPECT_EQ(prior.position_covariance, orig_db_cov);
  }
}

TEST(BundleAdjustmentController, WithExplicitPositionPriors) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 60;
  synthetic_options.prior_position = true;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  BundleAdjustmentPriors priors;
  priors.pose_priors = database->ReadAllPosePriors();

  auto reconstruction = std::make_shared<Reconstruction>(gt_reconstruction);
  reconstruction->Transform(
      Sim3d(3.0, Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 2, 3)));

  OptionManager options;
  options.bundle_adjustment->use_prior_position = true;
  options.bundle_adjustment->prior_position_alignment_ransac_options
      .random_seed = 0;

  BundleAdjustmentController controller(
      options, reconstruction, std::move(priors));
  controller.Run();

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.05,
                                 /*max_scale_error=*/1e-2,
                                 /*num_obs_tolerance=*/0.0,
                                 /*align=*/false));
}

TEST(BundleAdjustmentController, MissingDatabasePathThrowsWhenPriorsEnabled) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 30;
  SynthesizeDataset(synthetic_options, &gt_reconstruction);

  auto reconstruction = std::make_shared<Reconstruction>(gt_reconstruction);
  OptionManager options;
  options.bundle_adjustment->use_prior_position = true;

  BundleAdjustmentController empty_db_controller(options, reconstruction);
  EXPECT_ANY_THROW(empty_db_controller.Run());

  const auto missing_db_path = CreateTestDir() / "missing.db";
  *options.database_path = missing_db_path;
  BundleAdjustmentController missing_db_controller(options, reconstruction);
  EXPECT_ANY_THROW(missing_db_controller.Run());
  EXPECT_FALSE(std::filesystem::exists(missing_db_path));
}

}  // namespace
}  // namespace colmap
