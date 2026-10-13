// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/bundle_adjustment.h"

#include "colmap/scene/database.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

BundleAdjustmentConfig StandardConfig(const Reconstruction& reconstruction) {
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    ba_config.AddImage(image_id);
  }
  return ba_config;
}

TEST(BundleAdjustmentController, EmptyReconstruction) {
  auto reconstruction = std::make_shared<Reconstruction>();

  BundleAdjustmentOptions ba_options;
  BundleAdjustmentController controller(
      ba_options, BundleAdjustmentConfig(), reconstruction);
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
  BundleAdjustmentOptions ba_options;
  BundleAdjustmentController controller(
      ba_options, StandardConfig(*reconstruction), reconstruction);
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

  BundleAdjustmentOptions ba_options;
  BundleAdjustmentController controller(
      ba_options, StandardConfig(*reconstruction), reconstruction);
  controller.Run();

  EXPECT_TRUE(controller.Summary()->IsSolutionUsable());
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

TEST(BundleAdjustmentController, PosePriorReconstruction) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 100;
  synthetic_options.prior_position = true;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>(gt_reconstruction);

  SyntheticNoiseOptions noise_options;
  noise_options.point2D_stddev = 0.1;
  noise_options.point3D_stddev = 0.1;
  noise_options.rig_from_world_rotation_stddev = 0.1;
  noise_options.rig_from_world_translation_stddev = 0.1;
  noise_options.prior_position_stddev = 0.05;
  SynthesizeNoise(noise_options, reconstruction.get());

  const std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  BundleAdjustmentOptions ba_options;
  PosePriorBundleAdjustmentOptions prior_options;
  prior_options.alignment_ransac_options.random_seed = 0;
  BundleAdjustmentController controller(ba_options,
                                        StandardConfig(*reconstruction),
                                        prior_options,
                                        pose_priors,
                                        reconstruction);
  controller.Run();

  ASSERT_NE(controller.Summary(), nullptr);
  EXPECT_TRUE(controller.Summary()->IsSolutionUsable());
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(BundleAdjustmentController, PriorWithoutPriorsFallsBack) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 50;
  SynthesizeDataset(synthetic_options, &gt_reconstruction);

  auto reconstruction = std::make_shared<Reconstruction>(gt_reconstruction);

  BundleAdjustmentOptions ba_options;
  PosePriorBundleAdjustmentOptions prior_options;
  BundleAdjustmentController controller(ba_options,
                                        StandardConfig(*reconstruction),
                                        prior_options,
                                        /*pose_priors=*/{},
                                        reconstruction);
  EXPECT_NO_THROW(controller.Run());

  ASSERT_NE(controller.Summary(), nullptr);
  EXPECT_TRUE(controller.Summary()->IsSolutionUsable());
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-3,
                                 /*max_proj_center_error=*/1e-3,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

}  // namespace
}  // namespace colmap
