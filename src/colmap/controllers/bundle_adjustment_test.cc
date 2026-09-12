// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/bundle_adjustment.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"

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

}  // namespace
}  // namespace colmap
