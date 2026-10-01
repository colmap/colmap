// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/rotation_averaging.h"

#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

void ExpectEqualRotations(const Reconstruction& gt,
                          const Reconstruction& computed,
                          const double max_rotation_error_deg) {
  const double max_rotation_error_rad = DegToRad(max_rotation_error_deg);
  const std::vector<image_t> reg_image_ids = gt.RegImageIds();
  for (size_t i = 0; i < reg_image_ids.size(); i++) {
    const image_t image_id1 = reg_image_ids[i];
    for (size_t j = 0; j < i; j++) {
      const image_t image_id2 = reg_image_ids[j];
      const Eigen::Quaterniond cam2_from_cam1 =
          computed.Image(image_id2).CamFromWorld().rotation() *
          computed.Image(image_id1).CamFromWorld().rotation().inverse();
      const Eigen::Quaterniond cam2_from_cam1_gt =
          gt.Image(image_id2).CamFromWorld().rotation() *
          gt.Image(image_id1).CamFromWorld().rotation().inverse();
      EXPECT_LT(cam2_from_cam1.angularDistance(cam2_from_cam1_gt),
                max_rotation_error_rad);
    }
  }
}

TEST(RotationAveragingPipeline, WithoutNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  RotationAveragingPipelineOptions options;
  RotationAveragingPipeline controller(options, database, reconstruction);
  controller.Run();

  ExpectEqualRotations(gt_reconstruction,
                       *reconstruction,
                       /*max_rotation_error_deg=*/1e-2);
}

TEST(RotationAveragingPipeline, WithNoiseAndOutliers) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  RotationAveragingPipelineOptions options;
  RotationAveragingPipeline controller(options, database, reconstruction);
  controller.Run();

  ExpectEqualRotations(gt_reconstruction,
                       *reconstruction,
                       /*max_rotation_error_deg=*/3);
}

void ExpectExactEqualRotations(const Reconstruction& reconstruction1,
                               const Reconstruction& reconstruction2) {
  const std::vector<image_t> reg_image_ids = reconstruction1.RegImageIds();
  ASSERT_EQ(reg_image_ids.size(), reconstruction2.RegImageIds().size());
  for (const image_t image_id : reg_image_ids) {
    EXPECT_EQ(
        reconstruction1.Image(image_id).CamFromWorld().rotation().coeffs(),
        reconstruction2.Image(image_id).CamFromWorld().rotation().coeffs());
  }
}

TEST(RotationAveragingPipeline, WithRandomSeedStability) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto run_controller = [&](int num_threads, int random_seed) {
    auto reconstruction = std::make_shared<Reconstruction>();
    RotationAveragingPipelineOptions options;
    options.num_threads = num_threads;
    options.random_seed = random_seed;
    RotationAveragingPipeline controller(options, database, reconstruction);
    controller.Run();
    return reconstruction;
  };

  constexpr int kRandomSeed = 42;

  // Single-threaded execution.
  {
    auto reconstruction0 =
        run_controller(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
    auto reconstruction1 =
        run_controller(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
    ExpectExactEqualRotations(*reconstruction0, *reconstruction1);
  }

  // Multi-threaded execution.
  {
    auto reconstruction0 =
        run_controller(/*num_threads=*/3, /*random_seed=*/kRandomSeed);
    auto reconstruction1 =
        run_controller(/*num_threads=*/3, /*random_seed=*/kRandomSeed);
    // Same seed should produce similar results, up to floating-point variations
    // in optimization.
    ExpectEqualRotations(*reconstruction0,
                         *reconstruction1,
                         /*max_rotation_error_deg=*/1e-10);
  }
}

}  // namespace
}  // namespace colmap
