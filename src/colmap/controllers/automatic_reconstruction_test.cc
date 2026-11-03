// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/automatic_reconstruction.h"

#include "colmap/scene/reconstruction_manager.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

// A small synthetic scene rendered to images, and controller options that
// reconstruct it sparsely.
struct SyntheticScene {
  Reconstruction gt_reconstruction;
  AutomaticReconstructionController::Options options;
};

SyntheticScene CreateSyntheticScene() {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  SyntheticScene scene;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 5;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.num_points2D_without_point3D = 10;
  SynthesizeDataset(synthetic_dataset_options, &scene.gt_reconstruction);
  SynthesizeImages(
      SyntheticImageOptions(), scene.gt_reconstruction, image_path);

  scene.options.workspace_path = workspace_path;
  scene.options.image_path = image_path;
  scene.options.data_type =
      AutomaticReconstructionController::DataType::INDIVIDUAL;
  scene.options.quality = AutomaticReconstructionController::Quality::LOW;
  scene.options.single_camera = false;
  scene.options.dense = false;  // Disable dense reconstruction to avoid GPU
  scene.options.use_gpu = false;
  scene.options.random_seed = 1;
  return scene;
}

void ReconstructAndExpectNear(const SyntheticScene& scene) {
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(scene.options,
                                               reconstruction_manager);
  controller.Setup();
  controller.Start();
  controller.Wait();

  EXPECT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(*reconstruction_manager->Get(0),
              ReconstructionNear(scene.gt_reconstruction,
                                 /*max_rotation_error_deg=*/0.6,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.9,
                                 /*align=*/true));
}

class ParameterizedAutomaticReconstructionTests
    : public ::testing::TestWithParam<
          AutomaticReconstructionController::Mapper> {};

TEST_P(ParameterizedAutomaticReconstructionTests, Nominal) {
  SyntheticScene scene = CreateSyntheticScene();
  scene.options.mapper = GetParam();
  ReconstructAndExpectNear(scene);
}

// TODO: Add GLOBAL mapper test. Currently excluded because the test produces
// fewer observations than expected. The global pipeline is tested separately
// in global_pipeline_test.cc.
INSTANTIATE_TEST_SUITE_P(
    AutomaticReconstructionTests,
    ParameterizedAutomaticReconstructionTests,
    ::testing::Values(AutomaticReconstructionController::Mapper::INCREMENTAL,
                      AutomaticReconstructionController::Mapper::HIERARCHICAL));

TEST(AutomaticReconstructionController, CusiftSetupNeedsNoGpu) {
  // Construction resolves the vocabulary tree for the feature type; cusift
  // writes SIFT descriptors and must get the SIFT tree rather than throw.
  SyntheticScene scene = CreateSyntheticScene();
  scene.options.feature = AutomaticReconstructionController::Feature::CUSIFT;
  scene.options.quality = AutomaticReconstructionController::Quality::HIGH;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  EXPECT_NO_THROW(
      AutomaticReconstructionController(scene.options, reconstruction_manager));
}

#if defined(COLMAP_CUDA_ENABLED)
TEST(AutomaticReconstructionController, CusiftHighQuality) {
  // The high quality preset enables covariant SIFT and guided matching, which
  // cusift rejects; the controller must turn them off for cusift, so this
  // reconstructs instead of throwing.
  SyntheticScene scene = CreateSyntheticScene();
  scene.options.feature = AutomaticReconstructionController::Feature::CUSIFT;
  scene.options.quality = AutomaticReconstructionController::Quality::HIGH;
  scene.options.use_gpu = true;
  scene.options.mapper = AutomaticReconstructionController::Mapper::INCREMENTAL;
  ReconstructAndExpectNear(scene);
}
#endif  // COLMAP_CUDA_ENABLED

}  // namespace
}  // namespace colmap
