// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/automatic_reconstruction.h"

#include "colmap/scene/reconstruction_manager.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/cuda.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Renders a small synthetic scene to images and sets controller options that
// reconstruct it sparsely on the CPU; tests adjust options_ before running.
class SyntheticAutomaticReconstructionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto test_dir = CreateTestDir();
    const auto workspace_path = test_dir / "workspace";
    const auto image_path = test_dir / "images";
    CreateDirIfNotExists(workspace_path);
    CreateDirIfNotExists(image_path);

    SyntheticDatasetOptions synthetic_dataset_options;
    synthetic_dataset_options.num_rigs = 5;
    synthetic_dataset_options.num_cameras_per_rig = 1;
    synthetic_dataset_options.num_frames_per_rig = 1;
    synthetic_dataset_options.num_points3D = 200;
    synthetic_dataset_options.num_points2D_without_point3D = 10;
    SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction_);
    SynthesizeImages(SyntheticImageOptions(), gt_reconstruction_, image_path);

    options_.workspace_path = workspace_path;
    options_.image_path = image_path;
    options_.data_type =
        AutomaticReconstructionController::DataType::INDIVIDUAL;
    options_.quality = AutomaticReconstructionController::Quality::LOW;
    options_.single_camera = false;
    options_.dense = false;  // Disable dense reconstruction to avoid GPU
    options_.use_gpu = false;
    options_.random_seed = 1;
  }

  void ReconstructAndExpectNear() const {
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    AutomaticReconstructionController controller(options_,
                                                 reconstruction_manager);
    controller.Setup();
    controller.Start();
    controller.Wait();

    EXPECT_EQ(reconstruction_manager->Size(), 1);
    EXPECT_THAT(*reconstruction_manager->Get(0),
                ReconstructionNear(gt_reconstruction_,
                                   /*max_rotation_error_deg=*/0.6,
                                   /*max_proj_center_error=*/0.1,
                                   /*max_scale_error=*/std::nullopt,
                                   /*num_obs_tolerance=*/0.9,
                                   /*align=*/true));
  }

  Reconstruction gt_reconstruction_;
  AutomaticReconstructionController::Options options_;
};

class ParameterizedAutomaticReconstructionTests
    : public SyntheticAutomaticReconstructionTest,
      public ::testing::WithParamInterface<
          AutomaticReconstructionController::Mapper> {};

TEST_P(ParameterizedAutomaticReconstructionTests, Nominal) {
  options_.mapper = GetParam();
  ReconstructAndExpectNear();
}

// TODO: Add GLOBAL mapper test. Currently excluded because the test produces
// fewer observations than expected. The global pipeline is tested separately
// in global_pipeline_test.cc.
INSTANTIATE_TEST_SUITE_P(
    AutomaticReconstructionTests,
    ParameterizedAutomaticReconstructionTests,
    ::testing::Values(AutomaticReconstructionController::Mapper::INCREMENTAL,
                      AutomaticReconstructionController::Mapper::HIERARCHICAL));

TEST_F(SyntheticAutomaticReconstructionTest, CusiftSetupNeedsNoGpu) {
  // Construction resolves the vocabulary tree for the feature type; cusift
  // writes SIFT descriptors and must get the SIFT tree rather than throw.
  options_.feature = AutomaticReconstructionController::Feature::CUSIFT;
  options_.quality = AutomaticReconstructionController::Quality::HIGH;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  EXPECT_NO_THROW(
      AutomaticReconstructionController(options_, reconstruction_manager));
}

#if defined(COLMAP_CUDA_ENABLED)
TEST_F(SyntheticAutomaticReconstructionTest, CusiftHighQuality) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // The high quality preset enables covariant SIFT and guided matching, which
  // cusift rejects; the controller must turn them off for cusift, so this
  // reconstructs instead of throwing.
  options_.feature = AutomaticReconstructionController::Feature::CUSIFT;
  options_.quality = AutomaticReconstructionController::Quality::HIGH;
  options_.use_gpu = true;
  options_.mapper = AutomaticReconstructionController::Mapper::INCREMENTAL;
  ReconstructAndExpectNear();
}
#endif  // COLMAP_CUDA_ENABLED

}  // namespace
}  // namespace colmap
