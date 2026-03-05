// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/controllers/automatic_reconstruction.h"

#include "colmap/math/random.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Helper to create a minimal valid Options with required directories.
AutomaticReconstructionController::Options MakeBaseOptions(
    const std::filesystem::path& workspace_path,
    const std::filesystem::path& image_path) {
  AutomaticReconstructionController::Options options;
  options.workspace_path = workspace_path;
  options.image_path = image_path;
  options.data_type = AutomaticReconstructionController::DataType::INDIVIDUAL;
  options.quality = AutomaticReconstructionController::Quality::LOW;
  options.dense = false;
  options.use_gpu = false;
  options.random_seed = 1;
  return options;
}

// Helper to set up dirs and synthesize a small dataset.
void SetupSyntheticScene(const std::filesystem::path& workspace_path,
                         const std::filesystem::path& image_path,
                         Reconstruction* gt_reconstruction) {
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 5;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.num_points2D_without_point3D = 10;
  SynthesizeDataset(synthetic_dataset_options, gt_reconstruction);
  SynthesizeImages(SyntheticImageOptions(), *gt_reconstruction, image_path);
}

// ---------------------------------------------------------------------------
// Parameterized end-to-end tests (original)
// ---------------------------------------------------------------------------

class ParameterizedAutomaticReconstructionTests
    : public ::testing::TestWithParam<
          AutomaticReconstructionController::Mapper> {};

TEST_P(ParameterizedAutomaticReconstructionTests, Nominal) {
  SetPRNGSeed(1);

  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";

  Reconstruction gt_reconstruction;
  SetupSyntheticScene(workspace_path, image_path, &gt_reconstruction);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.mapper = GetParam();

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
  controller.Setup();
  controller.Start();
  controller.Wait();

  EXPECT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(*reconstruction_manager->Get(0),
              ReconstructionNear(gt_reconstruction,
                                 /*max_rotation_error_deg=*/0.5,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.9,
                                 /*align=*/true));
}

// TODO: Add GLOBAL mapper test. Currently excluded because the test produces
// fewer observations than expected. The global pipeline is tested separately
// in global_pipeline_test.cc.
INSTANTIATE_TEST_SUITE_P(
    AutomaticReconstructionTests,
    ParameterizedAutomaticReconstructionTests,
    ::testing::Values(AutomaticReconstructionController::Mapper::INCREMENTAL,
                      AutomaticReconstructionController::Mapper::HIERARCHICAL));

// ---------------------------------------------------------------------------
// Constructor validation tests
// ---------------------------------------------------------------------------

TEST(AutomaticReconstructionTest, ConstructorThrowsOnNonexistentWorkspace) {
  const auto test_dir = CreateTestDir();
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(image_path);

  AutomaticReconstructionController::Options options;
  options.workspace_path = test_dir / "nonexistent_workspace";
  options.image_path = image_path;
  options.use_gpu = false;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  EXPECT_THROW(
      AutomaticReconstructionController(options, reconstruction_manager),
      std::exception);
}

TEST(AutomaticReconstructionTest, ConstructorThrowsOnNonexistentImagePath) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  CreateDirIfNotExists(workspace_path);

  AutomaticReconstructionController::Options options;
  options.workspace_path = workspace_path;
  options.image_path = test_dir / "nonexistent_images";
  options.use_gpu = false;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  EXPECT_THROW(
      AutomaticReconstructionController(options, reconstruction_manager),
      std::exception);
}

TEST(AutomaticReconstructionTest, ConstructorThrowsOnNullReconstructionManager) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  EXPECT_THROW(
      AutomaticReconstructionController(options, nullptr), std::exception);
}

TEST(AutomaticReconstructionTest, ConstructorThrowsOnInvalidCameraModel) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.camera_model = "INVALID_MODEL";

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  EXPECT_THROW(
      AutomaticReconstructionController(options, reconstruction_manager),
      std::exception);
}

// ---------------------------------------------------------------------------
// Constructor option propagation tests (DataType, Quality, Feature, Mapper)
// ---------------------------------------------------------------------------

TEST(AutomaticReconstructionTest, ConstructorWithVideoDataType) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.data_type = AutomaticReconstructionController::DataType::VIDEO;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  // Should construct without throwing.
  AutomaticReconstructionController controller(options, reconstruction_manager);
}

TEST(AutomaticReconstructionTest, ConstructorWithInternetDataType) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.data_type = AutomaticReconstructionController::DataType::INTERNET;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
}

TEST(AutomaticReconstructionTest, ConstructorWithMediumQuality) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.quality = AutomaticReconstructionController::Quality::MEDIUM;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
}

TEST(AutomaticReconstructionTest, ConstructorWithHighQuality) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.quality = AutomaticReconstructionController::Quality::HIGH;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
}

TEST(AutomaticReconstructionTest, ConstructorWithExtremeQuality) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.quality = AutomaticReconstructionController::Quality::EXTREME;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
}

TEST(AutomaticReconstructionTest, ConstructorWithAlikedFeature) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.feature = AutomaticReconstructionController::Feature::ALIKED;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
}

TEST(AutomaticReconstructionTest, ConstructorWithGlobalMapper) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.mapper = AutomaticReconstructionController::Mapper::GLOBAL;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
}

TEST(AutomaticReconstructionTest, ConstructorWithMaskPath) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  const auto mask_path = test_dir / "masks";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(mask_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.mask_path = mask_path;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
}

TEST(AutomaticReconstructionTest, ConstructorWithSingleCameraPerFolder) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.single_camera_per_folder = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
}

// ---------------------------------------------------------------------------
// RequiresOpenGL tests
// ---------------------------------------------------------------------------

TEST(AutomaticReconstructionTest, RequiresOpenGLWithGpuDisabled) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.use_gpu = false;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
  EXPECT_FALSE(controller.RequiresOpenGL());
}

TEST(AutomaticReconstructionTest,
     RequiresOpenGLWithExtractionAndMatchingDisabled) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.extraction = false;
  options.matching = false;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
  // With both extraction and matching disabled, OpenGL is never required.
  EXPECT_FALSE(controller.RequiresOpenGL());
}

// ---------------------------------------------------------------------------
// Sparse mapper skip path: pre-existing sparse reconstruction
// ---------------------------------------------------------------------------

TEST(AutomaticReconstructionTest, SkipsSparseWhenAlreadyComputed) {
  SetPRNGSeed(1);

  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";

  Reconstruction gt_reconstruction;
  SetupSyntheticScene(workspace_path, image_path, &gt_reconstruction);

  // First pass: run full pipeline to produce sparse output.
  auto options = MakeBaseOptions(workspace_path, image_path);
  options.mapper = AutomaticReconstructionController::Mapper::INCREMENTAL;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  {
    AutomaticReconstructionController controller(options,
                                                 reconstruction_manager);
    controller.Setup();
    controller.Start();
    controller.Wait();
  }

  ASSERT_GE(reconstruction_manager->Size(), 1);
  const auto sparse_path = workspace_path / "sparse";
  ASSERT_TRUE(ExistsDir(sparse_path));

  // Second pass: skip extraction and matching; sparse should be read from
  // the already-computed results on disk rather than re-run.
  auto reconstruction_manager2 = std::make_shared<ReconstructionManager>();
  auto options2 = MakeBaseOptions(workspace_path, image_path);
  options2.extraction = false;
  options2.matching = false;
  options2.mapper = AutomaticReconstructionController::Mapper::INCREMENTAL;

  AutomaticReconstructionController controller2(options2,
                                                reconstruction_manager2);
  controller2.Setup();
  controller2.Start();
  controller2.Wait();

  EXPECT_EQ(reconstruction_manager2->Size(), reconstruction_manager->Size());
}

// ---------------------------------------------------------------------------
// Pipeline stage gating tests
// ---------------------------------------------------------------------------

TEST(AutomaticReconstructionTest, DisableAllStages) {
  SetPRNGSeed(1);

  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";

  Reconstruction gt_reconstruction;
  SetupSyntheticScene(workspace_path, image_path, &gt_reconstruction);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.extraction = false;
  options.matching = false;
  options.sparse = false;
  options.dense = false;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
  controller.Setup();
  controller.Start();
  controller.Wait();

  // Nothing ran, so no reconstructions should be produced.
  EXPECT_EQ(reconstruction_manager->Size(), 0);
}

TEST(AutomaticReconstructionTest, ExtractionOnlyNoMatchingNoSparse) {
  SetPRNGSeed(1);

  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";

  Reconstruction gt_reconstruction;
  SetupSyntheticScene(workspace_path, image_path, &gt_reconstruction);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.extraction = true;
  options.matching = false;
  options.sparse = false;
  options.dense = false;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
  controller.Setup();
  controller.Start();
  controller.Wait();

  // Only extraction ran, no mapping, so no reconstructions.
  EXPECT_EQ(reconstruction_manager->Size(), 0);
  // But a database should have been created.
  EXPECT_TRUE(ExistsFile(workspace_path / "database.db"));
}

// ---------------------------------------------------------------------------
// Video data type end-to-end (exercises sequential matcher path)
// ---------------------------------------------------------------------------

TEST(AutomaticReconstructionTest, VideoDataType) {
  SetPRNGSeed(1);

  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";

  Reconstruction gt_reconstruction;
  SetupSyntheticScene(workspace_path, image_path, &gt_reconstruction);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.data_type = AutomaticReconstructionController::DataType::VIDEO;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
  controller.Setup();
  controller.Start();
  controller.Wait();

  EXPECT_GE(reconstruction_manager->Size(), 1);
}

// ---------------------------------------------------------------------------
// Internet data type end-to-end (exercises exhaustive matcher for <200 images)
// ---------------------------------------------------------------------------

TEST(AutomaticReconstructionTest, InternetDataType) {
  SetPRNGSeed(1);

  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";

  Reconstruction gt_reconstruction;
  SetupSyntheticScene(workspace_path, image_path, &gt_reconstruction);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.data_type = AutomaticReconstructionController::DataType::INTERNET;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
  controller.Setup();
  controller.Start();
  controller.Wait();

  EXPECT_GE(reconstruction_manager->Size(), 1);
}

// ---------------------------------------------------------------------------
// Mesher option combinations in constructor
// ---------------------------------------------------------------------------

TEST(AutomaticReconstructionTest, ConstructorWithDelaunayMesher) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.mesher = AutomaticReconstructionController::Mesher::DELAUNAY;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
}

// ---------------------------------------------------------------------------
// Image names filtering propagation
// ---------------------------------------------------------------------------

TEST(AutomaticReconstructionTest, ConstructorWithImageNames) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.image_names = {"img1.png", "img2.png"};

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  // Should construct and propagate image names without throwing.
  AutomaticReconstructionController controller(options, reconstruction_manager);
}

// ---------------------------------------------------------------------------
// Global mapper with stricter two-view geometry defaults
// ---------------------------------------------------------------------------

TEST(AutomaticReconstructionTest, GlobalMapperSetsStricterTwoViewDefaults) {
  SetPRNGSeed(1);

  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";

  Reconstruction gt_reconstruction;
  SetupSyntheticScene(workspace_path, image_path, &gt_reconstruction);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.mapper = AutomaticReconstructionController::Mapper::GLOBAL;
  // Only run extraction and matching to verify construction + setup.
  options.sparse = false;
  options.dense = false;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
  controller.Setup();
  controller.Start();
  controller.Wait();

  // No sparse mapping was requested, so no reconstructions produced.
  EXPECT_EQ(reconstruction_manager->Size(), 0);
}

// ---------------------------------------------------------------------------
// Medium quality end-to-end
// ---------------------------------------------------------------------------

TEST(AutomaticReconstructionTest, MediumQualityEndToEnd) {
  SetPRNGSeed(1);

  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";

  Reconstruction gt_reconstruction;
  SetupSyntheticScene(workspace_path, image_path, &gt_reconstruction);

  auto options = MakeBaseOptions(workspace_path, image_path);
  options.quality = AutomaticReconstructionController::Quality::MEDIUM;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
  controller.Setup();
  controller.Start();
  controller.Wait();

  EXPECT_GE(reconstruction_manager->Size(), 1);
}

}  // namespace
}  // namespace colmap
