// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/automatic_reconstruction.h"

#include "colmap/controllers/pairing.h"
#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/retrieval/resources.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <string>

#include <gtest/gtest.h>

namespace colmap {
namespace {

class ParameterizedAutomaticReconstructionTests
    : public ::testing::TestWithParam<
          AutomaticReconstructionController::Mapper> {};

TEST_P(ParameterizedAutomaticReconstructionTests, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 5;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.num_points2D_without_point3D = 10;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction);
  SynthesizeImages(SyntheticImageOptions(), gt_reconstruction, image_path);

  AutomaticReconstructionController::Options options;
  options.workspace_path = workspace_path;
  options.image_path = image_path;
  options.data_type = AutomaticReconstructionController::DataType::INDIVIDUAL;
  options.quality = AutomaticReconstructionController::Quality::LOW;
  options.single_camera = false;
  options.dense = false;  // Disable dense reconstruction to avoid GPU
  options.use_gpu = false;
  options.random_seed = 1;
  options.mapper = GetParam();

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
  controller.Setup();
  controller.Start();
  controller.Wait();

  EXPECT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(*reconstruction_manager->Get(0),
              ReconstructionNear(gt_reconstruction,
                                 /*max_rotation_error_deg=*/0.6,
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

using Quality = AutomaticReconstructionController::Quality;

OptionManager ResolveFeatureOptions(
    const AutomaticReconstructionController::Feature feature,
    const Quality quality) {
  AutomaticReconstructionController::Options options;
  options.feature = feature;
  options.quality = quality;
  OptionManager option_manager;
  option_manager.AddAllOptions();
  AutomaticReconstructionController::ApplyFeatureOptions(options,
                                                         &option_manager);
  return option_manager;
}

class FeatureOptionsAtHighQualityTest
    : public ::testing::TestWithParam<Quality> {};

// The high and extreme presets turn on affine shape and guided matching, and
// extreme also domain-size pooling.
TEST_P(FeatureOptionsAtHighQualityTest, SiftKeepsThePresets) {
  const OptionManager option_manager = ResolveFeatureOptions(
      AutomaticReconstructionController::Feature::SIFT, GetParam());
  EXPECT_TRUE(option_manager.feature_extraction->sift->estimate_affine_shape);
  EXPECT_TRUE(option_manager.feature_matching->guided_matching);
  EXPECT_EQ(option_manager.feature_extraction->sift->domain_size_pooling,
            GetParam() == Quality::EXTREME);
}

// cusift supports neither, and writes SIFT descriptors, so it gets the SIFT
// vocabulary tree.
TEST_P(FeatureOptionsAtHighQualityTest, CusiftClearsWhatItDoesNotSupport) {
  const OptionManager option_manager = ResolveFeatureOptions(
      AutomaticReconstructionController::Feature::CUSIFT, GetParam());
  EXPECT_EQ(option_manager.feature_extraction->type,
            FeatureExtractorType::CUSIFT);
  EXPECT_EQ(option_manager.feature_matching->type,
            FeatureMatcherType::CUSIFT_GEMM);
  const std::string sift_tree =
      GetVocabTreeUriForFeatureType(FeatureExtractorType::SIFT).string();
  EXPECT_EQ(option_manager.vocab_tree_pairing->vocab_tree_path, sift_tree);
  EXPECT_EQ(option_manager.sequential_pairing->vocab_tree_path, sift_tree);
  EXPECT_FALSE(option_manager.feature_matching->guided_matching);
  EXPECT_FALSE(option_manager.feature_extraction->sift->estimate_affine_shape);
  EXPECT_FALSE(option_manager.feature_extraction->sift->domain_size_pooling);
}

INSTANTIATE_TEST_SUITE_P(AutomaticReconstructionController,
                         FeatureOptionsAtHighQualityTest,
                         ::testing::Values(Quality::HIGH, Quality::EXTREME));

}  // namespace
}  // namespace colmap
