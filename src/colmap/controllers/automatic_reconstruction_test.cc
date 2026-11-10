// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/automatic_reconstruction.h"

#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <map>
#include <vector>

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

class ParameterizedAutomaticReconstructionIntrinsicsTests
    : public ::testing::TestWithParam<
          AutomaticReconstructionController::Mapper> {};

TEST_P(ParameterizedAutomaticReconstructionIntrinsicsTests, FixedIntrinsics) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  // Like the benchmark, write the features and matches to the database
  // instead of extracting and matching them. The cameras have no prior focal
  // length, which the global mapper would otherwise calibrate.
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 5;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 200;
  std::map<camera_t, std::vector<double>> expected_params;
  {
    auto database = Database::Open(workspace_path / "database.db");
    SynthesizeDataset(
        synthetic_dataset_options, &gt_reconstruction, database.get());
    SyntheticNoiseOptions synthetic_noise_options;
    synthetic_noise_options.point2D_stddev = 0.5;
    SynthesizeNoise(
        synthetic_noise_options, &gt_reconstruction, database.get());
    // Perturb the focal lengths, which refinement or view graph calibration
    // would otherwise move back towards the GT.
    for (Camera camera : database->ReadAllCameras()) {
      ASSERT_FALSE(camera.HasPriorFocalLength());
      camera.SetFocalLength(1.1 * camera.FocalLength());
      database->UpdateCamera(camera);
      expected_params[camera.camera_id] = camera.params;
    }
  }

  AutomaticReconstructionController::Options options;
  options.workspace_path = workspace_path;
  options.image_path = image_path;
  options.extraction = false;
  options.matching = false;
  options.refine_intrinsics = false;
  options.dense = false;
  options.use_gpu = false;
  options.random_seed = 1;
  options.mapper = GetParam();

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, reconstruction_manager);
  controller.Setup();
  controller.Start();
  controller.Wait();

  ASSERT_GE(reconstruction_manager->Size(), 1);
  const Reconstruction& reconstruction = *reconstruction_manager->Get(0);
  ASSERT_EQ(reconstruction.NumRegImages(),
            synthetic_dataset_options.num_rigs *
                synthetic_dataset_options.num_frames_per_rig);
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    EXPECT_EQ(camera.params, expected_params.at(camera_id));
  }
  // View graph calibration is skipped.
  auto database = Database::Open(workspace_path / "database.db");
  for (const auto& [camera_id, _] : expected_params) {
    EXPECT_EQ(database->ReadAllCameraSources(camera_id).count(
                  CameraSource::VIEW_GRAPH),
              0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    AutomaticReconstructionTests,
    ParameterizedAutomaticReconstructionIntrinsicsTests,
    ::testing::Values(AutomaticReconstructionController::Mapper::INCREMENTAL,
                      AutomaticReconstructionController::Mapper::HIERARCHICAL,
                      AutomaticReconstructionController::Mapper::GLOBAL));

}  // namespace
}  // namespace colmap
