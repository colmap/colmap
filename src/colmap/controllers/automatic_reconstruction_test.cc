// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/automatic_reconstruction.h"

#include "colmap/math/random.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <string>
#include <utility>
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

TEST(AutomaticReconstruction, ImageReaderNamesSurviveVideoPreset) {
  const auto workspace = CreateTestDir();
  const auto image_path = workspace / "images";
  CreateDirIfNotExists(image_path);
  Bitmap bitmap(100, 100, /*as_rgb=*/false);
  bitmap.Fill(BitmapColor<uint8_t>(0));
  for (int y = 30; y < 70; ++y) {
    for (int x = 30; x < 70; ++x) {
      bitmap.SetPixel(x, y, BitmapColor<uint8_t>(255));
    }
  }
  ASSERT_TRUE(bitmap.Write(image_path / "selected.png"));
  ASSERT_TRUE(bitmap.Write(image_path / "excluded.png"));

  AutomaticReconstructionController::Options options;
  options.workspace_path = workspace;
  options.image_path = image_path;
  options.image_names = {"selected.png"};
  options.data_type = AutomaticReconstructionController::DataType::VIDEO;
  options.quality = AutomaticReconstructionController::Quality::LOW;
  options.extraction = true;
  options.single_view_calibration = false;
  options.matching = false;
  options.sparse = false;
  options.dense = false;
  options.use_gpu = false;
  options.num_threads = 1;
  options.camera_model = "SIMPLE_PINHOLE";
  options.camera_params = "100,50,50";
  auto manager = std::make_shared<ReconstructionManager>();
  AutomaticReconstructionController controller(options, manager);
  controller.Setup();
  controller.Start();
  ASSERT_NO_THROW(controller.Wait());

  auto database = Database::Open(workspace / "database.db");
  const auto images = database->ReadAllImages();
  ASSERT_EQ(images.size(), 1);
  EXPECT_EQ(images.front().Name(), "selected.png");
  EXPECT_TRUE(database->ExistsKeypoints(images.front().ImageId()));
  EXPECT_TRUE(database->ExistsDescriptors(images.front().ImageId()));
}

TEST(AutomaticReconstruction, ImageNamesSurvivePresetsAndDispatch) {
  using DataType = AutomaticReconstructionController::DataType;
  using Mapper = AutomaticReconstructionController::Mapper;
  const auto test_dir = CreateTestDir();
  size_t case_idx = 0;
  for (const auto& [mapper, data_type] :
       {std::pair{Mapper::GLOBAL, DataType::INDIVIDUAL},
        std::pair{Mapper::GLOBAL, DataType::VIDEO},
        std::pair{Mapper::INCREMENTAL, DataType::VIDEO},
        std::pair{Mapper::HIERARCHICAL, DataType::VIDEO}}) {
    SCOPED_TRACE(static_cast<int>(mapper));
    SCOPED_TRACE(static_cast<int>(data_type));
    for (const bool select_subset : {false, true}) {
      SCOPED_TRACE(select_subset);
      SetPRNGSeed(1);
      const auto workspace = test_dir / std::to_string(case_idx++);
      CreateDirIfNotExists(workspace);
      const auto image_path = workspace / "images";
      CreateDirIfNotExists(image_path);
      auto database = Database::Open(workspace / "database.db");
      Reconstruction expected;
      SyntheticDatasetOptions dataset_options;
      dataset_options.num_rigs = 2;
      dataset_options.num_cameras_per_rig = 1;
      dataset_options.num_frames_per_rig = 5;
      dataset_options.num_points3D = 100;
      dataset_options.camera_width = 128;
      dataset_options.camera_height = 96;
      dataset_options.camera_params = {160, 64, 48, 0.05};
      dataset_options.camera_has_prior_focal_length = true;
      dataset_options.two_view_geometry_has_relative_pose = true;
      SynthesizeDataset(dataset_options, &expected, database.get());
      SynthesizeImages(SyntheticImageOptions(), expected, image_path);

      AutomaticReconstructionController::Options options;
      options.workspace_path = workspace;
      options.image_path = image_path;
      options.extraction = false;
      options.matching = false;
      options.dense = false;
      options.use_gpu = false;
      options.num_threads = 1;
      options.random_seed = 1;
      options.data_type = data_type;
      options.mapper = mapper;
      if (select_subset) {
        // Select complete frames from one rig. Build the oracle from known
        // synthetic geometry, independently of DatabaseCache's name filter.
        const rig_t selected_rig = expected.Rigs().begin()->first;
        std::vector<frame_t> excluded_frames;
        for (const auto& [frame_id, frame] : expected.Frames()) {
          if (frame.RigId() == selected_rig) {
            for (const data_t& data_id : frame.ImageIds()) {
              options.image_names.push_back(expected.Image(data_id.id).Name());
            }
          } else {
            excluded_frames.push_back(frame_id);
          }
        }
        for (const frame_t frame_id : excluded_frames) {
          expected.DeRegisterFrame(frame_id);
        }
        expected.TearDown();
        ASSERT_EQ(expected.NumRegImages(), 5);
      }

      auto manager = std::make_shared<ReconstructionManager>();
      AutomaticReconstructionController controller(options, manager);
      controller.Setup();
      controller.Start();
      ASSERT_NO_THROW(controller.Wait());
      ASSERT_EQ(manager->Size(), 1);
      const Reconstruction& actual = *manager->Get(0);
      ASSERT_TRUE(actual.IsValid());
      EXPECT_EQ(actual.NumRegImages(), expected.NumRegImages());
      for (const image_t image_id : actual.RegImageIds()) {
        EXPECT_TRUE(expected.ExistsImage(image_id));
      }
      EXPECT_THAT(actual,
                  ReconstructionNear(expected,
                                     /*max_rotation_error_deg=*/1e-2,
                                     /*max_proj_center_error=*/1e-4));
    }
  }
}

// TODO: Add GLOBAL mapper test. Currently excluded because the test produces
// fewer observations than expected. The global pipeline is tested separately
// in global_pipeline_test.cc.
INSTANTIATE_TEST_SUITE_P(
    AutomaticReconstructionTests,
    ParameterizedAutomaticReconstructionTests,
    ::testing::Values(AutomaticReconstructionController::Mapper::INCREMENTAL,
                      AutomaticReconstructionController::Mapper::HIERARCHICAL));

}  // namespace
}  // namespace colmap
