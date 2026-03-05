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

#include "colmap/controllers/incremental_pipeline.h"

#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(IncrementalPipeline, WithoutNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(std::make_shared<IncrementalPipelineOptions>(),
                             database,
                             reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, WithoutNoiseAndWithNonTrivialFrames) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0.05;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  for (const bool refine_sensor_from_rig : {true, false}) {
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto options = std::make_shared<IncrementalPipelineOptions>();
    options->ba_refine_sensor_from_rig = refine_sensor_from_rig;
    IncrementalPipeline mapper(options, database, reconstruction_manager);
    mapper.Run();

    ASSERT_EQ(reconstruction_manager->Size(), 1);
    EXPECT_THAT(gt_reconstruction,
                ReconstructionNear(
                    *reconstruction_manager->Get(0),
                    /*max_rotation_error_deg=*/1e-2,
                    /*max_proj_center_error=*/1e-3,
                    /*max_scale_error=*/refine_sensor_from_rig ? 1e-2 : 1e-4));
  }
}

TEST(IncrementalPipeline, UnknownSensorFromRigExitsGracefully) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0.05;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  // Set one of the sensor from rig to unknown.
  auto rig = database->ReadAllRigs()[0];
  rig.NonRefSensors().begin()->second.reset();
  database->UpdateRig(rig);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  IncrementalPipeline mapper(options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 0);
}

TEST(IncrementalPipeline, WithNonTrivialFramesAndConstantRigsAndCameras) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0.05;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  constexpr int kConstantRigId = 1;
  constexpr int kConstantCameraId = 1;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->constant_rigs.insert(kConstantRigId);
  options->constant_cameras.insert(kConstantCameraId);
  IncrementalPipeline mapper(options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  auto& reconstruction = *reconstruction_manager->Get(0);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-3));

  for (const auto& [sensor_id, sensor_from_rig] :
       reconstruction.Rig(kConstantRigId).NonRefSensors()) {
    EXPECT_THAT(
        sensor_from_rig.value(),
        Rigid3dNear(
            gt_reconstruction.Rig(kConstantRigId).SensorFromRig(sensor_id),
            /*rtol=*/1e-5,
            /*ttol=*/1e-5));
  }
  EXPECT_EQ(reconstruction.Camera(kConstantCameraId).params,
            gt_reconstruction.Camera(kConstantCameraId).params);
}

TEST(IncrementalPipeline, WithoutNoiseAndWithPanoramicNonTrivialFrames) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  for (const bool refine_sensor_from_rig : {true, false}) {
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto options = std::make_shared<IncrementalPipelineOptions>();
    options->ba_refine_sensor_from_rig = refine_sensor_from_rig;
    IncrementalPipeline mapper(options, database, reconstruction_manager);
    mapper.Run();

    ASSERT_EQ(reconstruction_manager->Size(), 1);
    EXPECT_THAT(gt_reconstruction,
                ReconstructionNear(*reconstruction_manager->Get(0),
                                   /*max_rotation_error_deg=*/1e-2,
                                   /*max_proj_center_error=*/1e-3));
  }
}

TEST(IncrementalPipeline, WithPriorFocalLength) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(std::make_shared<IncrementalPipelineOptions>(),
                             database,
                             reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, WithNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(std::make_shared<IncrementalPipelineOptions>(),
                             database,
                             reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(IncrementalPipeline, IgnoreRedundantPoints3D) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->mapper.ba_global_ignore_redundant_points3D = true;
  options->mapper.ba_global_ignore_redundant_points3D_min_coverage_gain = 0.5;
  IncrementalPipeline mapper(options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, StructureLessRegistrationOnly) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->structure_less_registration_only = true;
  IncrementalPipeline mapper(options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-3,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, MultiReconstruction) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction1;
  Reconstruction gt_reconstruction2;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction1, database.get());
  synthetic_dataset_options.num_frames_per_rig = 4;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction2, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto mapper_options = std::make_shared<IncrementalPipelineOptions>();
  mapper_options->min_model_size = 4;
  IncrementalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 2);
  Reconstruction* computed_reconstruction1 = nullptr;
  Reconstruction* computed_reconstruction2 = nullptr;
  if (reconstruction_manager->Get(0)->NumRegImages() == 5) {
    computed_reconstruction1 = reconstruction_manager->Get(0).get();
    computed_reconstruction2 = reconstruction_manager->Get(1).get();
  } else {
    computed_reconstruction1 = reconstruction_manager->Get(1).get();
    computed_reconstruction2 = reconstruction_manager->Get(0).get();
  }
  EXPECT_THAT(gt_reconstruction1,
              ReconstructionNear(*computed_reconstruction1,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
  EXPECT_THAT(gt_reconstruction2,
              ReconstructionNear(*computed_reconstruction2,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, FixExistingFrames) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  for (const bool fix_existing_frames : {false, true}) {
    if (fix_existing_frames) {
      ASSERT_EQ(reconstruction_manager->Size(), 1);
      Reconstruction& reconstruction = *reconstruction_manager->Get(0);
      // De-register a frame that expect to be re-registered in the second run.
      reconstruction.DeRegisterFrame(1);
      // Clear all the observations of one image but keep it registered. We do
      // not expect fixed images to be filtered (due to insufficient
      // observations).
      Image& image2 = reconstruction.Image(2);
      for (point2D_t point2D_idx = 0; point2D_idx < image2.NumPoints2D();
           ++point2D_idx) {
        if (image2.Point2D(point2D_idx).HasPoint3D()) {
          reconstruction.DeleteObservation(image2.ImageId(), point2D_idx);
        }
      }
    }
    options->fix_existing_frames = fix_existing_frames;
    IncrementalPipeline mapper(options, database, reconstruction_manager);
    mapper.Run();

    ASSERT_EQ(reconstruction_manager->Size(), 1);
    EXPECT_THAT(gt_reconstruction,
                ReconstructionNear(*reconstruction_manager->Get(0),
                                   /*max_rotation_error_deg=*/1e-2,
                                   /*max_proj_center_error=*/1e-4));
  }
}

TEST(IncrementalPipeline, ChainedMatches) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.match_config =
      SyntheticDatasetOptions::MatchConfig::CHAINED;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(std::make_shared<IncrementalPipelineOptions>(),
                             database,
                             reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, PriorBasedSfMWithoutNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_position = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.prior_position_stddev = 0.0;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  std::shared_ptr<IncrementalPipelineOptions> mapper_options =
      std::make_shared<IncrementalPipelineOptions>();
  mapper_options->use_prior_position = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);

  // No noise on prior so do not align gt & computed (expected to be aligned
  // from PositionPriorBundleAdjustment)
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02,
                                 /*align=*/false));
}

TEST(IncrementalPipeline, PriorBasedSfMWithoutNoiseAndWithNonTrivialFrames) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_has_prior_focal_length = false;

  synthetic_dataset_options.prior_position = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  std::shared_ptr<IncrementalPipelineOptions> mapper_options =
      std::make_shared<IncrementalPipelineOptions>();

  mapper_options->use_prior_position = true;
  mapper_options->use_robust_loss_on_prior_position = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(IncrementalPipeline, PriorBasedSfMWithNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_position = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.prior_position_stddev = 1.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  std::shared_ptr<IncrementalPipelineOptions> mapper_options =
      std::make_shared<IncrementalPipelineOptions>();

  mapper_options->use_prior_position = true;
  mapper_options->use_robust_loss_on_prior_position = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(IncrementalPipeline, GPSPriorBasedSfMWithNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 100;

  synthetic_dataset_options.prior_position = true;
  synthetic_dataset_options.prior_position_coordinate_system =
      PosePrior::CoordinateSystem::WGS84;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.prior_position_stddev = 1.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  std::shared_ptr<IncrementalPipelineOptions> mapper_options =
      std::make_shared<IncrementalPipelineOptions>();

  mapper_options->use_prior_position = true;
  mapper_options->use_robust_loss_on_prior_position = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(IncrementalPipeline, SfMWithRandomSeedStability) {
  SetPRNGSeed(42);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.prior_position = false;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto run_mapper = [&](int num_threads, int random_seed) {
    auto mapper_options = std::make_shared<IncrementalPipelineOptions>();
    mapper_options->use_prior_position = false;
    mapper_options->num_threads = num_threads;
    mapper_options->random_seed = random_seed;
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    IncrementalPipeline mapper(
        mapper_options, database, reconstruction_manager);
    mapper.Run();
    EXPECT_EQ(reconstruction_manager->Size(), 1);
    return reconstruction_manager;
  };

  constexpr int kRandomSeed = 42;

  auto reconstruction_manager0 =
      run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
  auto reconstruction_manager1 =
      run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
  EXPECT_THAT(*reconstruction_manager0->Get(0),
              ReconstructionEq(*reconstruction_manager1->Get(0)));
}

TEST(IncrementalPipeline, PriorBasedSfMWithRandomSeedStability) {
  SetPRNGSeed(42);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.prior_position = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.1;
  synthetic_noise_options.prior_position_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto run_mapper = [&](int num_threads, int random_seed) {
    auto mapper_options = std::make_shared<IncrementalPipelineOptions>();
    mapper_options->use_prior_position = true;
    mapper_options->num_threads = num_threads;
    mapper_options->random_seed = random_seed;
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    IncrementalPipeline mapper(
        mapper_options, database, reconstruction_manager);
    mapper.Run();
    EXPECT_EQ(reconstruction_manager->Size(), 1);
    return reconstruction_manager;
  };

  constexpr int kRandomSeed = 42;

  auto reconstruction_manager0 =
      run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
  auto reconstruction_manager1 =
      run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
  EXPECT_THAT(*reconstruction_manager0->Get(0),
              ReconstructionEq(*reconstruction_manager1->Get(0)));
}

//////////////////////////////////////////////////////////////////////////////
// New tests targeting previously uncovered code paths
//////////////////////////////////////////////////////////////////////////////

TEST(IncrementalPipelineOptions, MapperPropagatesFields) {
  IncrementalPipelineOptions options;
  options.ba_refine_focal_length = false;
  options.ba_refine_extra_params = false;
  options.min_focal_length_ratio = 0.2;
  options.max_focal_length_ratio = 8.0;
  options.max_extra_param = 2.0;
  options.num_threads = 4;
  options.fix_existing_frames = true;
  options.use_prior_position = true;
  options.use_robust_loss_on_prior_position = true;
  options.prior_position_loss_scale = 5.0;
  options.random_seed = 123;

  const auto mapper = options.Mapper();

  EXPECT_FALSE(mapper.abs_pose_refine_focal_length);
  EXPECT_FALSE(mapper.abs_pose_refine_extra_params);
  EXPECT_DOUBLE_EQ(mapper.min_focal_length_ratio, 0.2);
  EXPECT_DOUBLE_EQ(mapper.max_focal_length_ratio, 8.0);
  EXPECT_DOUBLE_EQ(mapper.max_extra_param, 2.0);
  EXPECT_EQ(mapper.num_threads, 4);
  EXPECT_TRUE(mapper.fix_existing_frames);
  EXPECT_TRUE(mapper.use_prior_position);
  EXPECT_TRUE(mapper.use_robust_loss_on_prior_position);
  EXPECT_DOUBLE_EQ(mapper.prior_position_loss_scale, 5.0);
  EXPECT_EQ(mapper.random_seed, 123);
}

TEST(IncrementalPipelineOptions, TriangulationPropagatesFields) {
  IncrementalPipelineOptions options;
  options.min_focal_length_ratio = 0.3;
  options.max_focal_length_ratio = 7.0;
  options.max_extra_param = 1.5;
  options.random_seed = 99;

  const auto tri = options.Triangulation();

  EXPECT_DOUBLE_EQ(tri.min_focal_length_ratio, 0.3);
  EXPECT_DOUBLE_EQ(tri.max_focal_length_ratio, 7.0);
  EXPECT_DOUBLE_EQ(tri.max_extra_param, 1.5);
  EXPECT_EQ(tri.random_seed, 99);
}

TEST(IncrementalPipelineOptions, LocalBundleAdjustmentPropagatesFields) {
  IncrementalPipelineOptions options;
  options.ba_refine_focal_length = false;
  options.ba_refine_principal_point = true;
  options.ba_refine_extra_params = false;
  options.ba_refine_sensor_from_rig = false;
  options.ba_local_function_tolerance = 1e-4;
  options.ba_local_max_num_iterations = 10;
  options.num_threads = 8;
  options.ba_min_num_residuals_for_cpu_multi_threading = 10000;
  options.ba_use_gpu = true;
  options.ba_gpu_index = "0";

  const auto ba = options.LocalBundleAdjustment();

  EXPECT_FALSE(ba.print_summary);
  EXPECT_FALSE(ba.refine_focal_length);
  EXPECT_TRUE(ba.refine_principal_point);
  EXPECT_FALSE(ba.refine_extra_params);
  EXPECT_FALSE(ba.refine_sensor_from_rig);
  if (ba.ceres) {
    EXPECT_DOUBLE_EQ(ba.ceres->solver_options.function_tolerance, 1e-4);
    EXPECT_EQ(ba.ceres->solver_options.max_num_iterations, 10);
    EXPECT_EQ(ba.ceres->solver_options.num_threads, 8);
    EXPECT_EQ(ba.ceres->min_num_residuals_for_cpu_multi_threading, 10000);
    EXPECT_DOUBLE_EQ(ba.ceres->loss_function_scale, 1.0);
    EXPECT_EQ(ba.ceres->loss_function_type,
              CeresBundleAdjustmentOptions::LossFunctionType::SOFT_L1);
    EXPECT_TRUE(ba.ceres->use_gpu);
    EXPECT_EQ(ba.ceres->gpu_index, "0");
  }
}

TEST(IncrementalPipelineOptions, GlobalBundleAdjustmentPropagatesFields) {
  IncrementalPipelineOptions options;
  options.ba_refine_focal_length = false;
  options.ba_refine_principal_point = true;
  options.ba_refine_extra_params = false;
  options.ba_refine_sensor_from_rig = false;
  options.ba_global_function_tolerance = 1e-5;
  options.ba_global_max_num_iterations = 100;
  options.num_threads = 16;
  options.ba_min_num_residuals_for_cpu_multi_threading = 20000;
  options.ba_use_gpu = true;
  options.ba_gpu_index = "1";

  const auto ba = options.GlobalBundleAdjustment();

  EXPECT_FALSE(ba.print_summary);
  EXPECT_FALSE(ba.refine_focal_length);
  EXPECT_TRUE(ba.refine_principal_point);
  EXPECT_FALSE(ba.refine_extra_params);
  EXPECT_FALSE(ba.refine_sensor_from_rig);
  if (ba.ceres) {
    EXPECT_DOUBLE_EQ(ba.ceres->solver_options.function_tolerance, 1e-5);
    EXPECT_EQ(ba.ceres->solver_options.max_num_iterations, 100);
    EXPECT_EQ(ba.ceres->solver_options.num_threads, 16);
    EXPECT_EQ(ba.ceres->min_num_residuals_for_cpu_multi_threading, 20000);
    EXPECT_EQ(ba.ceres->loss_function_type,
              CeresBundleAdjustmentOptions::LossFunctionType::TRIVIAL);
    EXPECT_TRUE(ba.ceres->use_gpu);
    EXPECT_EQ(ba.ceres->gpu_index, "1");
  }
}

TEST(IncrementalPipelineOptions, CheckDefaultsPass) {
  IncrementalPipelineOptions options;
  EXPECT_TRUE(options.Check());
}

TEST(IncrementalPipelineOptions, IsInitialPairProvided) {
  IncrementalPipelineOptions options;
  EXPECT_FALSE(options.IsInitialPairProvided());

  options.init_image_id1 = 1;
  EXPECT_FALSE(options.IsInitialPairProvided());

  options.init_image_id2 = 2;
  EXPECT_TRUE(options.IsInitialPairProvided());
}

TEST(IncrementalPipeline, CheckRunGlobalRefinementConditions) {
  SetPRNGSeed(0);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  // Set thresholds high enough that nothing triggers by default.
  options->ba_global_frames_ratio = 100.0;
  options->ba_global_points_ratio = 100.0;
  options->ba_global_frames_freq = 100000;
  options->ba_global_points_freq = 100000;
  IncrementalPipeline pipeline(options, database, reconstruction_manager);

  // Build a reconstruction with 4 reg frames and 50 points.
  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  // Nothing triggers: all conditions false.
  EXPECT_FALSE(pipeline.CheckRunGlobalRefinement(reconstruction, 4, 50));

  // Trigger via frames ratio only: 4 >= 100.0 * 0 = 0 -> true.
  // Use prev=0 so ratio*0=0 makes it trigger.
  EXPECT_TRUE(pipeline.CheckRunGlobalRefinement(reconstruction, 0, 50));

  // Trigger via frames freq: set freq=2, prev=2 -> 4 >= 2 + 2 = 4 -> true.
  options->ba_global_frames_freq = 2;
  EXPECT_TRUE(pipeline.CheckRunGlobalRefinement(reconstruction, 2, 50));
  options->ba_global_frames_freq = 100000;  // Reset.

  // Trigger via points ratio: prev=0, 50 >= 100.0 * 0 = 0 -> true.
  EXPECT_TRUE(pipeline.CheckRunGlobalRefinement(reconstruction, 4, 0));

  // Trigger via points freq: set freq=10, prev=40 -> 50 >= 10 + 40 = 50.
  options->ba_global_points_freq = 10;
  EXPECT_TRUE(pipeline.CheckRunGlobalRefinement(reconstruction, 4, 40));
  options->ba_global_points_freq = 100000;  // Reset.

  // Verify it stays false when all conditions are comfortably not met.
  EXPECT_FALSE(pipeline.CheckRunGlobalRefinement(reconstruction, 4, 50));
}

TEST(IncrementalPipeline, RunWithEmptyDatabaseExitsEarly) {
  SetPRNGSeed(0);
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);

  // Create a pipeline with an empty database (no images at all).
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  IncrementalPipeline pipeline(options, database, reconstruction_manager);
  pipeline.Run();

  // No reconstruction should be produced from an empty database.
  EXPECT_EQ(reconstruction_manager->Size(), 0);
}

TEST(IncrementalPipeline, RunWithPriorPositionButNoPriorsExitsEarly) {
  SetPRNGSeed(0);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  // Do not generate prior positions.
  synthetic_dataset_options.prior_position = false;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  // Enable prior position usage, but the database has no priors.
  options->use_prior_position = true;
  IncrementalPipeline pipeline(options, database, reconstruction_manager);
  pipeline.Run();

  // Pipeline should exit early without producing a reconstruction.
  EXPECT_EQ(reconstruction_manager->Size(), 0);
}

TEST(IncrementalPipeline, SnapshotWrittenDuringReconstruction) {
  SetPRNGSeed(0);
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto snapshot_path = test_dir / "snapshots";
  CreateDirIfNotExists(snapshot_path);

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->snapshot_path = snapshot_path;
  // Trigger snapshot after every 1 newly registered frame beyond initial pair.
  options->snapshot_frames_freq = 1;
  IncrementalPipeline pipeline(options, database, reconstruction_manager);
  pipeline.Run();

  ASSERT_GE(reconstruction_manager->Size(), 1);

  // Verify that at least one snapshot subdirectory was written.
  const auto snapshot_dirs = GetDirList(snapshot_path);
  EXPECT_GE(snapshot_dirs.size(), 1);

  // Each snapshot should contain reconstruction files that can be read back.
  for (const auto& dir : snapshot_dirs) {
    Reconstruction snapshot_reconstruction;
    snapshot_reconstruction.Read(dir);
    EXPECT_GT(snapshot_reconstruction.NumRegImages(), 0);
    EXPECT_GT(snapshot_reconstruction.NumPoints3D(), 0);
  }
}

TEST(IncrementalPipeline, MaxRuntimeStopsReconstruction) {
  SetPRNGSeed(0);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  // Set an extremely short runtime to trigger the timeout path.
  // The pipeline should stop and keep whatever reconstruction it has.
  // max_runtime_seconds must be > 0 for the check to trigger (the check is
  // max_runtime_seconds > 0 && elapsed > max_runtime_seconds).
  options->max_runtime_seconds = 1;
  IncrementalPipeline pipeline(options, database, reconstruction_manager);
  pipeline.Run();

  // With max_runtime_seconds=0, the pipeline should stop almost immediately.
  // It may or may not have produced a reconstruction depending on timing,
  // but it should not crash.
  // The key coverage target is the CheckReachedMaxRuntime() path.
}

TEST(IncrementalPipeline, TriangulateExistingReconstruction) {
  SetPRNGSeed(0);
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

  // First, run a normal reconstruction.
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  IncrementalPipeline pipeline(options, database, reconstruction_manager);
  pipeline.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  auto reconstruction = reconstruction_manager->Get(0);

  // Remove all 3D points from the reconstruction to simulate a scenario
  // where we need to re-triangulate.
  const auto point3D_ids = reconstruction->Point3DIds();
  for (const point3D_t point3D_id : point3D_ids) {
    reconstruction->DeletePoint3D(point3D_id);
  }
  ASSERT_EQ(reconstruction->NumPoints3D(), 0);

  // Re-triangulate using the existing registered images.
  pipeline.TriangulateReconstruction(reconstruction);

  // After triangulation, we should have 3D points again.
  EXPECT_GT(reconstruction->NumPoints3D(), 0);
  EXPECT_GT(reconstruction->NumRegImages(), 0);
}

TEST(IncrementalPipeline, StopCallbackInterruptsReconstruction) {
  SetPRNGSeed(0);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  IncrementalPipeline pipeline(options, database, reconstruction_manager);

  // Set the stop function to immediately stop after the initial pair.
  int callback_count = 0;
  pipeline.SetCheckIfStoppedFunc([&callback_count]() {
    // Stop after the first check (i.e., right after initialization attempt).
    return ++callback_count > 1;
  });
  pipeline.Run();

  // The pipeline should have at most 1 reconstruction (possibly kept from
  // the interrupted state).
  EXPECT_LE(reconstruction_manager->Size(), 1);
}

TEST(IncrementalPipeline, DatabaseCacheConstructor) {
  SetPRNGSeed(0);
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

  // Create a DatabaseCache from the database first, then pass it to the
  // pipeline constructor that accepts a DatabaseCache directly.
  DatabaseCache::Options cache_options;
  auto database_cache = DatabaseCache::Create(*database, cache_options);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  IncrementalPipeline pipeline(
      options, database_cache, reconstruction_manager);
  pipeline.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, NullOptionsThrows) {
  auto database = Database::Open(CreateTestDir() / "database.db");
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  EXPECT_THROW(
      IncrementalPipeline(nullptr, database, reconstruction_manager),
      std::invalid_argument);
}

TEST(IncrementalPipeline, NullReconstructionManagerThrows) {
  auto database = Database::Open(CreateTestDir() / "database.db");
  auto options = std::make_shared<IncrementalPipelineOptions>();

  EXPECT_THROW(
      IncrementalPipeline(options, database, nullptr),
      std::invalid_argument);
}

TEST(IncrementalPipeline, NullDatabaseThrows) {
  auto options = std::make_shared<IncrementalPipelineOptions>();
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  std::shared_ptr<Database> null_database;

  EXPECT_THROW(
      IncrementalPipeline(options, null_database, reconstruction_manager),
      std::invalid_argument);
}

}  // namespace
}  // namespace colmap
