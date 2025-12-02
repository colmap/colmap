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

#include "colmap/estimators/alignment.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(IncrementalPipeline, WithoutNoise) {
  const std::string database_path = CreateTestDir() + "/database.db";

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
                             /*image_path=*/"",
                             database_path,
                             reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, WithoutNoiseAndWithNonTrivialFrames) {
  const std::string database_path = CreateTestDir() + "/database.db";

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
    IncrementalPipeline mapper(options,
                               /*image_path=*/"",
                               database_path,
                               reconstruction_manager);
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

TEST(IncrementalPipeline, WithNonTrivialFramesAndConstantRigsAndCameras) {
  const std::string database_path = CreateTestDir() + "/database.db";

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
  IncrementalPipeline mapper(options,
                             /*image_path=*/"",
                             database_path,
                             reconstruction_manager);
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
  const std::string database_path = CreateTestDir() + "/database.db";

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
    IncrementalPipeline mapper(options,
                               /*image_path=*/"",
                               database_path,
                               reconstruction_manager);
    mapper.Run();

    ASSERT_EQ(reconstruction_manager->Size(), 1);
    EXPECT_THAT(gt_reconstruction,
                ReconstructionNear(*reconstruction_manager->Get(0),
                                   /*max_rotation_error_deg=*/1e-2,
                                   /*max_proj_center_error=*/1e-3));
  }
}

TEST(IncrementalPipeline, WithPriorFocalLength) {
  const std::string database_path = CreateTestDir() + "/database.db";

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
                             /*image_path=*/"",
                             database_path,
                             reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, WithNoise) {
  const std::string database_path = CreateTestDir() + "/database.db";

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
                             /*image_path=*/"",
                             database_path,
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
  const std::string database_path = CreateTestDir() + "/database.db";

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
  IncrementalPipeline mapper(options,
                             /*image_path=*/"",
                             database_path,
                             reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, StructureLessRegistrationOnly) {
  const std::string database_path = CreateTestDir() + "/database.db";

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
  IncrementalPipeline mapper(options,
                             /*image_path=*/"",
                             database_path,
                             reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-3,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, MultiReconstruction) {
  const std::string database_path = CreateTestDir() + "/database.db";

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
  IncrementalPipeline mapper(mapper_options,
                             /*image_path=*/"",
                             database_path,
                             reconstruction_manager);
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
  const std::string database_path = CreateTestDir() + "/database.db";

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
    IncrementalPipeline mapper(options,
                               /*image_path=*/"",
                               database_path,
                               reconstruction_manager);
    mapper.Run();

    ASSERT_EQ(reconstruction_manager->Size(), 1);
    EXPECT_THAT(gt_reconstruction,
                ReconstructionNear(*reconstruction_manager->Get(0),
                                   /*max_rotation_error_deg=*/1e-2,
                                   /*max_proj_center_error=*/1e-4));
  }
}

TEST(IncrementalPipeline, ChainedMatches) {
  const std::string database_path = CreateTestDir() + "/database.db";

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
                             /*image_path=*/"",
                             database_path,
                             reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(IncrementalPipeline, PriorBasedSfMWithoutNoise) {
  const std::string database_path = CreateTestDir() + "/database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.use_prior_position = true;
  synthetic_dataset_options.prior_position_stddev = 0.0;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  std::shared_ptr<IncrementalPipelineOptions> mapper_options =
      std::make_shared<IncrementalPipelineOptions>();
  mapper_options->use_prior_position = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(mapper_options,
                             /*image_path=*/"",
                             database_path,
                             reconstruction_manager);
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
  const std::string database_path = CreateTestDir() + "/database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_has_prior_focal_length = false;

  synthetic_dataset_options.use_prior_position = true;
  synthetic_dataset_options.prior_position_stddev = 0.0;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  std::shared_ptr<IncrementalPipelineOptions> mapper_options =
      std::make_shared<IncrementalPipelineOptions>();

  mapper_options->use_prior_position = true;
  mapper_options->use_robust_loss_on_prior_position = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(mapper_options,
                             /*image_path=*/"",
                             database_path,
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

TEST(IncrementalPipeline, PriorBasedSfMWithNoise) {
  const std::string database_path = CreateTestDir() + "/database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.use_prior_position = true;
  synthetic_dataset_options.prior_position_stddev = 1.5;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  std::shared_ptr<IncrementalPipelineOptions> mapper_options =
      std::make_shared<IncrementalPipelineOptions>();

  mapper_options->use_prior_position = true;
  mapper_options->use_robust_loss_on_prior_position = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(mapper_options,
                             /*image_path=*/"",
                             database_path,
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

TEST(IncrementalPipeline, GPSPriorBasedSfMWithNoise) {
  const std::string database_path = CreateTestDir() + "/database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 100;

  synthetic_dataset_options.use_prior_position = true;
  synthetic_dataset_options.use_wgs84_prior = true;
  synthetic_dataset_options.prior_position_stddev = 1.5;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  std::shared_ptr<IncrementalPipelineOptions> mapper_options =
      std::make_shared<IncrementalPipelineOptions>();

  mapper_options->use_prior_position = true;
  mapper_options->use_robust_loss_on_prior_position = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(mapper_options,
                             /*image_path=*/"",
                             database_path,
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

TEST(IncrementalPipeline, SfMWithRandomSeedStability) {
  SetPRNGSeed(1);

  const std::string database_path = CreateTestDir() + "/database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.use_prior_position = false;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto run_mapper = [&](int num_threads, int random_seed) {
    auto mapper_options = std::make_shared<IncrementalPipelineOptions>();
    mapper_options->use_prior_position = false;
    mapper_options->num_threads = num_threads;
    mapper_options->random_seed = random_seed;
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    IncrementalPipeline mapper(mapper_options,
                               /*image_path=*/"",
                               database_path,
                               reconstruction_manager);
    mapper.Run();
    EXPECT_EQ(reconstruction_manager->Size(), 1);
    return reconstruction_manager;
  };

  constexpr int kRandomSeed = 42;

  // Single-threaded execution.
  {
    auto reconstruction_manager0 =
        run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
    auto reconstruction_manager1 =
        run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
    EXPECT_THAT(*reconstruction_manager0->Get(0),
                ReconstructionEq(*reconstruction_manager1->Get(0)));
  }

  // Multi-threaded execution.
  {
    auto reconstruction_manager0 =
        run_mapper(/*num_threads=*/3, /*random_seed=*/kRandomSeed);
    auto reconstruction_manager1 =
        run_mapper(/*num_threads=*/3, /*random_seed=*/kRandomSeed);
    // Same seed should produce similar results, up to floating-point variations
    // in optimization.
    EXPECT_THAT(*reconstruction_manager0->Get(0),
                ReconstructionNear(*reconstruction_manager1->Get(0),
                                   /*max_rotation_error_deg=*/1e-10,
                                   /*max_proj_center_error=*/1e-10,
                                   /*max_scale_error=*/std::nullopt,
                                   /*num_obs_tolerance=*/0.01,
                                   /*align=*/false));
  }
}

TEST(IncrementalPipeline, PriorBasedSfMWithRandomSeedStability) {
  SetPRNGSeed(1);

  const std::string database_path = CreateTestDir() + "/database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.use_prior_position = true;
  synthetic_dataset_options.prior_position_stddev = 1.0;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto run_mapper = [&](int num_threads, int random_seed) {
    auto mapper_options = std::make_shared<IncrementalPipelineOptions>();
    mapper_options->use_prior_position = true;
    mapper_options->num_threads = num_threads;
    mapper_options->random_seed = random_seed;
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    IncrementalPipeline mapper(mapper_options,
                               /*image_path=*/"",
                               database_path,
                               reconstruction_manager);
    mapper.Run();
    EXPECT_EQ(reconstruction_manager->Size(), 1);
    return reconstruction_manager;
  };

  constexpr int kRandomSeed = 42;

  // Single-threaded execution.
  {
    auto reconstruction_manager0 =
        run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
    auto reconstruction_manager1 =
        run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
    EXPECT_THAT(*reconstruction_manager0->Get(0),
                ReconstructionEq(*reconstruction_manager1->Get(0)));
  }

  // Multi-threaded execution.
  {
    auto reconstruction_manager0 =
        run_mapper(/*num_threads=*/3, /*random_seed=*/kRandomSeed);
    auto reconstruction_manager1 =
        run_mapper(/*num_threads=*/3, /*random_seed=*/kRandomSeed);
    // Same seed should produce similar results, up to floating-point variations
    // in optimization.
    EXPECT_THAT(*reconstruction_manager0->Get(0),
                ReconstructionNear(*reconstruction_manager1->Get(0),
                                   /*max_rotation_error_deg=*/1e-10,
                                   /*max_proj_center_error=*/1e-10,
                                   /*max_scale_error=*/std::nullopt,
                                   /*num_obs_tolerance=*/0.01,
                                   /*align=*/false));
  }
}

}  // namespace
}  // namespace colmap
