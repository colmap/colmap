// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/controllers/incremental_mapper.h"

#include "colmap/estimators/alignment.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

void ExpectEqualReconstructions(const Reconstruction& gt,
                                const Reconstruction& computed,
                                const double max_rotation_error_deg,
                                const double max_proj_center_error,
                                const double num_obs_tolerance) {
  EXPECT_EQ(computed.NumCameras(), gt.NumCameras());
  EXPECT_EQ(computed.NumImages(), gt.NumImages());
  EXPECT_EQ(computed.NumRegImages(), gt.NumRegImages());
  EXPECT_GE(computed.ComputeNumObservations(),
            (1 - num_obs_tolerance) * gt.ComputeNumObservations());

  Sim3d gtFromComputed;
  AlignReconstructionsViaProjCenters(computed,
                                     gt,
                                     /*max_proj_center_error=*/0.1,
                                     &gtFromComputed);

  const std::vector<ImageAlignmentError> errors =
      ComputeImageAlignmentError(computed, gt, gtFromComputed);
  EXPECT_EQ(errors.size(), gt.NumImages());
  for (const auto& error : errors) {
    EXPECT_LT(error.rotation_error_deg, max_rotation_error_deg);
    EXPECT_LT(error.proj_center_error, max_proj_center_error);
  }
}

TEST(IncrementalMapperController, WithoutNoise) {
  const std::string database_path = CreateTestDir() + "/database.db";

  Database database(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras = 2;
  synthetic_dataset_options.num_images = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.point2D_stddev = 0;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction, &database);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalMapperController mapper(
      std::make_shared<IncrementalPipelineOptions>(),
      /*image_path=*/"",
      database_path,
      reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  ExpectEqualReconstructions(gt_reconstruction,
                             *reconstruction_manager->Get(0),
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/1e-4,
                             /*num_obs_tolerance=*/0);
}

TEST(IncrementalMapperController, WithNoise) {
  const std::string database_path = CreateTestDir() + "/database.db";

  Database database(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras = 2;
  synthetic_dataset_options.num_images = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.point2D_stddev = 0.5;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction, &database);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalMapperController mapper(
      std::make_shared<IncrementalPipelineOptions>(),
      /*image_path=*/"",
      database_path,
      reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  ExpectEqualReconstructions(gt_reconstruction,
                             *reconstruction_manager->Get(0),
                             /*max_rotation_error_deg=*/1e-1,
                             /*max_proj_center_error=*/1e-1,
                             /*num_obs_tolerance=*/0.02);
}

TEST(IncrementalMapperController, MultiReconstruction) {
  const std::string database_path = CreateTestDir() + "/database.db";

  Database database(database_path);
  Reconstruction gt_reconstruction1;
  Reconstruction gt_reconstruction2;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras = 1;
  synthetic_dataset_options.num_images = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.point2D_stddev = 0;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction1, &database);
  synthetic_dataset_options.num_images = 4;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction2, &database);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  auto mapper_options = std::make_shared<IncrementalPipelineOptions>();
  mapper_options->min_model_size = 4;
  IncrementalMapperController mapper(mapper_options,
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
  ExpectEqualReconstructions(gt_reconstruction1,
                             *computed_reconstruction1,
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/1e-4,
                             /*num_obs_tolerance=*/0);
  ExpectEqualReconstructions(gt_reconstruction2,
                             *computed_reconstruction2,
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/1e-4,
                             /*num_obs_tolerance=*/0);
}

TEST(IncrementalMapperController, ChainedMatches) {
  const std::string database_path = CreateTestDir() + "/database.db";

  Database database(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.match_config =
      SyntheticDatasetOptions::MatchConfig::CHAINED;
  synthetic_dataset_options.num_cameras = 1;
  synthetic_dataset_options.num_images = 4;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.point2D_stddev = 0;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction, &database);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalMapperController mapper(
      std::make_shared<IncrementalPipelineOptions>(),
      /*image_path=*/"",
      database_path,
      reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  ExpectEqualReconstructions(gt_reconstruction,
                             *reconstruction_manager->Get(0),
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/1e-4,
                             /*num_obs_tolerance=*/0);
}

}  // namespace
}  // namespace colmap
