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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/controllers/incremental_mapper.h"

#include "colmap/base/synthetic.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {

TEST(SynthesizeDataset, NoNoiseNoOutliers) {
  const std::string database_path = CreateTestDir() + "/database.db";

  Database database(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions options;
  options.point2D_stddev = 0;
  options.num_outlier_matches_per_pair = 0;
  SynthesizeDataset(options, &gt_reconstruction, &database);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalMapperController mapper(
      std::make_shared<IncrementalMapperOptions>(),
      /*image_path=*/"",
      database_path,
      reconstruction_manager);
  mapper.Start();
  mapper.Wait();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  const auto& reconstruction = reconstruction_manager->Get(0);
  EXPECT_EQ(reconstruction->NumCameras(), gt_reconstruction.NumCameras());
  EXPECT_EQ(reconstruction->NumImages(), gt_reconstruction.NumImages());
  EXPECT_EQ(reconstruction->NumRegImages(), gt_reconstruction.NumRegImages());
  EXPECT_NEAR(reconstruction->ComputeMeanReprojectionError(), 0, 1e-3);
  EXPECT_EQ(reconstruction->ComputeNumObservations(),
            gt_reconstruction.ComputeNumObservations());

  SimilarityTransform3 gtFromComputed;
  ComputeAlignmentBetweenReconstructions(*reconstruction,
                                         gt_reconstruction,
                                         /*max_proj_center_error=*/0.1,
                                         &gtFromComputed);

  const std::vector<ImageAlignmentError> errors = ComputeImageAlignmentError(
      *reconstruction, gt_reconstruction, gtFromComputed);
  EXPECT_EQ(errors.size(), gt_reconstruction.NumImages());
  for (const auto& error : errors) {
    EXPECT_LT(error.rotation_error_deg, 1e-2);
    EXPECT_LT(error.proj_center_error, 1e-4);
  }
}

TEST(SynthesizeDataset, NoiseNoOutliers) {
  const std::string database_path = CreateTestDir() + "/database.db";

  Database database(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions options;
  options.point2D_stddev = 1.0;
  options.num_outlier_matches_per_pair = 0;
  SynthesizeDataset(options, &gt_reconstruction, &database);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalMapperController mapper(
      std::make_shared<IncrementalMapperOptions>(),
      /*image_path=*/"",
      database_path,
      reconstruction_manager);
  mapper.Start();
  mapper.Wait();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  const auto& reconstruction = reconstruction_manager->Get(0);
  EXPECT_EQ(reconstruction->NumCameras(), gt_reconstruction.NumCameras());
  EXPECT_EQ(reconstruction->NumImages(), gt_reconstruction.NumImages());
  EXPECT_EQ(reconstruction->NumRegImages(), gt_reconstruction.NumRegImages());
  EXPECT_EQ(reconstruction->ComputeNumObservations(),
            gt_reconstruction.ComputeNumObservations());

  SimilarityTransform3 gtFromComputed;
  ComputeAlignmentBetweenReconstructions(*reconstruction,
                                         gt_reconstruction,
                                         /*max_proj_center_error=*/0.1,
                                         &gtFromComputed);

  const std::vector<ImageAlignmentError> errors = ComputeImageAlignmentError(
      *reconstruction, gt_reconstruction, gtFromComputed);
  EXPECT_EQ(errors.size(), gt_reconstruction.NumImages());
  for (const auto& error : errors) {
    EXPECT_LT(error.rotation_error_deg, 1e-1);
    EXPECT_LT(error.proj_center_error, 1e-2);
  }
}

}  // namespace colmap
