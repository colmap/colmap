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

#include "colmap/sfm/incremental_mapper.h"

#include "colmap/scene/database_cache.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

std::shared_ptr<DatabaseCache> CreateDatabaseCache(const Database& database) {
  DatabaseCache::Options options;
  return DatabaseCache::Create(database, options);
}

SyntheticDatasetOptions DefaultSyntheticOptions() {
  SyntheticDatasetOptions options;
  options.num_rigs = 2;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 5;
  options.num_points3D = 50;
  return options;
}

TEST(IncrementalMapper, GettersAfterBeginReconstruction) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction gt_reconstruction;
  auto synthetic_options = DefaultSyntheticOptions();
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  auto cache = CreateDatabaseCache(*database);
  IncrementalMapper mapper(cache);

  auto reconstruction = std::make_shared<Reconstruction>();
  mapper.BeginReconstruction(reconstruction);

  EXPECT_NE(mapper.Reconstruction(), nullptr);
  EXPECT_EQ(mapper.Reconstruction().get(), reconstruction.get());
  EXPECT_NO_THROW(mapper.Triangulator());
  EXPECT_NO_THROW(mapper.ObservationManager());
  EXPECT_TRUE(mapper.FilteredFrames().empty());
  EXPECT_TRUE(mapper.ExistingFrameIds().empty());
  EXPECT_EQ(mapper.NumTotalRegImages(), 0);
  EXPECT_EQ(mapper.NumSharedRegImages(), 0);

  mapper.EndReconstruction(/*discard=*/false);
}

TEST(IncrementalMapper, EndReconstructionDiscard) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction gt_reconstruction;
  auto synthetic_options = DefaultSyntheticOptions();
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  auto cache = CreateDatabaseCache(*database);
  IncrementalMapper mapper(cache);

  auto reconstruction = std::make_shared<Reconstruction>();
  mapper.BeginReconstruction(reconstruction);

  IncrementalMapper::Options options;
  options.init_min_num_inliers = 10;
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  Rigid3d cam2_from_cam1;
  EXPECT_TRUE(mapper.FindInitialImagePair(
      options, image_id1, image_id2, cam2_from_cam1));
  mapper.RegisterInitialImagePair(
      options, image_id1, image_id2, cam2_from_cam1);
  EXPECT_GT(mapper.NumTotalRegImages(), 0);

  // Discard the reconstruction
  mapper.EndReconstruction(/*discard=*/true);
  EXPECT_EQ(mapper.NumTotalRegImages(), 0);
}

TEST(IncrementalMapper, EstimateInitialTwoViewGeometry) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction gt_reconstruction;
  auto synthetic_options = DefaultSyntheticOptions();
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  auto cache = CreateDatabaseCache(*database);
  IncrementalMapper mapper(cache);

  auto reconstruction = std::make_shared<Reconstruction>();
  mapper.BeginReconstruction(reconstruction);

  // Get two images that should have valid geometry
  const auto& images = cache->Images();
  ASSERT_GE(images.size(), 2);
  auto it = images.begin();
  const image_t image_id1 = it->first;
  ++it;
  const image_t image_id2 = it->first;

  IncrementalMapper::Options options;
  options.init_min_num_inliers = 10;
  Rigid3d cam2_from_cam1;
  // The synthetic dataset should yield a valid two-view geometry
  EXPECT_TRUE(mapper.EstimateInitialTwoViewGeometry(
      options, image_id1, image_id2, cam2_from_cam1));

  mapper.EndReconstruction(/*discard=*/false);
}

TEST(IncrementalMapper, ModifiedPoints3D) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction gt_reconstruction;
  auto synthetic_options = DefaultSyntheticOptions();
  synthetic_options.num_points3D = 100;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  auto cache = CreateDatabaseCache(*database);
  IncrementalMapper mapper(cache);

  auto reconstruction = std::make_shared<Reconstruction>();
  mapper.BeginReconstruction(reconstruction);

  // Initially no modified points
  mapper.ClearModifiedPoints3D();
  EXPECT_TRUE(mapper.GetModifiedPoints3D().empty());

  // Register an initial pair and triangulate to produce modified points
  IncrementalMapper::Options options;
  options.init_min_num_inliers = 10;
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  Rigid3d cam2_from_cam1;
  ASSERT_TRUE(mapper.FindInitialImagePair(
      options, image_id1, image_id2, cam2_from_cam1));
  mapper.RegisterInitialImagePair(
      options, image_id1, image_id2, cam2_from_cam1);

  IncrementalTriangulator::Options tri_options;
  mapper.TriangulateImage(tri_options, image_id1);
  mapper.TriangulateImage(tri_options, image_id2);

  // After triangulation, modified points should be non-empty
  EXPECT_FALSE(mapper.GetModifiedPoints3D().empty());

  // After clearing, modified points should be empty again
  mapper.ClearModifiedPoints3D();
  EXPECT_TRUE(mapper.GetModifiedPoints3D().empty());

  mapper.EndReconstruction(/*discard=*/false);
}

TEST(IncrementalMapper, FullPipeline) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction gt_reconstruction;
  auto synthetic_options = DefaultSyntheticOptions();
  synthetic_options.num_points3D = 100;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  auto cache = CreateDatabaseCache(*database);
  IncrementalMapper mapper(cache);

  auto reconstruction = std::make_shared<Reconstruction>();
  mapper.BeginReconstruction(reconstruction);

  IncrementalMapper::Options options;
  options.init_min_num_inliers = 10;
  options.abs_pose_min_num_inliers = 10;
  options.abs_pose_min_inlier_ratio = 0.1;

  // Step 1: Find and register initial image pair
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  Rigid3d cam2_from_cam1;
  ASSERT_TRUE(mapper.FindInitialImagePair(
      options, image_id1, image_id2, cam2_from_cam1));
  EXPECT_NE(image_id1, kInvalidImageId);
  EXPECT_NE(image_id2, kInvalidImageId);

  mapper.RegisterInitialImagePair(
      options, image_id1, image_id2, cam2_from_cam1);
  EXPECT_GE(reconstruction->NumRegFrames(), 2);

  // Step 2: Triangulate initial observations
  IncrementalTriangulator::Options tri_options;
  mapper.TriangulateImage(tri_options, image_id1);
  mapper.TriangulateImage(tri_options, image_id2);
  EXPECT_GT(reconstruction->NumPoints3D(), 0);

  // Step 3: Find and register next images
  const auto next_image_ids = mapper.FindNextImages(options);
  EXPECT_FALSE(next_image_ids.empty());

  for (const auto image_id : next_image_ids) {
    if (mapper.RegisterNextImage(options, image_id)) {
      mapper.TriangulateImage(tri_options, image_id);
    }
  }
  EXPECT_GT(reconstruction->NumRegFrames(), 2);

  // Step 4: Global bundle adjustment
  BundleAdjustmentOptions ba_options;
  EXPECT_TRUE(mapper.AdjustGlobalBundle(options, ba_options));

  // Step 5: Filtering
  mapper.FilterPoints(options);
  mapper.FilterFrames(options);

  // Step 6: Track completion and merging
  mapper.CompleteAndMergeTracks(tri_options);

  // Verify the reconstruction is reasonable
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1));

  // Verify registration stats are consistent
  const auto& num_per_rig = mapper.NumRegFramesPerRig();
  EXPECT_FALSE(num_per_rig.empty());
  const auto& num_per_camera = mapper.NumRegImagesPerCamera();
  EXPECT_FALSE(num_per_camera.empty());

  mapper.EndReconstruction(/*discard=*/false);
}

TEST(IncrementalMapper, FindLocalBundle) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction gt_reconstruction;
  auto synthetic_options = DefaultSyntheticOptions();
  synthetic_options.num_points3D = 100;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  auto cache = CreateDatabaseCache(*database);
  IncrementalMapper mapper(cache);

  auto reconstruction = std::make_shared<Reconstruction>();
  mapper.BeginReconstruction(reconstruction);

  IncrementalMapper::Options options;
  options.init_min_num_inliers = 10;
  options.abs_pose_min_num_inliers = 10;
  options.abs_pose_min_inlier_ratio = 0.1;

  // Register initial pair and at least one more image to have a local bundle
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  Rigid3d cam2_from_cam1;
  ASSERT_TRUE(mapper.FindInitialImagePair(
      options, image_id1, image_id2, cam2_from_cam1));
  mapper.RegisterInitialImagePair(
      options, image_id1, image_id2, cam2_from_cam1);

  IncrementalTriangulator::Options tri_options;
  mapper.TriangulateImage(tri_options, image_id1);
  mapper.TriangulateImage(tri_options, image_id2);

  // Register more images
  const auto next_image_ids = mapper.FindNextImages(options);
  for (const auto image_id : next_image_ids) {
    if (mapper.RegisterNextImage(options, image_id)) {
      mapper.TriangulateImage(tri_options, image_id);
    }
  }

  if (reconstruction->NumRegFrames() >= 3) {
    const auto local_bundle = mapper.FindLocalBundle(options, image_id1);
    EXPECT_FALSE(local_bundle.empty());
  }

  mapper.EndReconstruction(/*discard=*/false);
}

TEST(IncrementalMapper, ResetInitializationStats) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction gt_reconstruction;
  auto synthetic_options = DefaultSyntheticOptions();
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  auto cache = CreateDatabaseCache(*database);
  IncrementalMapper mapper(cache);

  auto reconstruction = std::make_shared<Reconstruction>();
  mapper.BeginReconstruction(reconstruction);

  IncrementalMapper::Options options;
  options.init_min_num_inliers = 10;

  // Try to find an initial pair (updates init stats internally)
  image_t image_id1, image_id2;
  Rigid3d cam2_from_cam1;
  mapper.FindInitialImagePair(options, image_id1, image_id2, cam2_from_cam1);

  // Reset stats and try again — should not throw
  mapper.ResetInitializationStats();
  EXPECT_NO_THROW(mapper.FindInitialImagePair(
      options, image_id1, image_id2, cam2_from_cam1));

  mapper.EndReconstruction(/*discard=*/false);
}

}  // namespace
}  // namespace colmap
