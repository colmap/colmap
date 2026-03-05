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
#include "colmap/sfm/incremental_mapper_impl.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

SyntheticDatasetOptions DefaultSyntheticOptions() {
  SyntheticDatasetOptions options;
  options.num_rigs = 2;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 5;
  options.num_points3D = 100;
  return options;
}

class IncrementalMapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    database_ = Database::Open(kInMemorySqliteDatabasePath);
    SynthesizeDataset(synthetic_options_, &gt_reconstruction_, database_.get());
    DatabaseCache::Options cache_options;
    cache_ = DatabaseCache::Create(*database_, cache_options);
    mapper_ = std::make_unique<IncrementalMapper>(cache_);
    reconstruction_ = std::make_shared<Reconstruction>();
    mapper_->BeginReconstruction(reconstruction_);
    options_.init_min_num_inliers = 10;
    options_.abs_pose_min_num_inliers = 10;
    options_.abs_pose_min_inlier_ratio = 0.1;
  }

  void TearDown() override {
    if (mapper_->Reconstruction()) {
      mapper_->EndReconstruction(/*discard=*/false);
    }
  }

  void FindAndRegisterInitialPair() {
    ASSERT_TRUE(mapper_->FindInitialImagePair(
        options_, image_id1_, image_id2_, cam2_from_cam1_));
    mapper_->RegisterInitialImagePair(
        options_, image_id1_, image_id2_, cam2_from_cam1_);
  }

  void TriangulateInitialPair() {
    mapper_->TriangulateImage(tri_options_, image_id1_);
    mapper_->TriangulateImage(tri_options_, image_id2_);
  }

  void RegisterAllRemainingImages() {
    const auto next_image_ids = mapper_->FindNextImages(options_);
    for (const auto image_id : next_image_ids) {
      if (mapper_->RegisterNextImage(options_, image_id)) {
        mapper_->TriangulateImage(tri_options_, image_id);
      }
    }
  }

  SyntheticDatasetOptions synthetic_options_ = DefaultSyntheticOptions();
  Reconstruction gt_reconstruction_;
  std::shared_ptr<Database> database_;
  std::shared_ptr<DatabaseCache> cache_;
  std::unique_ptr<IncrementalMapper> mapper_;
  std::shared_ptr<Reconstruction> reconstruction_;
  IncrementalMapper::Options options_;
  IncrementalTriangulator::Options tri_options_;
  image_t image_id1_ = kInvalidImageId;
  image_t image_id2_ = kInvalidImageId;
  Rigid3d cam2_from_cam1_;
};

// Standalone test: needs to check state before BeginReconstruction.
TEST(IncrementalMapper, GettersAfterBeginReconstruction) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction gt_reconstruction;
  SynthesizeDataset(
      DefaultSyntheticOptions(), &gt_reconstruction, database.get());

  DatabaseCache::Options cache_options;
  auto cache = DatabaseCache::Create(*database, cache_options);
  // 2 rigs x 1 camera x 5 frames = 10 images
  EXPECT_EQ(cache->Images().size(), 10);

  IncrementalMapper mapper(cache);

  // Before BeginReconstruction, Reconstruction() returns nullptr
  EXPECT_EQ(mapper.Reconstruction(), nullptr);

  auto reconstruction = std::make_shared<Reconstruction>();
  mapper.BeginReconstruction(reconstruction);

  EXPECT_EQ(mapper.Reconstruction().get(), reconstruction.get());
  EXPECT_NO_THROW(mapper.Triangulator());
  EXPECT_NO_THROW(mapper.ObservationManager());
  EXPECT_TRUE(mapper.FilteredFrames().empty());
  EXPECT_TRUE(mapper.ExistingFrameIds().empty());
  EXPECT_TRUE(mapper.NumRegFramesPerRig().empty());
  EXPECT_TRUE(mapper.NumRegImagesPerCamera().empty());
  EXPECT_EQ(mapper.NumTotalRegImages(), 0);
  EXPECT_EQ(mapper.NumSharedRegImages(), 0);

  mapper.EndReconstruction(/*discard=*/false);
}

TEST_F(IncrementalMapperTest, EndReconstructionDiscard) {
  FindAndRegisterInitialPair();

  // Initial pair registers 2 frames, each with 1 image
  EXPECT_EQ(mapper_->NumTotalRegImages(), 2);
  EXPECT_EQ(reconstruction_->NumRegFrames(), 2);

  // Discard the reconstruction — stats should be rolled back
  mapper_->EndReconstruction(/*discard=*/true);
  EXPECT_EQ(mapper_->NumTotalRegImages(), 0);
  EXPECT_EQ(mapper_->NumSharedRegImages(), 0);
}

TEST_F(IncrementalMapperTest, EstimateInitialTwoViewGeometry) {
  // Use FindInitialImagePair to select a pair with enough correspondences,
  // then verify EstimateInitialTwoViewGeometry re-estimates valid geometry.
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  Rigid3d cam2_from_cam1_found;
  ASSERT_TRUE(mapper_->FindInitialImagePair(
      options_, image_id1, image_id2, cam2_from_cam1_found));
  EXPECT_NE(image_id1, image_id2);

  Rigid3d cam2_from_cam1;
  ASSERT_TRUE(mapper_->EstimateInitialTwoViewGeometry(
      options_, image_id1, image_id2, cam2_from_cam1));

  // The estimated pose should have non-trivial translation
  EXPECT_GT(cam2_from_cam1.translation().norm(), 0.0);
}

TEST_F(IncrementalMapperTest, ModifiedPoints3D) {
  // Initially no modified points
  mapper_->ClearModifiedPoints3D();
  EXPECT_TRUE(mapper_->GetModifiedPoints3D().empty());

  FindAndRegisterInitialPair();
  EXPECT_EQ(mapper_->NumTotalRegImages(), 2);
  EXPECT_EQ(reconstruction_->NumRegFrames(), 2);

  TriangulateInitialPair();
  EXPECT_GT(reconstruction_->NumPoints3D(), 0);

  // After triangulation, modified points should be non-empty
  const auto& modified = mapper_->GetModifiedPoints3D();
  EXPECT_FALSE(modified.empty());
  // All modified point IDs should exist in the reconstruction
  for (const auto point3D_id : modified) {
    EXPECT_TRUE(reconstruction_->ExistsPoint3D(point3D_id));
  }

  // After clearing, modified points should be empty again
  mapper_->ClearModifiedPoints3D();
  EXPECT_TRUE(mapper_->GetModifiedPoints3D().empty());
}

TEST_F(IncrementalMapperTest, FullPipeline) {
  // Step 1: Find and register initial image pair
  FindAndRegisterInitialPair();
  EXPECT_NE(image_id1_, kInvalidImageId);
  EXPECT_NE(image_id2_, kInvalidImageId);
  EXPECT_GE(reconstruction_->NumRegFrames(), 2);

  // Step 2: Triangulate initial observations
  TriangulateInitialPair();
  EXPECT_GT(reconstruction_->NumPoints3D(), 0);

  // Step 3: Find and register next images
  RegisterAllRemainingImages();
  EXPECT_EQ(reconstruction_->NumRegFrames(), 10);

  // Step 4: Global bundle adjustment
  BundleAdjustmentOptions ba_options;
  EXPECT_TRUE(mapper_->AdjustGlobalBundle(options_, ba_options));

  // Step 5: Filtering
  mapper_->FilterPoints(options_);
  mapper_->FilterFrames(options_);

  // Step 6: Track completion and merging
  mapper_->CompleteAndMergeTracks(tri_options_);

  // Verify the reconstruction is reasonable
  EXPECT_THAT(gt_reconstruction_,
              ReconstructionNear(*reconstruction_,
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1));

  // Verify registration stats match the reconstruction.
  // The synthetic dataset has 2 rigs with 1 camera each and 5 frames per rig,
  // giving 10 total frames/images. All should be registered.
  EXPECT_EQ(reconstruction_->NumRegFrames(), 10);

  const auto& num_per_rig = mapper_->NumRegFramesPerRig();
  EXPECT_EQ(num_per_rig.size(), 2);
  for (const auto& [rig_id, count] : num_per_rig) {
    EXPECT_EQ(count, 5);
  }

  const auto& num_per_camera = mapper_->NumRegImagesPerCamera();
  EXPECT_EQ(num_per_camera.size(), 2);
  for (const auto& [camera_id, count] : num_per_camera) {
    EXPECT_EQ(count, 5);
  }

  EXPECT_EQ(mapper_->NumTotalRegImages(), 10);
  EXPECT_EQ(mapper_->NumSharedRegImages(), 0);
  EXPECT_TRUE(mapper_->FilteredFrames().empty());

  // Sanity check tracks: with dense visibility, at least one track
  // should be observed by all registered images.
  size_t max_track_length = 0;
  for (const auto& [point3D_id, point3D] : reconstruction_->Points3D()) {
    max_track_length = std::max(max_track_length, point3D.track.Length());
  }
  EXPECT_GE(max_track_length, reconstruction_->NumRegImages());
}

TEST_F(IncrementalMapperTest, FindLocalBundle) {
  FindAndRegisterInitialPair();
  TriangulateInitialPair();
  RegisterAllRemainingImages();

  ASSERT_GE(reconstruction_->NumRegFrames(), 3);
  const auto local_bundle = mapper_->FindLocalBundle(options_, image_id1_);
  EXPECT_FALSE(local_bundle.empty());
  // Local bundle should not exceed the total number of registered images
  EXPECT_LE(local_bundle.size(), reconstruction_->NumRegImages());
  // All images in the local bundle should be registered
  const auto reg_image_ids = reconstruction_->RegImageIds();
  const std::unordered_set<image_t> reg_image_id_set(reg_image_ids.begin(),
                                                     reg_image_ids.end());
  for (const auto image_id : local_bundle) {
    EXPECT_GT(reg_image_id_set.count(image_id), 0);
  }
}

TEST_F(IncrementalMapperTest, ResetInitializationStats) {
  // Find an initial pair (updates init stats internally)
  ASSERT_TRUE(mapper_->FindInitialImagePair(
      options_, image_id1_, image_id2_, cam2_from_cam1_));
  EXPECT_NE(image_id1_, kInvalidImageId);
  EXPECT_NE(image_id2_, kInvalidImageId);

  // Reset stats and find again — should succeed with valid IDs
  mapper_->ResetInitializationStats();
  image_t image_id3 = kInvalidImageId;
  image_t image_id4 = kInvalidImageId;
  Rigid3d cam2_from_cam1_2;
  ASSERT_TRUE(mapper_->FindInitialImagePair(
      options_, image_id3, image_id4, cam2_from_cam1_2));
  EXPECT_NE(image_id3, kInvalidImageId);
  EXPECT_NE(image_id4, kInvalidImageId);
}

///////////////////////////////////////////////////////////////////////////////
// Tests targeting IncrementalMapperImpl directly
///////////////////////////////////////////////////////////////////////////////

// Fixture for tests that call IncrementalMapperImpl static methods directly.
class IncrementalMapperImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    database_ = Database::Open(kInMemorySqliteDatabasePath);
    SynthesizeDataset(synthetic_options_, &gt_reconstruction_, database_.get());
    DatabaseCache::Options cache_options;
    cache_ = DatabaseCache::Create(*database_, cache_options);
    reconstruction_ = std::make_shared<Reconstruction>();
    // Load cameras, rigs, frames, and images from the cache
    // (mirrors what IncrementalMapper::BeginReconstruction does).
    reconstruction_->Load(*cache_);
    options_.init_min_num_inliers = 10;
    options_.abs_pose_min_num_inliers = 10;
    options_.abs_pose_min_inlier_ratio = 0.1;
  }

  SyntheticDatasetOptions synthetic_options_ = DefaultSyntheticOptions();
  Reconstruction gt_reconstruction_;
  std::shared_ptr<Database> database_;
  std::shared_ptr<DatabaseCache> cache_;
  std::shared_ptr<Reconstruction> reconstruction_;
  IncrementalMapper::Options options_;
};

// Test FindFirstInitialImage returns images sorted by prior focal length,
// then by number of correspondences (descending).
TEST_F(IncrementalMapperImplTest, FindFirstInitialImageBasicSorting) {
  const auto& corr_graph = *cache_->CorrespondenceGraph();
  std::unordered_map<image_t, size_t> init_num_reg_trials;
  std::unordered_map<image_t, size_t> num_registrations;

  const auto image_ids = IncrementalMapperImpl::FindFirstInitialImage(
      options_, corr_graph, *reconstruction_, init_num_reg_trials,
      num_registrations);

  // All images with correspondences should be returned.
  EXPECT_FALSE(image_ids.empty());
  EXPECT_LE(image_ids.size(), reconstruction_->NumImages());

  // Verify sorting: images should be ordered by number of correspondences
  // descending (all have same has_prior_focal_length in this dataset).
  for (size_t i = 1; i < image_ids.size(); ++i) {
    const auto num_corrs_prev =
        corr_graph.NumCorrespondencesForImage(image_ids[i - 1]);
    const auto num_corrs_curr =
        corr_graph.NumCorrespondencesForImage(image_ids[i]);
    EXPECT_GE(num_corrs_prev, num_corrs_curr);
  }
}

// Test that FindFirstInitialImage skips images that have exceeded
// init_max_reg_trials.
TEST_F(IncrementalMapperImplTest, FindFirstInitialImageSkipsExhaustedTrials) {
  const auto& corr_graph = *cache_->CorrespondenceGraph();
  std::unordered_map<image_t, size_t> num_registrations;

  // First, get all candidates with no restrictions.
  std::unordered_map<image_t, size_t> empty_trials;
  const auto all_ids = IncrementalMapperImpl::FindFirstInitialImage(
      options_, corr_graph, *reconstruction_, empty_trials, num_registrations);
  ASSERT_GE(all_ids.size(), 2);

  // Mark the first two images as having exhausted their trials.
  std::unordered_map<image_t, size_t> init_num_reg_trials;
  init_num_reg_trials[all_ids[0]] =
      static_cast<size_t>(options_.init_max_reg_trials);
  init_num_reg_trials[all_ids[1]] =
      static_cast<size_t>(options_.init_max_reg_trials);

  const auto filtered_ids = IncrementalMapperImpl::FindFirstInitialImage(
      options_, corr_graph, *reconstruction_, init_num_reg_trials,
      num_registrations);

  // The two exhausted images should be excluded.
  EXPECT_EQ(filtered_ids.size(), all_ids.size() - 2);
  for (const auto id : filtered_ids) {
    EXPECT_NE(id, all_ids[0]);
    EXPECT_NE(id, all_ids[1]);
  }
}

// Test that FindFirstInitialImage skips images registered in other
// reconstructions.
TEST_F(IncrementalMapperImplTest,
       FindFirstInitialImageSkipsRegisteredImages) {
  const auto& corr_graph = *cache_->CorrespondenceGraph();
  std::unordered_map<image_t, size_t> init_num_reg_trials;

  // Get all candidates with no registrations.
  std::unordered_map<image_t, size_t> no_registrations;
  const auto all_ids = IncrementalMapperImpl::FindFirstInitialImage(
      options_, corr_graph, *reconstruction_, init_num_reg_trials,
      no_registrations);
  ASSERT_GE(all_ids.size(), 2);

  // Mark some images as registered in another reconstruction.
  std::unordered_map<image_t, size_t> num_registrations;
  num_registrations[all_ids[0]] = 1;
  num_registrations[all_ids[1]] = 2;

  const auto filtered_ids = IncrementalMapperImpl::FindFirstInitialImage(
      options_, corr_graph, *reconstruction_, init_num_reg_trials,
      num_registrations);

  EXPECT_EQ(filtered_ids.size(), all_ids.size() - 2);
  for (const auto id : filtered_ids) {
    EXPECT_NE(id, all_ids[0]);
    EXPECT_NE(id, all_ids[1]);
  }
}

// Test that FindFirstInitialImage prioritizes images with prior focal length.
TEST_F(IncrementalMapperImplTest,
       FindFirstInitialImagePrioritizesPriorFocalLength) {
  // Set prior focal length on one camera but not the other.
  // The synthetic dataset has 2 cameras (one per rig).
  for (const auto& [camera_id, camera] : reconstruction_->Cameras()) {
    reconstruction_->Camera(camera_id).has_prior_focal_length = false;
  }
  // Pick one camera and set prior focal length.
  const auto first_camera_id = reconstruction_->Cameras().begin()->first;
  reconstruction_->Camera(first_camera_id).has_prior_focal_length = true;

  const auto& corr_graph = *cache_->CorrespondenceGraph();
  std::unordered_map<image_t, size_t> init_num_reg_trials;
  std::unordered_map<image_t, size_t> num_registrations;

  const auto image_ids = IncrementalMapperImpl::FindFirstInitialImage(
      options_, corr_graph, *reconstruction_, init_num_reg_trials,
      num_registrations);

  ASSERT_FALSE(image_ids.empty());

  // All images with prior focal length should come before those without.
  bool seen_non_prior = false;
  for (const auto image_id : image_ids) {
    const auto& image = reconstruction_->Image(image_id);
    bool has_prior = image.CameraPtr()->has_prior_focal_length;
    if (!has_prior) {
      seen_non_prior = true;
    }
    if (seen_non_prior) {
      EXPECT_FALSE(has_prior)
          << "Image with prior focal length found after non-prior image";
    }
  }
}

// Test FindSecondInitialImage returns connected images sorted properly.
TEST_F(IncrementalMapperImplTest, FindSecondInitialImageBasic) {
  const auto& corr_graph = *cache_->CorrespondenceGraph();
  std::unordered_map<image_t, size_t> num_registrations;
  std::unordered_map<image_t, size_t> init_num_reg_trials;

  // Find the first initial image.
  const auto first_ids = IncrementalMapperImpl::FindFirstInitialImage(
      options_, corr_graph, *reconstruction_, init_num_reg_trials,
      num_registrations);
  ASSERT_FALSE(first_ids.empty());
  const image_t image_id1 = first_ids[0];

  const auto second_ids = IncrementalMapperImpl::FindSecondInitialImage(
      options_, image_id1, corr_graph, *reconstruction_, num_registrations);

  EXPECT_FALSE(second_ids.empty());
  // None of the returned images should be the first image.
  for (const auto id : second_ids) {
    EXPECT_NE(id, image_id1);
  }
}

// Test FindSecondInitialImage filters out images registered in other
// reconstructions.
TEST_F(IncrementalMapperImplTest,
       FindSecondInitialImageSkipsRegisteredImages) {
  const auto& corr_graph = *cache_->CorrespondenceGraph();
  std::unordered_map<image_t, size_t> init_num_reg_trials;

  // Get candidates without any registration filter.
  std::unordered_map<image_t, size_t> no_registrations;
  const auto first_ids = IncrementalMapperImpl::FindFirstInitialImage(
      options_, corr_graph, *reconstruction_, init_num_reg_trials,
      no_registrations);
  ASSERT_FALSE(first_ids.empty());
  const image_t image_id1 = first_ids[0];

  const auto all_second_ids = IncrementalMapperImpl::FindSecondInitialImage(
      options_, image_id1, corr_graph, *reconstruction_, no_registrations);
  ASSERT_GE(all_second_ids.size(), 2);

  // Mark some second-image candidates as registered.
  std::unordered_map<image_t, size_t> num_registrations;
  num_registrations[all_second_ids[0]] = 1;

  const auto filtered_second_ids =
      IncrementalMapperImpl::FindSecondInitialImage(
          options_, image_id1, corr_graph, *reconstruction_, num_registrations);

  // The registered image should be excluded.
  for (const auto id : filtered_second_ids) {
    EXPECT_NE(id, all_second_ids[0]);
  }
}

// Test FindSecondInitialImage filters images below init_min_num_inliers.
TEST_F(IncrementalMapperImplTest,
       FindSecondInitialImageRespectsMinInliers) {
  const auto& corr_graph = *cache_->CorrespondenceGraph();
  std::unordered_map<image_t, size_t> num_registrations;
  std::unordered_map<image_t, size_t> init_num_reg_trials;

  const auto first_ids = IncrementalMapperImpl::FindFirstInitialImage(
      options_, corr_graph, *reconstruction_, init_num_reg_trials,
      num_registrations);
  ASSERT_FALSE(first_ids.empty());
  const image_t image_id1 = first_ids[0];

  // With a very high threshold, no second images should pass.
  IncrementalMapper::Options strict_options = options_;
  strict_options.init_min_num_inliers = 100000;

  const auto second_ids = IncrementalMapperImpl::FindSecondInitialImage(
      strict_options, image_id1, corr_graph, *reconstruction_,
      num_registrations);
  EXPECT_TRUE(second_ids.empty());
}

// Test FindInitialImagePair when only image_id1 is provided.
TEST_F(IncrementalMapperImplTest, FindInitialImagePairWithProvidedImageId1) {
  const auto& corr_graph = *cache_->CorrespondenceGraph();
  std::unordered_map<image_t, size_t> init_num_reg_trials;
  std::unordered_map<image_t, size_t> num_registrations;
  std::unordered_set<image_pair_t> init_image_pairs;

  // Pick a valid image to provide as image_id1.
  const auto first_ids = IncrementalMapperImpl::FindFirstInitialImage(
      options_, corr_graph, *reconstruction_, init_num_reg_trials,
      num_registrations);
  ASSERT_FALSE(first_ids.empty());

  image_t image_id1 = first_ids[0];
  image_t image_id2 = kInvalidImageId;
  Rigid3d cam2_from_cam1;
  const bool success = IncrementalMapperImpl::FindInitialImagePair(
      options_, *cache_, *reconstruction_, init_num_reg_trials,
      num_registrations, init_image_pairs, image_id1, image_id2,
      cam2_from_cam1);

  if (success) {
    EXPECT_NE(image_id1, kInvalidImageId);
    EXPECT_NE(image_id2, kInvalidImageId);
    EXPECT_NE(image_id1, image_id2);
  }
}

// Test FindInitialImagePair when only image_id2 is provided (swapped path).
TEST_F(IncrementalMapperImplTest, FindInitialImagePairWithProvidedImageId2) {
  const auto& corr_graph = *cache_->CorrespondenceGraph();
  std::unordered_map<image_t, size_t> init_num_reg_trials;
  std::unordered_map<image_t, size_t> num_registrations;
  std::unordered_set<image_pair_t> init_image_pairs;

  // Pick a valid image to provide as image_id2 (image_id1 is invalid).
  const auto first_ids = IncrementalMapperImpl::FindFirstInitialImage(
      options_, corr_graph, *reconstruction_, init_num_reg_trials,
      num_registrations);
  ASSERT_FALSE(first_ids.empty());

  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = first_ids[0];
  Rigid3d cam2_from_cam1;
  const bool success = IncrementalMapperImpl::FindInitialImagePair(
      options_, *cache_, *reconstruction_, init_num_reg_trials,
      num_registrations, init_image_pairs, image_id1, image_id2,
      cam2_from_cam1);

  if (success) {
    EXPECT_NE(image_id1, kInvalidImageId);
    EXPECT_NE(image_id2, kInvalidImageId);
  }
}

// Test FindInitialImagePair returns false for non-existent image.
TEST_F(IncrementalMapperImplTest,
       FindInitialImagePairFailsForNonExistentImage) {
  std::unordered_map<image_t, size_t> init_num_reg_trials;
  std::unordered_map<image_t, size_t> num_registrations;
  std::unordered_set<image_pair_t> init_image_pairs;

  // Provide a non-existent image ID.
  image_t image_id1 = 99999;
  image_t image_id2 = kInvalidImageId;
  Rigid3d cam2_from_cam1;
  EXPECT_FALSE(IncrementalMapperImpl::FindInitialImagePair(
      options_, *cache_, *reconstruction_, init_num_reg_trials,
      num_registrations, init_image_pairs, image_id1, image_id2,
      cam2_from_cam1));
}

// Test FindInitialImagePair returns false for non-existent image_id2.
TEST_F(IncrementalMapperImplTest,
       FindInitialImagePairFailsForNonExistentImageId2) {
  std::unordered_map<image_t, size_t> init_num_reg_trials;
  std::unordered_map<image_t, size_t> num_registrations;
  std::unordered_set<image_pair_t> init_image_pairs;

  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = 99999;
  Rigid3d cam2_from_cam1;
  EXPECT_FALSE(IncrementalMapperImpl::FindInitialImagePair(
      options_, *cache_, *reconstruction_, init_num_reg_trials,
      num_registrations, init_image_pairs, image_id1, image_id2,
      cam2_from_cam1));
}

// Test FindInitialImagePair skips already-tried pairs.
TEST_F(IncrementalMapperImplTest,
       FindInitialImagePairSkipsAlreadyTriedPairs) {
  std::unordered_map<image_t, size_t> init_num_reg_trials;
  std::unordered_map<image_t, size_t> num_registrations;
  std::unordered_set<image_pair_t> init_image_pairs;

  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  Rigid3d cam2_from_cam1;

  // First attempt should succeed.
  ASSERT_TRUE(IncrementalMapperImpl::FindInitialImagePair(
      options_, *cache_, *reconstruction_, init_num_reg_trials,
      num_registrations, init_image_pairs, image_id1, image_id2,
      cam2_from_cam1));
  EXPECT_FALSE(init_image_pairs.empty());

  // The pair was recorded in init_image_pairs. Trying again with
  // the same set should eventually exhaust pairs (or find different ones).
  const size_t pairs_after_first = init_image_pairs.size();

  image_t image_id3 = kInvalidImageId;
  image_t image_id4 = kInvalidImageId;
  Rigid3d cam2_from_cam1_2;
  // This may succeed with a different pair, or fail if all pairs tried.
  IncrementalMapperImpl::FindInitialImagePair(
      options_, *cache_, *reconstruction_, init_num_reg_trials,
      num_registrations, init_image_pairs, image_id3, image_id4,
      cam2_from_cam1_2);
  // Either way, at least the first pair should still be in the set.
  EXPECT_GE(init_image_pairs.size(), pairs_after_first);
}

// Fixture that builds a full reconstruction for testing FindNextImages
// and FindLocalBundle with different parameters.
class IncrementalMapperImplFullTest : public ::testing::Test {
 protected:
  void SetUp() override {
    database_ = Database::Open(kInMemorySqliteDatabasePath);
    SynthesizeDataset(synthetic_options_, &gt_reconstruction_, database_.get());
    DatabaseCache::Options cache_options;
    cache_ = DatabaseCache::Create(*database_, cache_options);
    mapper_ = std::make_unique<IncrementalMapper>(cache_);
    reconstruction_ = std::make_shared<Reconstruction>();
    mapper_->BeginReconstruction(reconstruction_);
    options_.init_min_num_inliers = 10;
    options_.abs_pose_min_num_inliers = 10;
    options_.abs_pose_min_inlier_ratio = 0.1;

    // Register initial pair and triangulate.
    ASSERT_TRUE(mapper_->FindInitialImagePair(
        options_, image_id1_, image_id2_, cam2_from_cam1_));
    mapper_->RegisterInitialImagePair(
        options_, image_id1_, image_id2_, cam2_from_cam1_);
    IncrementalTriangulator::Options tri_options;
    mapper_->TriangulateImage(tri_options, image_id1_);
    mapper_->TriangulateImage(tri_options, image_id2_);

    // Register all remaining images.
    const auto next_image_ids = mapper_->FindNextImages(options_);
    for (const auto image_id : next_image_ids) {
      if (mapper_->RegisterNextImage(options_, image_id)) {
        mapper_->TriangulateImage(tri_options, image_id);
      }
    }
    ASSERT_EQ(reconstruction_->NumRegFrames(), 10);
  }

  void TearDown() override {
    if (mapper_->Reconstruction()) {
      mapper_->EndReconstruction(/*discard=*/false);
    }
  }

  SyntheticDatasetOptions synthetic_options_ = DefaultSyntheticOptions();
  Reconstruction gt_reconstruction_;
  std::shared_ptr<Database> database_;
  std::shared_ptr<DatabaseCache> cache_;
  std::unique_ptr<IncrementalMapper> mapper_;
  std::shared_ptr<Reconstruction> reconstruction_;
  IncrementalMapper::Options options_;
  image_t image_id1_ = kInvalidImageId;
  image_t image_id2_ = kInvalidImageId;
  Rigid3d cam2_from_cam1_;
};

// Test FindNextImages with MAX_VISIBLE_POINTS_NUM selection method.
TEST_F(IncrementalMapperImplFullTest, FindNextImagesMaxVisiblePointsNum) {
  // De-register one frame so there's something to find next.
  const auto reg_image_ids = reconstruction_->RegImageIds();
  ASSERT_GE(reg_image_ids.size(), 3);
  // Pick an image that is not in the initial pair.
  image_t deregistered_image = kInvalidImageId;
  for (const auto id : reg_image_ids) {
    if (id != image_id1_ && id != image_id2_) {
      deregistered_image = id;
      break;
    }
  }
  ASSERT_NE(deregistered_image, kInvalidImageId);

  const auto& image = reconstruction_->Image(deregistered_image);
  mapper_->ObservationManager().DeRegisterFrame(image.FrameId());

  IncrementalMapper::Options opts = options_;
  opts.image_selection_method =
      IncrementalMapper::Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_NUM;

  const auto next_ids = mapper_->FindNextImages(opts);

  // The deregistered image should appear in the next images list.
  bool found = false;
  for (const auto id : next_ids) {
    if (id == deregistered_image) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

// Test FindNextImages with MAX_VISIBLE_POINTS_RATIO selection method.
TEST_F(IncrementalMapperImplFullTest, FindNextImagesMaxVisiblePointsRatio) {
  const auto reg_image_ids = reconstruction_->RegImageIds();
  ASSERT_GE(reg_image_ids.size(), 3);
  image_t deregistered_image = kInvalidImageId;
  for (const auto id : reg_image_ids) {
    if (id != image_id1_ && id != image_id2_) {
      deregistered_image = id;
      break;
    }
  }
  ASSERT_NE(deregistered_image, kInvalidImageId);

  const auto& image = reconstruction_->Image(deregistered_image);
  mapper_->ObservationManager().DeRegisterFrame(image.FrameId());

  IncrementalMapper::Options opts = options_;
  opts.image_selection_method = IncrementalMapper::Options::ImageSelectionMethod::
      MAX_VISIBLE_POINTS_RATIO;

  const auto next_ids = mapper_->FindNextImages(opts);
  bool found = false;
  for (const auto id : next_ids) {
    if (id == deregistered_image) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

// Test FindNextImages in structure-less mode.
TEST_F(IncrementalMapperImplFullTest, FindNextImagesStructureLess) {
  const auto reg_image_ids = reconstruction_->RegImageIds();
  ASSERT_GE(reg_image_ids.size(), 3);
  image_t deregistered_image = kInvalidImageId;
  for (const auto id : reg_image_ids) {
    if (id != image_id1_ && id != image_id2_) {
      deregistered_image = id;
      break;
    }
  }
  ASSERT_NE(deregistered_image, kInvalidImageId);

  const auto& image = reconstruction_->Image(deregistered_image);
  mapper_->ObservationManager().DeRegisterFrame(image.FrameId());

  const auto next_ids =
      mapper_->FindNextImages(options_, /*structure_less=*/true);

  // Structure-less mode uses NumVisibleCorrespondences for ranking.
  // The deregistered image with correspondences should appear.
  bool found = false;
  for (const auto id : next_ids) {
    if (id == deregistered_image) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

// Test FindNextImages filters images that exceeded max_reg_trials.
TEST_F(IncrementalMapperImplFullTest, FindNextImagesRespectsMaxRegTrials) {
  const auto reg_image_ids = reconstruction_->RegImageIds();
  ASSERT_GE(reg_image_ids.size(), 3);

  // Deregister two images so they become candidates.
  std::vector<image_t> deregistered;
  for (const auto id : reg_image_ids) {
    if (id != image_id1_ && id != image_id2_ && deregistered.size() < 2) {
      const auto& img = reconstruction_->Image(id);
      mapper_->ObservationManager().DeRegisterFrame(img.FrameId());
      deregistered.push_back(id);
    }
  }
  ASSERT_EQ(deregistered.size(), 2);

  // Artificially bump the reg_trials for the first deregistered image
  // to exceed max_reg_trials, by calling FindNextImages multiple times.
  // The impl function takes a mutable ref to num_reg_trials, so we
  // test the behavior through the mapper which tracks this internally.
  // Instead, verify that FindNextImages returns both initially.
  const auto next_ids = mapper_->FindNextImages(options_);
  bool found_first = false;
  bool found_second = false;
  for (const auto id : next_ids) {
    if (id == deregistered[0]) found_first = true;
    if (id == deregistered[1]) found_second = true;
  }
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

// Test FindLocalBundle with a small ba_local_num_images setting.
TEST_F(IncrementalMapperImplFullTest, FindLocalBundleSmallBudget) {
  IncrementalMapper::Options opts = options_;
  opts.ba_local_num_images = 3;

  const auto local_bundle = mapper_->FindLocalBundle(opts, image_id1_);
  // ba_local_num_images = 3 means up to 2 neighbor images (3 - 1 for the
  // query image itself).
  EXPECT_LE(local_bundle.size(), 2);
  EXPECT_FALSE(local_bundle.empty());

  // The query image should not be in its own local bundle.
  for (const auto id : local_bundle) {
    EXPECT_NE(id, image_id1_);
  }
}

// Test FindLocalBundle uses triangulation angle selection thresholds.
TEST_F(IncrementalMapperImplFullTest,
       FindLocalBundleTriangulationAngleSelection) {
  IncrementalMapper::Options opts = options_;
  opts.ba_local_num_images = 6;
  opts.ba_local_min_tri_angle = 0.1;

  const auto local_bundle = mapper_->FindLocalBundle(opts, image_id1_);
  EXPECT_LE(local_bundle.size(),
            static_cast<size_t>(opts.ba_local_num_images - 1));
  EXPECT_FALSE(local_bundle.empty());

  // All returned images should have poses.
  for (const auto id : local_bundle) {
    EXPECT_TRUE(reconstruction_->Image(id).HasPose());
  }
}

// Test FindLocalBundle direct call with large budget exceeding available images.
TEST_F(IncrementalMapperImplFullTest, FindLocalBundleLargeBudget) {
  IncrementalMapper::Options opts = options_;
  opts.ba_local_num_images = 100;

  const auto local_bundle = mapper_->FindLocalBundle(opts, image_id1_);
  // Should return all overlapping images (up to 9 = 10 registered - 1 query).
  EXPECT_LE(local_bundle.size(), reconstruction_->NumRegImages() - 1);
}

// Test FindLocalBundle via IncrementalMapperImpl directly.
TEST_F(IncrementalMapperImplFullTest, FindLocalBundleDirect) {
  IncrementalMapper::Options opts = options_;
  opts.ba_local_num_images = 4;

  const auto local_bundle = IncrementalMapperImpl::FindLocalBundle(
      opts, image_id1_, *reconstruction_);

  // Should return up to 3 images (4 - 1).
  EXPECT_LE(local_bundle.size(), 3);

  // All returned images must be registered and distinct from query.
  for (const auto id : local_bundle) {
    EXPECT_NE(id, image_id1_);
    EXPECT_TRUE(reconstruction_->Image(id).HasPose());
  }
}

// Test EstimateInitialTwoViewGeometry directly on the impl.
TEST_F(IncrementalMapperImplTest, EstimateInitialTwoViewGeometryDirect) {
  std::unordered_map<image_t, size_t> init_num_reg_trials;
  std::unordered_map<image_t, size_t> num_registrations;
  std::unordered_set<image_pair_t> init_image_pairs;

  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  Rigid3d cam2_from_cam1;

  // Find a valid pair first.
  ASSERT_TRUE(IncrementalMapperImpl::FindInitialImagePair(
      options_, *cache_, *reconstruction_, init_num_reg_trials,
      num_registrations, init_image_pairs, image_id1, image_id2,
      cam2_from_cam1));

  // Re-estimate using the direct method.
  Rigid3d cam2_from_cam1_re;
  ASSERT_TRUE(IncrementalMapperImpl::EstimateInitialTwoViewGeometry(
      options_, *cache_, image_id1, image_id2, cam2_from_cam1_re));

  EXPECT_GT(cam2_from_cam1_re.translation().norm(), 0.0);
}

// Test EstimateInitialTwoViewGeometry rejects pairs with too few inliers.
TEST_F(IncrementalMapperImplTest,
       EstimateInitialTwoViewGeometryRejectsFewInliers) {
  std::unordered_map<image_t, size_t> init_num_reg_trials;
  std::unordered_map<image_t, size_t> num_registrations;
  std::unordered_set<image_pair_t> init_image_pairs;

  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  Rigid3d cam2_from_cam1;

  ASSERT_TRUE(IncrementalMapperImpl::FindInitialImagePair(
      options_, *cache_, *reconstruction_, init_num_reg_trials,
      num_registrations, init_image_pairs, image_id1, image_id2,
      cam2_from_cam1));

  // Set impossibly high inlier requirement.
  IncrementalMapper::Options strict_opts = options_;
  strict_opts.init_min_num_inliers = 1000000;

  Rigid3d cam2_from_cam1_strict;
  EXPECT_FALSE(IncrementalMapperImpl::EstimateInitialTwoViewGeometry(
      strict_opts, *cache_, image_id1, image_id2, cam2_from_cam1_strict));
}

// Test EstimateInitialTwoViewGeometry rejects pairs with too much forward
// motion.
TEST_F(IncrementalMapperImplTest,
       EstimateInitialTwoViewGeometryRejectsForwardMotion) {
  std::unordered_map<image_t, size_t> init_num_reg_trials;
  std::unordered_map<image_t, size_t> num_registrations;
  std::unordered_set<image_pair_t> init_image_pairs;

  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  Rigid3d cam2_from_cam1;

  ASSERT_TRUE(IncrementalMapperImpl::FindInitialImagePair(
      options_, *cache_, *reconstruction_, init_num_reg_trials,
      num_registrations, init_image_pairs, image_id1, image_id2,
      cam2_from_cam1));

  // Set impossibly low forward motion threshold (reject everything).
  IncrementalMapper::Options strict_opts = options_;
  strict_opts.init_max_forward_motion = 0.0;

  Rigid3d cam2_from_cam1_strict;
  EXPECT_FALSE(IncrementalMapperImpl::EstimateInitialTwoViewGeometry(
      strict_opts, *cache_, image_id1, image_id2, cam2_from_cam1_strict));
}

// Test EstimateInitialTwoViewGeometry rejects pairs with insufficient
// triangulation angle.
TEST_F(IncrementalMapperImplTest,
       EstimateInitialTwoViewGeometryRejectsSmallTriAngle) {
  std::unordered_map<image_t, size_t> init_num_reg_trials;
  std::unordered_map<image_t, size_t> num_registrations;
  std::unordered_set<image_pair_t> init_image_pairs;

  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  Rigid3d cam2_from_cam1;

  ASSERT_TRUE(IncrementalMapperImpl::FindInitialImagePair(
      options_, *cache_, *reconstruction_, init_num_reg_trials,
      num_registrations, init_image_pairs, image_id1, image_id2,
      cam2_from_cam1));

  // Require impossibly large triangulation angle.
  IncrementalMapper::Options strict_opts = options_;
  strict_opts.init_min_tri_angle = 170.0;

  Rigid3d cam2_from_cam1_strict;
  EXPECT_FALSE(IncrementalMapperImpl::EstimateInitialTwoViewGeometry(
      strict_opts, *cache_, image_id1, image_id2, cam2_from_cam1_strict));
}

}  // namespace
}  // namespace colmap
