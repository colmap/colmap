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

#include <algorithm>

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
    while (true) {
      bool any_image_registered = false;
      const auto next_image_ids = mapper_->FindNextImages(options_);
      for (const auto image_id : next_image_ids) {
        if (mapper_->RegisterNextImage(options_, image_id)) {
          mapper_->TriangulateImage(tri_options_, image_id);
          any_image_registered = true;
        }
      }
      if (!any_image_registered) {
        break;
      }
    }
  }

  void BeginWithSynthesizedReconstruction() {
    if (mapper_->Reconstruction()) {
      mapper_->EndReconstruction(/*discard=*/false);
    }
    reconstruction_ = std::make_shared<Reconstruction>(gt_reconstruction_);
    mapper_->BeginReconstruction(reconstruction_);
  }

  bool IsFrameRegistered(const frame_t frame_id) const {
    const auto& reg_frame_ids = reconstruction_->RegFrameIds();
    return std::find(reg_frame_ids.begin(), reg_frame_ids.end(), frame_id) !=
           reg_frame_ids.end();
  }

  size_t CountFramePoints3D(const frame_t frame_id) const {
    size_t num_points3D = 0;
    const auto& frame = reconstruction_->Frame(frame_id);
    for (const data_t& data_id : frame.ImageIds()) {
      num_points3D += reconstruction_->Image(data_id.id).NumPoints3D();
    }
    return num_points3D;
  }

  size_t CountRegisteredFramesWithZeroPoints3D() const {
    size_t num_zero_point_frames = 0;
    for (const frame_t frame_id : reconstruction_->RegFrameIds()) {
      if (CountFramePoints3D(frame_id) == 0) {
        num_zero_point_frames += 1;
      }
    }
    return num_zero_point_frames;
  }

  frame_t FindRegisteredFrameWithPoints3D() const {
    for (const frame_t frame_id : reconstruction_->RegFrameIds()) {
      if (CountFramePoints3D(frame_id) > 0) {
        return frame_id;
      }
    }
    return kInvalidFrameId;
  }

  void DeleteAllObservationsInFrame(const frame_t frame_id) {
    std::vector<std::pair<image_t, point2D_t>> observations_to_delete;
    const auto& frame = reconstruction_->Frame(frame_id);
    for (const data_t& data_id : frame.ImageIds()) {
      const auto& image = reconstruction_->Image(data_id.id);
      for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
           ++point2D_idx) {
        if (image.Point2D(point2D_idx).HasPoint3D()) {
          observations_to_delete.emplace_back(data_id.id, point2D_idx);
        }
      }
    }

    for (const auto& [image_id, point2D_idx] : observations_to_delete) {
      if (reconstruction_->Image(image_id).Point2D(point2D_idx).HasPoint3D()) {
        mapper_->ObservationManager().DeleteObservation(image_id, point2D_idx);
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

class IncrementalMapperLargeDatasetTest : public IncrementalMapperTest {
 protected:
  void SetUp() override {
    synthetic_options_.num_frames_per_rig = 12;
    IncrementalMapperTest::SetUp();
  }
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

// Frame filtering is disabled before the 20-frame mapper threshold.
TEST_F(IncrementalMapperTest, FilterFramesNoOpBelowMinFrames) {
  BeginWithSynthesizedReconstruction();

  ASSERT_LT(reconstruction_->NumRegFrames(), 20);

  const frame_t target_frame_id = FindRegisteredFrameWithPoints3D();
  ASSERT_NE(target_frame_id, kInvalidFrameId);
  ASSERT_TRUE(IsFrameRegistered(target_frame_id));
  ASSERT_GT(CountFramePoints3D(target_frame_id), 0);

  DeleteAllObservationsInFrame(target_frame_id);
  ASSERT_EQ(CountFramePoints3D(target_frame_id), 0);

  const size_t num_reg_frames_before = reconstruction_->NumRegFrames();
  const size_t num_filtered_frames = mapper_->FilterFrames(options_);

  EXPECT_EQ(num_filtered_frames, 0);
  EXPECT_EQ(reconstruction_->NumRegFrames(), num_reg_frames_before);
  EXPECT_TRUE(IsFrameRegistered(target_frame_id));
  EXPECT_EQ(mapper_->FilteredFrames().count(target_frame_id), 0);
}

// At or above threshold, filtering removes the chosen zero-observation frame.
TEST_F(IncrementalMapperLargeDatasetTest,
       FilterFramesRemovesZeroObservationFrameAfterThreshold) {
  BeginWithSynthesizedReconstruction();

  ASSERT_GE(reconstruction_->NumRegFrames(), 20);

  const frame_t target_frame_id = FindRegisteredFrameWithPoints3D();
  ASSERT_NE(target_frame_id, kInvalidFrameId);
  ASSERT_TRUE(IsFrameRegistered(target_frame_id));
  ASSERT_GT(CountFramePoints3D(target_frame_id), 0);

  DeleteAllObservationsInFrame(target_frame_id);
  ASSERT_EQ(CountFramePoints3D(target_frame_id), 0);
  ASSERT_TRUE(IsFrameRegistered(target_frame_id));
  ASSERT_EQ(CountRegisteredFramesWithZeroPoints3D(), 1);

  const size_t num_reg_frames_before = reconstruction_->NumRegFrames();
  const size_t num_filtered_frames = mapper_->FilterFrames(options_);

  EXPECT_EQ(num_filtered_frames, 1);
  EXPECT_EQ(reconstruction_->NumRegFrames(), num_reg_frames_before - 1);
  EXPECT_FALSE(IsFrameRegistered(target_frame_id));
  EXPECT_EQ(mapper_->FilteredFrames().count(target_frame_id), 1);
}

// Strict reprojection filtering removes the intentionally corrupted point.
TEST_F(IncrementalMapperTest, FilterPointsRemovesCorruptedPoint) {
  BeginWithSynthesizedReconstruction();

  ASSERT_GT(reconstruction_->NumPoints3D(), 0);

  const auto point3D_ids = reconstruction_->Point3DIds();
  ASSERT_FALSE(point3D_ids.empty());
  const auto it = std::max_element(point3D_ids.begin(), point3D_ids.end());
  ASSERT_NE(it, point3D_ids.end());
  const point3D_t corrupted_point3D_id = *it;
  ASSERT_GE(reconstruction_->Point3D(corrupted_point3D_id).track.Length(), 2);
  reconstruction_->Point3D(corrupted_point3D_id).xyz +=
      Eigen::Vector3d(100, 100, 100);

  IncrementalMapper::Options strict_options = options_;
  strict_options.filter_max_reproj_error = 0.1;
  strict_options.filter_min_tri_angle = 0.0;
  const size_t num_filtered_observations =
      mapper_->FilterPoints(strict_options);

  EXPECT_GT(num_filtered_observations, 0);
  EXPECT_FALSE(reconstruction_->ExistsPoint3D(corrupted_point3D_id));
}

TEST_F(IncrementalMapperLargeDatasetTest,
       ObservationBookkeepingAfterInitAndFilter) {
  BeginWithSynthesizedReconstruction();

  const size_t num_reg_frames_before = reconstruction_->NumRegFrames();
  const size_t num_reg_images_before = mapper_->NumTotalRegImages();

  // All images are registered, so num_visible_correspondences must exactly
  // equal num_correspondences (every correspondence partner is registered).
  const auto& obs_manager = mapper_->ObservationManager();
  for (const auto& [image_id, image] : reconstruction_->Images()) {
    EXPECT_EQ(obs_manager.NumObservations(image_id), image.NumPoints3D());
    EXPECT_EQ(obs_manager.NumVisiblePoints3D(image_id),
              reconstruction_->NumPoints3D());
    EXPECT_EQ(obs_manager.NumVisibleCorrespondences(image_id),
              obs_manager.NumCorrespondences(image_id))
        << "num_visible_correspondences wrong for image " << image_id;
  }

  const frame_t target_frame_id = FindRegisteredFrameWithPoints3D();
  ASSERT_NE(target_frame_id, kInvalidFrameId);
  const Frame& target_frame = reconstruction_->Frame(target_frame_id);
  const size_t num_images_in_frame = std::distance(
      target_frame.ImageIds().begin(), target_frame.ImageIds().end());

  DeleteAllObservationsInFrame(target_frame_id);
  ASSERT_EQ(CountFramePoints3D(target_frame_id), 0);

  mapper_->FilterFrames(options_);

  // Verify reconstruction state.
  EXPECT_EQ(reconstruction_->NumRegFrames(), num_reg_frames_before - 1);

  // Verify reg_stats_ consistency after filtering.
  EXPECT_EQ(mapper_->NumTotalRegImages(),
            num_reg_images_before - num_images_in_frame);
  EXPECT_EQ(mapper_->FilteredFrames().count(target_frame_id), 1);

  std::unordered_set<image_t> filtered_image_ids;
  for (const data_t& data_id : target_frame.ImageIds()) {
    filtered_image_ids.insert(data_id.id);
  }

  // After filtering, verify exact num_visible_correspondences for every image.
  // DeRegisterFrame decrements the correspondence partners' counters, so each
  // remaining image loses exactly the matches it had to the deregistered
  // images. The deregistered images' own counters are unchanged
  // (DeRegisterFrame only touches partners). An underflowed uint32_t would wrap
  // to a very large value, failing these exact checks.
  const auto& corr_graph = *cache_->CorrespondenceGraph();
  for (const auto& [image_id, image] : reconstruction_->Images()) {
    point2D_t expected_visible_corrs = obs_manager.NumCorrespondences(image_id);
    if (!filtered_image_ids.count(image_id)) {
      // Registered image: lost matches to each deregistered image.
      for (const image_t filtered_image_id : filtered_image_ids) {
        expected_visible_corrs -=
            corr_graph.NumMatchesBetweenImages(image_id, filtered_image_id);
      }
    }
    EXPECT_EQ(obs_manager.NumVisibleCorrespondences(image_id),
              expected_visible_corrs)
        << "num_visible_correspondences wrong for image " << image_id;

    // Visibility counts must remain internally consistent.
    EXPECT_LE(obs_manager.NumVisiblePoints3D(image_id),
              obs_manager.NumObservations(image_id));
  }
}

// Re-initializing from a discarded reconstruction must produce correct
// observation bookkeeping, verifying symmetry of register/deregister paths.
TEST_F(IncrementalMapperLargeDatasetTest,
       ObservationBookkeepingAfterDiscardAndReinit) {
  BeginWithSynthesizedReconstruction();

  const size_t num_reg_frames = reconstruction_->NumRegFrames();
  ASSERT_GE(num_reg_frames, 20);

  // Discard and re-initialize with the same ground truth reconstruction.
  mapper_->EndReconstruction(/*discard=*/true);
  BeginWithSynthesizedReconstruction();

  EXPECT_EQ(reconstruction_->NumRegFrames(), num_reg_frames);
  EXPECT_EQ(mapper_->NumTotalRegImages(), reconstruction_->NumRegImages());

  // After a full discard + reinit cycle, the observation bookkeeping
  // must be correct again — no state leaks from the previous cycle.
  const auto& obs_manager = mapper_->ObservationManager();
  for (const auto& [image_id, image] : reconstruction_->Images()) {
    EXPECT_EQ(obs_manager.NumVisibleCorrespondences(image_id),
              obs_manager.NumCorrespondences(image_id))
        << "num_visible_correspondences wrong for image " << image_id;
  }
}

// Filtering must update reg_stats_ (per-rig and per-camera counts) to match
// the actual reconstruction state.
TEST_F(IncrementalMapperLargeDatasetTest, FilterFramesRegStatsConsistency) {
  BeginWithSynthesizedReconstruction();

  ASSERT_GE(reconstruction_->NumRegFrames(), 20);

  const frame_t target_frame_id = FindRegisteredFrameWithPoints3D();
  ASSERT_NE(target_frame_id, kInvalidFrameId);
  const Frame& target_frame = reconstruction_->Frame(target_frame_id);
  const rig_t target_rig_id = target_frame.RigId();

  // Record per-rig and per-camera counts before filtering.
  const size_t rig_count_before =
      mapper_->NumRegFramesPerRig().at(target_rig_id);
  std::unordered_map<camera_t, size_t> camera_counts_before;
  for (const data_t& data_id : target_frame.ImageIds()) {
    const camera_t camera_id = reconstruction_->Image(data_id.id).CameraId();
    camera_counts_before[camera_id] =
        mapper_->NumRegImagesPerCamera().at(camera_id);
  }

  DeleteAllObservationsInFrame(target_frame_id);
  mapper_->FilterFrames(options_);

  // Per-rig count should decrease by exactly 1.
  EXPECT_EQ(mapper_->NumRegFramesPerRig().at(target_rig_id),
            rig_count_before - 1);

  // Per-camera counts should decrease by exactly the number of images in the
  // filtered frame that used each camera.
  for (const data_t& data_id : target_frame.ImageIds()) {
    const camera_t camera_id = reconstruction_->Image(data_id.id).CameraId();
    EXPECT_EQ(mapper_->NumRegImagesPerCamera().at(camera_id),
              camera_counts_before.at(camera_id) - 1);
  }
}

// Reproduces the crash when FilterFrames aggressively deregisters frames,
// leaving fewer than 2 images for a subsequent AdjustGlobalBundle call.
TEST_F(IncrementalMapperLargeDatasetTest,
       AdjustGlobalBundleReturnsFalseWithInsufficientFrames) {
  BeginWithSynthesizedReconstruction();

  ASSERT_GE(reconstruction_->NumRegFrames(), 20);

  for (const frame_t frame_id : reconstruction_->RegFrameIds()) {
    DeleteAllObservationsInFrame(frame_id);
  }

  mapper_->FilterFrames(options_);
  ASSERT_LT(reconstruction_->NumRegImages(), 2);

  EXPECT_FALSE(mapper_->AdjustGlobalBundle(options_, /*ba_options=*/{}));
}

TEST_F(IncrementalMapperTest, RegStatsResetBetweenReconstructions) {
  BeginWithSynthesizedReconstruction();

  const size_t num_reg_images_first = mapper_->NumTotalRegImages();
  const auto rig_counts_first = mapper_->NumRegFramesPerRig();
  const auto camera_counts_first = mapper_->NumRegImagesPerCamera();
  ASSERT_GT(num_reg_images_first, 0);
  ASSERT_FALSE(rig_counts_first.empty());
  ASSERT_FALSE(camera_counts_first.empty());
  ASSERT_FALSE(mapper_->ExistingFrameIds().empty());

  // End without discard (keeps cross-reconstruction state).
  mapper_->EndReconstruction(/*discard=*/false);

  // Begin a fresh empty reconstruction.
  reconstruction_ = std::make_shared<Reconstruction>();
  mapper_->BeginReconstruction(reconstruction_);

  // Per-reconstruction stats must be reset for a fresh reconstruction.
  EXPECT_TRUE(mapper_->NumRegFramesPerRig().empty());
  EXPECT_TRUE(mapper_->NumRegImagesPerCamera().empty());
  EXPECT_TRUE(mapper_->FilteredFrames().empty());
  EXPECT_TRUE(mapper_->ExistingFrameIds().empty());
  EXPECT_EQ(mapper_->NumSharedRegImages(), 0);

  // Cross-reconstruction stats must persist.
  EXPECT_EQ(mapper_->NumTotalRegImages(), num_reg_images_first);

  mapper_->EndReconstruction(/*discard=*/false);

  // Re-add the same reconstruction — images are now shared across two cycles.
  BeginWithSynthesizedReconstruction();

  // Per-reconstruction stats must match the first cycle.
  EXPECT_EQ(mapper_->NumRegFramesPerRig(), rig_counts_first);
  EXPECT_EQ(mapper_->NumRegImagesPerCamera(), camera_counts_first);
  EXPECT_FALSE(mapper_->ExistingFrameIds().empty());

  // All images were already registered in the first cycle, so they are
  // now shared. num_total_reg_images stays the same (same images, not new
  // ones), and num_shared_reg_images equals the number of re-registered images.
  EXPECT_EQ(mapper_->NumTotalRegImages(), num_reg_images_first);
  EXPECT_EQ(mapper_->NumSharedRegImages(), num_reg_images_first);

  mapper_->EndReconstruction(/*discard=*/false);
}

}  // namespace
}  // namespace colmap
