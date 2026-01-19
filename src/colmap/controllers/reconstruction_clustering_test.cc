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

#include "colmap/controllers/reconstruction_clustering.h"

#include "colmap/math/random.h"
#include "colmap/scene/synthetic.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Creates a reconstruction with two weakly connected clusters.
// The reconstruction is synthesized with `num_frames` frames, then split into
// two clusters by removing cross-cluster observations, keeping only
// `num_weak_links` 3D points that connect both clusters.
void CreateTwoWeaklyConnectedClusters(Reconstruction* reconstruction,
                                      int num_frames,
                                      int num_points3D,
                                      int num_weak_links) {
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = num_frames;
  synthetic_options.num_points3D = num_points3D;
  synthetic_options.match_config =
      SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE;
  SynthesizeDataset(synthetic_options, reconstruction);

  // Collect all frame IDs and split them into two clusters
  std::vector<frame_t> all_frames;
  for (const auto& [frame_id, frame] : reconstruction->Frames()) {
    if (frame.HasPose()) {
      all_frames.push_back(frame_id);
    }
  }
  std::sort(all_frames.begin(), all_frames.end());

  const size_t half = all_frames.size() / 2;
  std::unordered_set<frame_t> cluster1_frames(all_frames.begin(),
                                              all_frames.begin() + half);
  std::unordered_set<frame_t> cluster2_frames(all_frames.begin() + half,
                                              all_frames.end());

  // For each 3D point, randomly assign it to one cluster and remove all
  // observations from the other cluster. Keep a few points as weak links.
  std::vector<point3D_t> points_to_delete;
  int weak_link_count = 0;

  for (auto& [point3D_id, point3D] : reconstruction->Points3D()) {
    std::vector<std::pair<image_t, point2D_t>> cluster1_obs;
    std::vector<std::pair<image_t, point2D_t>> cluster2_obs;

    for (const auto& elem : point3D.track.Elements()) {
      const frame_t frame_id = reconstruction->Image(elem.image_id).FrameId();
      if (cluster1_frames.count(frame_id)) {
        cluster1_obs.emplace_back(elem.image_id, elem.point2D_idx);
      } else if (cluster2_frames.count(frame_id)) {
        cluster2_obs.emplace_back(elem.image_id, elem.point2D_idx);
      }
    }

    // If the point has observations in both clusters
    if (!cluster1_obs.empty() && !cluster2_obs.empty()) {
      // Keep a few points as weak links (with observations in both clusters)
      if (weak_link_count < num_weak_links) {
        weak_link_count++;
        continue;
      }

      // Randomly assign this point to one cluster
      bool assign_to_cluster1 = (RandomUniformInteger(0, 1) == 0);
      const auto& obs_to_remove =
          assign_to_cluster1 ? cluster2_obs : cluster1_obs;

      for (const auto& [image_id, point2D_idx] : obs_to_remove) {
        reconstruction->DeleteObservation(image_id, point2D_idx);
      }

      // If the track is now too short, mark for deletion
      if (point3D.track.Length() < 2) {
        points_to_delete.push_back(point3D_id);
      }
    }
  }

  // Delete points with insufficient observations
  for (point3D_t point3D_id : points_to_delete) {
    if (reconstruction->ExistsPoint3D(point3D_id)) {
      reconstruction->DeletePoint3D(point3D_id);
    }
  }
}

TEST(ReconstructionClustererController, EmptyReconstruction) {
  SetPRNGSeed(1);

  auto reconstruction = std::make_shared<Reconstruction>();
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  ReconstructionClusteringOptions options;
  ReconstructionClustererController controller(
      options, reconstruction, reconstruction_manager);
  EXPECT_NO_THROW(controller.Run());

  // Empty reconstruction should result in no output reconstructions
  EXPECT_EQ(reconstruction_manager->Size(), 0);
}

TEST(ReconstructionClustererController, SingleCluster) {
  SetPRNGSeed(1);

  auto reconstruction = std::make_shared<Reconstruction>();

  // Create a synthetic dataset with well-connected frames
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 200;  // More points for better covisibility
  synthetic_options.match_config =
      SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE;
  SynthesizeDataset(synthetic_options, reconstruction.get());

  EXPECT_EQ(reconstruction->NumRegFrames(), 5);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  // Use relaxed clustering options to ensure all frames stay connected
  ReconstructionClusteringOptions options;

  ReconstructionClustererController controller(
      options, reconstruction, reconstruction_manager);

  // Controller should run without crashing
  EXPECT_NO_THROW(controller.Run());

  EXPECT_EQ(reconstruction_manager->Size(), 1);
}

TEST(ReconstructionClustererController, SingleClusterWithOutlierFrames) {
  SetPRNGSeed(42);

  auto reconstruction = std::make_shared<Reconstruction>();

  // Create a well-connected reconstruction with more frames
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 10;
  synthetic_options.num_points3D = 500;
  synthetic_options.match_config =
      SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE;
  SynthesizeDataset(synthetic_options, reconstruction.get());

  const size_t total_frames = reconstruction->NumRegFrames();
  EXPECT_EQ(total_frames, 10);

  // Select the last 3 frames as outliers and remove all their 3D point
  // observations, making them isolated from the main cluster
  constexpr int kNumOutliers = 3;
  std::vector<frame_t> all_frame_ids(reconstruction->RegFrameIds().begin(),
                                     reconstruction->RegFrameIds().end());
  std::sort(all_frame_ids.begin(), all_frame_ids.end());

  std::unordered_set<frame_t> outlier_frame_ids;
  for (size_t i = total_frames - kNumOutliers; i < total_frames; i++) {
    outlier_frame_ids.insert(all_frame_ids[i]);
  }

  // Remove all 3D point observations from outlier frames
  for (const frame_t outlier_frame_id : outlier_frame_ids) {
    const Frame& frame = reconstruction->Frame(outlier_frame_id);
    for (const data_t& data_id : frame.ImageIds()) {
      const image_t image_id = data_id.id;
      Image& image = reconstruction->Image(image_id);
      const auto num_points2D = image.NumPoints2D();
      for (point2D_t point2D_idx = 0; point2D_idx < num_points2D;
           ++point2D_idx) {
        if (image.Point2D(point2D_idx).HasPoint3D()) {
          reconstruction->DeleteObservation(image_id, point2D_idx);
        }
      }
    }
  }

  const size_t main_cluster_frames = total_frames - kNumOutliers;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  ReconstructionClusteringOptions options;
  options.min_num_reg_frames = 3;

  ReconstructionClustererController controller(
      options, reconstruction, reconstruction_manager);
  controller.Run();

  // Should produce exactly one cluster containing only the main frames
  // The outlier frames should be filtered out (cluster_id = -1) because
  // they have no covisibility edges with any other frames
  EXPECT_EQ(reconstruction_manager->Size(), 1)
      << "Expected single cluster after filtering outliers";

  // The single cluster should contain only the main cluster frames
  EXPECT_EQ(reconstruction_manager->Get(0)->NumRegFrames(), main_cluster_frames)
      << "Outlier frames should not be included in the output reconstruction";
}

TEST(ReconstructionClustererController, MinNumRegFramesFilter) {
  SetPRNGSeed(1);

  auto reconstruction = std::make_shared<Reconstruction>();

  // Create a small synthetic dataset
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 2;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, reconstruction.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  // Set min_num_reg_frames higher than the number of frames
  ReconstructionClusteringOptions options;
  options.min_num_reg_frames = 5;
  ReconstructionClustererController controller(
      options, reconstruction, reconstruction_manager);
  controller.Run();

  // Should be filtered out due to min_num_reg_frames threshold
  EXPECT_EQ(reconstruction_manager->Size(), 0);
}

TEST(ReconstructionClustererController, TwoWeaklyConnectedClusters) {
  SetPRNGSeed(42);

  auto reconstruction = std::make_shared<Reconstruction>();

  // Create a reconstruction with two clusters connected by only 10 weak links
  constexpr int kNumFrames = 10;
  constexpr int kNumPoints3D = 500;
  constexpr int kNumWeakLinks = 10;
  CreateTwoWeaklyConnectedClusters(
      reconstruction.get(), kNumFrames, kNumPoints3D, kNumWeakLinks);

  EXPECT_EQ(reconstruction->NumRegFrames(), kNumFrames);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  // Use default clustering options - the algorithm should detect the weak
  // connection and split into two clusters
  ReconstructionClusteringOptions options;
  options.min_num_reg_frames = 3;

  ReconstructionClustererController controller(
      options, reconstruction, reconstruction_manager);
  controller.Run();

  // The algorithm should produce two separate reconstructions
  EXPECT_EQ(reconstruction_manager->Size(), 2)
      << "Expected two clusters from weakly connected reconstruction";
  // Check that each cluster has exactly half of the frames
  const size_t expected_frames_per_cluster = kNumFrames / 2;
  for (size_t i = 0; i < reconstruction_manager->Size(); i++) {
    EXPECT_EQ(reconstruction_manager->Get(i)->NumRegFrames(),
              expected_frames_per_cluster);
  }
}

}  // namespace
}  // namespace colmap
