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

#include "colmap/scene/reconstruction_clustering.h"

#include "colmap/math/random.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Helper function to partition frames into clusters based on assigned cluster
// IDs. Each cluster will only share 3D points among its member frames, and
// observations connecting frames from different clusters will be removed.
// This effectively creates isolated connected components in the covisibility
// graph.
//
// Args:
//   reconstruction: The reconstruction to modify.
//   frame_to_cluster: Map from frame_id to cluster_id. Frames in different
//                     clusters will not share any 3D points after this call.
//   keep_ratio: the ratio of kept observations across clusters
void PartitionFramesIntoClusters(
    Reconstruction& reconstruction,
    const std::unordered_map<frame_t, int>& frame_to_cluster,
    double keep_ratio = 0.1) {
  // Build reverse mapping: cluster_id -> set of frame_ids
  std::unordered_map<int, std::unordered_set<frame_t>> cluster_to_frames;
  for (const auto& [frame_id, cluster_id] : frame_to_cluster) {
    cluster_to_frames[cluster_id].insert(frame_id);
  }

  // Collect all observations to delete (image_id, point2D_idx pairs).
  // We need to collect them first because deleting while iterating is unsafe.
  std::vector<std::pair<image_t, point2D_t>> observations_to_delete;

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    // Determine which clusters observe this 3D point.
    std::unordered_map<int, std::vector<TrackElement>> cluster_observations;

    for (const auto& track_el : point3D.track.Elements()) {
      const frame_t frame_id =
          reconstruction.Image(track_el.image_id).FrameId();
      auto it = frame_to_cluster.find(frame_id);
      if (it != frame_to_cluster.end()) {
        cluster_observations[it->second].push_back(track_el);
      }
    }

    // If multiple clusters observe this point, randomly pick one cluster
    // to keep and remove observations from all other clusters.
    // However, keep a fraction (keep_ratio) of cross-cluster observations.
    if (cluster_observations.size() > 1) {
      // Randomly select a cluster to keep this point.
      std::vector<int> cluster_ids_vec;
      cluster_ids_vec.reserve(cluster_observations.size());
      for (const auto& [cluster_id, observations] : cluster_observations) {
        cluster_ids_vec.push_back(cluster_id);
      }
      std::sort(cluster_ids_vec.begin(), cluster_ids_vec.end());
      const int chosen_cluster =
          cluster_ids_vec[point3D_id % cluster_ids_vec.size()];

      // Mark observations from other clusters for deletion, but keep some
      // based on keep_ratio.
      for (const auto& [cluster_id, observations] : cluster_observations) {
        if (cluster_id != chosen_cluster) {
          for (size_t i = 0; i < observations.size(); ++i) {
            // Use a random decision to decide whether to keep this observation.
            const double random_value = RandomUniformReal<double>(0.0, 1.0);
            if (random_value >= keep_ratio) {
              observations_to_delete.emplace_back(observations[i].image_id,
                                                  observations[i].point2D_idx);
            }
          }
        }
      }
    }
  }

  // Delete the collected observations.
  for (const auto& [image_id, point2D_idx] : observations_to_delete) {
    // Check if the observation still exists (it may have been deleted
    // if the 3D point was removed due to track becoming too short).
    const auto& image = reconstruction.Image(image_id);
    if (point2D_idx < image.NumPoints2D() &&
        image.Point2D(point2D_idx).HasPoint3D()) {
      reconstruction.DeleteObservation(image_id, point2D_idx);
    }
  }
}

// Helper function to extract all registered frame IDs from a reconstruction
// and return them sorted for deterministic test behavior.
//
// NOTE: The tests in this file only cover basic clustering scenarios.
// TODO: Add tests with finer control over the connectivity graph.
std::vector<frame_t> ExtractSortedFrameIds(
    const Reconstruction& reconstruction) {
  std::vector<frame_t> frame_ids;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (frame.HasPose()) {
      frame_ids.push_back(frame_id);
    }
  }
  std::sort(frame_ids.begin(), frame_ids.end());
  return frame_ids;
}

// Helper function to build clusters from the clustering output.
// Returns a vector of sets, where result[cluster_id] contains the frame IDs
// in that cluster.
std::vector<std::unordered_set<frame_t>> BuildClustersFromOutput(
    const std::unordered_map<frame_t, int>& cluster_ids) {
  std::vector<std::unordered_set<frame_t>> clusters;
  for (const auto& [frame_id, cluster_id] : cluster_ids) {
    if (cluster_id == -1) continue;
    if (clusters.size() <= static_cast<size_t>(cluster_id)) {
      clusters.resize(cluster_id + 1);
    }
    clusters[cluster_id].insert(frame_id);
  }
  return clusters;
}

TEST(ClusterReconstructionFrames, Empty) {
  Reconstruction reconstruction;
  ReconstructionClusteringOptions options;
  const auto cluster_ids = ClusterReconstructionFrames(options, reconstruction);
  EXPECT_TRUE(cluster_ids.empty());
}

TEST(ClusterReconstructionFrames, WellConnectedReconstruction) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const size_t initial_num_reg_frames = reconstruction.NumRegFrames();
  EXPECT_EQ(initial_num_reg_frames, 5);

  ReconstructionClusteringOptions options;
  const auto cluster_ids = ClusterReconstructionFrames(options, reconstruction);

  // All frames should be assigned cluster 0
  for (const auto& [frame_id, cluster_id] : cluster_ids) {
    EXPECT_EQ(cluster_id, 0);
  }
}

TEST(ClusterReconstructionFrames, WeaklyConnectedReconstruction) {
  // Create a reconstruction with very few 3D points (10 total).
  // With so few shared observations, the covisibility between frames is weak
  // and each frame should be assigned to its own cluster.
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 10;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const size_t kNumFrames = reconstruction.NumRegFrames();
  EXPECT_EQ(kNumFrames, 10);

  ReconstructionClusteringOptions options;
  options.min_edge_weight_threshold =
      synthetic_dataset_options.num_points3D + 1;
  const auto cluster_ids = ClusterReconstructionFrames(options, reconstruction);

  // All frames should be assigned to clusters.
  EXPECT_EQ(cluster_ids.size(), kNumFrames);

  // Each frame should get id -1
  std::unordered_set<int> unique_cluster_ids;
  for (const auto& [frame_id, cluster_id] : cluster_ids) {
    EXPECT_EQ(cluster_id, -1);
  }
}

TEST(ClusterReconstructionFrames, OneMajorConnectedComponent) {
  // Create a reconstruction with 10 frames, all initially well-connected.
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 250;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const size_t initial_num_reg_frames = reconstruction.NumRegFrames();
  EXPECT_EQ(initial_num_reg_frames, 10);

  // Get all frame IDs sorted for deterministic behavior.
  const std::vector<frame_t> all_frame_ids =
      ExtractSortedFrameIds(reconstruction);

  // Partition into one cluster and independent frames:
  // first 8 frames in cluster 0, other frames are independent
  std::unordered_map<frame_t, int> frame_to_cluster;
  const size_t kLargeClusterSize = 8;
  for (size_t i = 0; i < kLargeClusterSize; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 0;
  }
  for (size_t i = kLargeClusterSize; i < 10; ++i) {
    frame_to_cluster[all_frame_ids[i]] = i - kLargeClusterSize + 1;
  }

  // Partition the reconstruction to disconnect the clusters.
  // With keep_ratio=0.1, some weak connections may remain.
  PartitionFramesIntoClusters(reconstruction, frame_to_cluster, 0.1);

  EXPECT_EQ(reconstruction.NumRegFrames(), initial_num_reg_frames);

  ReconstructionClusteringOptions options;
  const auto cluster_ids = ClusterReconstructionFrames(options, reconstruction);

  // All frames should be assigned to clusters.
  EXPECT_EQ(cluster_ids.size(), initial_num_reg_frames);

  // Build the resulting clusters.
  const auto clusters = BuildClustersFromOutput(cluster_ids);

  // Find the largest cluster.
  size_t largest_cluster_idx = 0;
  for (size_t i = 1; i < clusters.size(); ++i) {
    if (clusters[i].size() > clusters[largest_cluster_idx].size()) {
      largest_cluster_idx = i;
    }
  }
  // The largest cluster should be cluster 0.
  EXPECT_EQ(largest_cluster_idx, 0);

  // The largest cluster should have exactly kLargeClusterSize frames.
  EXPECT_EQ(clusters[largest_cluster_idx].size(), kLargeClusterSize);

  // Other clusters should be single-frame clusters.
  for (size_t i = 0; i < clusters.size(); ++i) {
    if (i != largest_cluster_idx) {
      EXPECT_EQ(clusters[i].size(), 1);
    }
  }
}

TEST(ClusterReconstructionFrames, MultipleWeaklyConnectedClusters) {
  // Create a reconstruction with frames that will be partitioned into
  // weakly connected clusters (some cross-cluster connections remain).
  const size_t kCluster0Size = 25;
  const size_t kCluster1Size = 5;
  const size_t kCluster2Size = 4;
  const size_t kTotalFrames = kCluster0Size + kCluster1Size + kCluster2Size;

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = kTotalFrames;
  synthetic_dataset_options.num_points3D = 400;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  EXPECT_EQ(reconstruction.NumRegFrames(), kTotalFrames);

  // Get all frame IDs sorted for deterministic behavior.
  const std::vector<frame_t> all_frame_ids =
      ExtractSortedFrameIds(reconstruction);

  // Partition frames into 3 clusters.
  std::unordered_map<frame_t, int> frame_to_cluster;
  std::vector<std::unordered_set<frame_t>> expected_clusters(3);

  for (size_t i = 0; i < kCluster0Size; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 0;
    expected_clusters[0].insert(all_frame_ids[i]);
  }
  for (size_t i = kCluster0Size; i < kCluster0Size + kCluster1Size; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 1;
    expected_clusters[1].insert(all_frame_ids[i]);
  }
  for (size_t i = kCluster0Size + kCluster1Size; i < kTotalFrames; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 2;
    expected_clusters[2].insert(all_frame_ids[i]);
  }

  // Partition with keep_ratio=0.1 to leave some weak connections.
  PartitionFramesIntoClusters(reconstruction, frame_to_cluster, 0.1);

  // All frames should still be registered.
  EXPECT_EQ(reconstruction.NumRegFrames(), kTotalFrames);

  ReconstructionClusteringOptions options;
  const auto cluster_ids = ClusterReconstructionFrames(options, reconstruction);

  // All frames should be assigned to clusters.
  EXPECT_EQ(cluster_ids.size(), kTotalFrames);

  // Build the resulting clusters.
  const auto result_clusters = BuildClustersFromOutput(cluster_ids);

  // Should have exactly 3 clusters
  EXPECT_EQ(result_clusters.size(), 3);

  // Sort expected cluster by size for comparison.
  // Result clusters are already sorted by size due to implementation.
  std::sort(expected_clusters.begin(),
            expected_clusters.end(),
            [](const auto& a, const auto& b) { return a.size() > b.size(); });

  // Verify that the clusters match exactly.
  EXPECT_EQ(result_clusters[0], expected_clusters[0]);
  EXPECT_EQ(result_clusters[1], expected_clusters[1]);
  EXPECT_EQ(result_clusters[2], expected_clusters[2]);
}

TEST(ClusterReconstructionFrames, MultipleDisjointClusters) {
  // Create a reconstruction with completely disjoint clusters (no shared
  // observations between clusters). Verify that the clustering exactly
  // matches the original partition.
  const size_t kCluster0Size = 10;
  const size_t kCluster1Size = 8;
  const size_t kCluster2Size = 6;
  const size_t kTotalFrames = kCluster0Size + kCluster1Size + kCluster2Size;

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = kTotalFrames;
  synthetic_dataset_options.num_points3D = 500;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  EXPECT_EQ(reconstruction.NumRegFrames(), kTotalFrames);

  // Get all frame IDs sorted for deterministic behavior.
  const std::vector<frame_t> all_frame_ids =
      ExtractSortedFrameIds(reconstruction);

  // Partition frames into 3 completely disjoint clusters.
  std::unordered_map<frame_t, int> frame_to_cluster;
  std::vector<std::unordered_set<frame_t>> expected_clusters(3);

  for (size_t i = 0; i < kCluster0Size; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 0;
    expected_clusters[0].insert(all_frame_ids[i]);
  }
  for (size_t i = kCluster0Size; i < kCluster0Size + kCluster1Size; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 1;
    expected_clusters[1].insert(all_frame_ids[i]);
  }
  for (size_t i = kCluster0Size + kCluster1Size; i < kTotalFrames; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 2;
    expected_clusters[2].insert(all_frame_ids[i]);
  }

  // Use keep_ratio=0.0 to create completely disjoint clusters with no
  // shared observations between them.
  PartitionFramesIntoClusters(reconstruction, frame_to_cluster, 0.0);

  // All frames should still be registered.
  EXPECT_EQ(reconstruction.NumRegFrames(), kTotalFrames);

  ReconstructionClusteringOptions options;
  const auto cluster_ids = ClusterReconstructionFrames(options, reconstruction);

  // All frames should be assigned to clusters.
  EXPECT_EQ(cluster_ids.size(), kTotalFrames);

  // Build the resulting clusters from the clustering output.
  const auto result_clusters = BuildClustersFromOutput(cluster_ids);

  // Should have exactly 3 clusters.
  EXPECT_EQ(result_clusters.size(), 3);

  // Sort expected cluster by size for comparison.
  std::sort(expected_clusters.begin(),
            expected_clusters.end(),
            [](const auto& a, const auto& b) { return a.size() > b.size(); });

  // The clusters should match exactly.
  EXPECT_EQ(result_clusters[0], expected_clusters[0]);
  EXPECT_EQ(result_clusters[1], expected_clusters[1]);
  EXPECT_EQ(result_clusters[2], expected_clusters[2]);
}

// Tests with non-trivial rigs (multiple cameras per frame).

TEST(ClusterReconstructionFrames, RigOneMajorConnectedComponent) {
  // Create a reconstruction with 10 frames from a rig with 3 cameras each.
  // Test that clustering correctly handles multi-camera rigs.
  // Note: With multi-camera rigs, covisibility between frames is stronger
  // because 3D points are often visible from multiple cameras in the same
  // frame.
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 300;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const size_t initial_num_reg_frames = reconstruction.NumRegFrames();
  EXPECT_EQ(initial_num_reg_frames, 10);
  // Should have 30 images (10 frames * 3 cameras).
  EXPECT_EQ(reconstruction.NumRegImages(), 30);

  // Get all frame IDs sorted for deterministic behavior.
  const std::vector<frame_t> all_frame_ids =
      ExtractSortedFrameIds(reconstruction);

  // Partition into one large cluster and independent frames:
  // first 7 frames in cluster 0, other 3 frames are independent.
  std::unordered_map<frame_t, int> frame_to_cluster;
  std::unordered_set<frame_t> expected_large_cluster;
  const size_t kLargeClusterSize = 7;
  for (size_t i = 0; i < kLargeClusterSize; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 0;
    expected_large_cluster.insert(all_frame_ids[i]);
  }
  for (size_t i = kLargeClusterSize; i < 10; ++i) {
    frame_to_cluster[all_frame_ids[i]] = i - kLargeClusterSize + 1;
  }

  // Use keep_ratio=0.0 to completely disconnect clusters.
  // With multi-camera rigs, weak connections are harder to break.
  PartitionFramesIntoClusters(reconstruction, frame_to_cluster, 0.0);

  EXPECT_EQ(reconstruction.NumRegFrames(), initial_num_reg_frames);

  ReconstructionClusteringOptions options;
  const auto cluster_ids = ClusterReconstructionFrames(options, reconstruction);

  // All frames should be assigned to clusters.
  EXPECT_EQ(cluster_ids.size(), initial_num_reg_frames);

  // Build the resulting clusters.
  const auto clusters = BuildClustersFromOutput(cluster_ids);

  // Should be only 1 large cluster
  EXPECT_EQ(clusters.size(), 1);
  // The largest cluster (cluster 0) should have exactly kLargeClusterSize.
  EXPECT_EQ(clusters[0].size(), kLargeClusterSize);
  EXPECT_EQ(clusters[0], expected_large_cluster);

  for (size_t i = kLargeClusterSize; i < 10; ++i) {
    // Other frames should not be in any cluster.
    EXPECT_EQ(cluster_ids.at(all_frame_ids[i]), -1);
  }
}

TEST(ClusterReconstructionFrames, RigMultipleWeaklyConnectedClusters) {
  // Create a reconstruction with frames from a rig with 2 cameras each.
  // Partition into 3 clusters. With multi-camera rigs, the covisibility
  // between frames is higher, so we use very low keep_ratio to ensure
  // cluster separation while still leaving some weak connections.
  const size_t kCluster0Size = 30;
  const size_t kCluster1Size = 4;
  const size_t kCluster2Size = 3;
  const size_t kTotalFrames = kCluster0Size + kCluster1Size + kCluster2Size;

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = kTotalFrames;
  synthetic_dataset_options.num_points3D = 500;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  EXPECT_EQ(reconstruction.NumRegFrames(), kTotalFrames);
  // Should have 60 images (30 frames * 2 cameras).
  EXPECT_EQ(reconstruction.NumRegImages(), kTotalFrames * 2);

  // Get all frame IDs sorted for deterministic behavior.
  const std::vector<frame_t> all_frame_ids =
      ExtractSortedFrameIds(reconstruction);

  // Partition frames into 3 clusters.
  std::unordered_map<frame_t, int> frame_to_cluster;
  std::vector<std::unordered_set<frame_t>> expected_clusters(3);

  for (size_t i = 0; i < kCluster0Size; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 0;
    expected_clusters[0].insert(all_frame_ids[i]);
  }
  for (size_t i = kCluster0Size; i < kCluster0Size + kCluster1Size; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 1;
    expected_clusters[1].insert(all_frame_ids[i]);
  }
  for (size_t i = kCluster0Size + kCluster1Size; i < kTotalFrames; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 2;
    expected_clusters[2].insert(all_frame_ids[i]);
  }

  // Use keep_ratio=0.05 to create disjoint clusters.
  PartitionFramesIntoClusters(reconstruction, frame_to_cluster, 0.05);

  // All frames should still be registered.
  EXPECT_EQ(reconstruction.NumRegFrames(), kTotalFrames);

  ReconstructionClusteringOptions options;
  const auto cluster_ids = ClusterReconstructionFrames(options, reconstruction);

  // All frames should be assigned to clusters.
  EXPECT_EQ(cluster_ids.size(), kTotalFrames);

  // Build the resulting clusters.
  const auto result_clusters = BuildClustersFromOutput(cluster_ids);

  // Should have exactly 3 clusters.
  EXPECT_EQ(result_clusters.size(), 3);

  // Sort expected clusters by size for comparison.
  std::sort(expected_clusters.begin(),
            expected_clusters.end(),
            [](const auto& a, const auto& b) { return a.size() > b.size(); });

  // Verify that the clusters match exactly.
  EXPECT_EQ(result_clusters[0], expected_clusters[0]);
  EXPECT_EQ(result_clusters[1], expected_clusters[1]);
  EXPECT_EQ(result_clusters[2], expected_clusters[2]);
}

TEST(ClusterReconstructionFrames, RigMultipleDisjointClusters) {
  // Create a reconstruction with frames from a rig with 4 cameras each.
  // Partition into 3 completely disjoint clusters.
  const size_t kCluster0Size = 8;
  const size_t kCluster1Size = 6;
  const size_t kCluster2Size = 4;
  const size_t kTotalFrames = kCluster0Size + kCluster1Size + kCluster2Size;
  const int kCamerasPerRig = 4;

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = kCamerasPerRig;
  synthetic_dataset_options.num_frames_per_rig = kTotalFrames;
  synthetic_dataset_options.num_points3D = 600;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  EXPECT_EQ(reconstruction.NumRegFrames(), kTotalFrames);
  // Should have 72 images (18 frames * 4 cameras).
  EXPECT_EQ(reconstruction.NumRegImages(), kTotalFrames * kCamerasPerRig);

  // Get all frame IDs sorted for deterministic behavior.
  const std::vector<frame_t> all_frame_ids =
      ExtractSortedFrameIds(reconstruction);

  // Partition frames into 3 completely disjoint clusters.
  std::unordered_map<frame_t, int> frame_to_cluster;
  std::vector<std::unordered_set<frame_t>> expected_clusters(3);

  for (size_t i = 0; i < kCluster0Size; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 0;
    expected_clusters[0].insert(all_frame_ids[i]);
  }
  for (size_t i = kCluster0Size; i < kCluster0Size + kCluster1Size; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 1;
    expected_clusters[1].insert(all_frame_ids[i]);
  }
  for (size_t i = kCluster0Size + kCluster1Size; i < kTotalFrames; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 2;
    expected_clusters[2].insert(all_frame_ids[i]);
  }

  // Use keep_ratio=0.0 to create completely disjoint clusters.
  PartitionFramesIntoClusters(reconstruction, frame_to_cluster, 0.0);

  // All frames should still be registered.
  EXPECT_EQ(reconstruction.NumRegFrames(), kTotalFrames);

  ReconstructionClusteringOptions options;
  const auto cluster_ids = ClusterReconstructionFrames(options, reconstruction);

  // All frames should be assigned to clusters.
  EXPECT_EQ(cluster_ids.size(), kTotalFrames);

  // Build the resulting clusters from the clustering output.
  const auto result_clusters = BuildClustersFromOutput(cluster_ids);

  // Should have exactly 3 clusters.
  EXPECT_EQ(result_clusters.size(), 3);

  // Sort expected clusters by size for comparison.
  std::sort(expected_clusters.begin(),
            expected_clusters.end(),
            [](const auto& a, const auto& b) { return a.size() > b.size(); });

  // The clusters should match exactly.
  EXPECT_EQ(result_clusters[0], expected_clusters[0]);
  EXPECT_EQ(result_clusters[1], expected_clusters[1]);
  EXPECT_EQ(result_clusters[2], expected_clusters[2]);
}

}  // namespace
}  // namespace colmap
