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

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"
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
            // Use a deterministic pseudo-random decision based on point3D_id
            // and observation index to decide whether to keep this observation.
            const double hash_value =
                static_cast<double>((point3D_id * 31 + i * 17) % 1000) / 1000.0;
            if (hash_value >= keep_ratio) {
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

TEST(ClusterReconstructionFrames, Empty) {
  Reconstruction reconstruction;
  ReconstructionClusteringOptions options;
  const auto cluster_ids =
      ClusterReconstructionFrames(options, reconstruction);
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
  const auto cluster_ids =
      ClusterReconstructionFrames(options, reconstruction);

  // All frames should remain since they are all well connected.
  EXPECT_EQ(reconstruction.NumRegFrames(), initial_num_reg_frames);
  // All frames should be assigned to clusters.
  EXPECT_EQ(cluster_ids.size(), initial_num_reg_frames);
}

TEST(ClusterReconstructionFrames, RemovesDisconnectedFrames) {
  // Create a reconstruction with 8 frames, all initially well-connected.
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 8;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const size_t initial_num_reg_frames = reconstruction.NumRegFrames();
  EXPECT_EQ(initial_num_reg_frames, 8);

  // Get all frame IDs and sort them.
  std::vector<frame_t> all_frame_ids;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (frame.HasPose()) {
      all_frame_ids.push_back(frame_id);
    }
  }
  std::sort(all_frame_ids.begin(), all_frame_ids.end());

  // Partition into 2 clusters: first 5 frames in cluster 0, last 3 in cluster 1
  const size_t num_to_disconnect = 3;
  std::unordered_map<frame_t, int> frame_to_cluster;
  for (size_t i = 0; i < all_frame_ids.size() - num_to_disconnect; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 0;
  }
  for (size_t i = all_frame_ids.size() - num_to_disconnect;
       i < all_frame_ids.size();
       ++i) {
    frame_to_cluster[all_frame_ids[i]] = 1;
  }

  // Partition the reconstruction to disconnect the clusters.
  PartitionFramesIntoClusters(reconstruction, frame_to_cluster, 0.0);

  // All frames should still be registered before pruning.
  EXPECT_EQ(reconstruction.NumRegFrames(), initial_num_reg_frames);

  ReconstructionClusteringOptions options;
  const auto cluster_ids =
      ClusterReconstructionFrames(options, reconstruction);

  // After pruning, only the well-connected frames should remain (5 frames).
  // The disconnected frames should be de-registered.
  const size_t expected_remaining = initial_num_reg_frames - num_to_disconnect;
  EXPECT_EQ(reconstruction.NumRegFrames(), expected_remaining);
  EXPECT_EQ(cluster_ids.size(), expected_remaining);

  // Verify disconnected frames no longer have poses.
  for (size_t i = all_frame_ids.size() - num_to_disconnect;
       i < all_frame_ids.size();
       ++i) {
    EXPECT_FALSE(reconstruction.Frame(all_frame_ids[i]).HasPose());
  }

  // Verify the remaining frames still have poses.
  for (const auto& [frame_id, cluster_id] : cluster_ids) {
    EXPECT_TRUE(reconstruction.Frame(frame_id).HasPose());
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

  // Get all frame IDs and sort them.
  std::vector<frame_t> all_frame_ids;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (frame.HasPose()) {
      all_frame_ids.push_back(frame_id);
    }
  }
  std::sort(all_frame_ids.begin(), all_frame_ids.end());

  // Partition into 2 clusters: first 8 frames in cluster 0, last 2 in cluster 1
  std::unordered_map<frame_t, int> frame_to_cluster;
  int frame_to_keep = 8;
  for (size_t i = 0; i < frame_to_keep; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 0;
  }
  for (size_t i = frame_to_keep; i < 10; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 1;
  }

  // Partition the reconstruction to disconnect the clusters.
  PartitionFramesIntoClusters(reconstruction, frame_to_cluster, 0.1);

  EXPECT_EQ(reconstruction.NumRegFrames(), initial_num_reg_frames);

  ReconstructionClusteringOptions options;
  const auto cluster_ids =
      ClusterReconstructionFrames(options, reconstruction);

  // Since the largest cluster is always assigned cluster ID 0, verify that
  // frames from the largest component have cluster ID 0.
  // For other two frames, each should only have a single image
  std::vector<std::vector<frame_t>> clusters;
  for (const auto& [frame_id, cluster_id] : cluster_ids) {
    if (clusters.size() <= static_cast<size_t>(cluster_id)) {
      clusters.resize(cluster_id + 1);
    }
    clusters[cluster_id].push_back(frame_id);
  }

  EXPECT_EQ(clusters[0].size(), frame_to_keep);
  EXPECT_EQ(clusters.size(), 3);

  // Verify frames from the largest cluster are assigned cluster ID 0.
  for (size_t i = 0; i < frame_to_keep; ++i) {
    EXPECT_EQ(cluster_ids.at(all_frame_ids[i]), 0);
  }
}

TEST(ClusterReconstructionFrames, MultipleDisconnectedClusters) {
  // Create a reconstruction with 12 frames, all initially well-connected.
  int max_cluster_size = 25;
  int other_cluster_size = 5;
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig =
      max_cluster_size + other_cluster_size * 2;
  synthetic_dataset_options.num_points3D = 400;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const size_t initial_num_reg_frames = reconstruction.NumRegFrames();
  EXPECT_EQ(initial_num_reg_frames, max_cluster_size + other_cluster_size * 2);

  // Get all frame IDs and sort them.
  std::vector<frame_t> all_frame_ids;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (frame.HasPose()) {
      all_frame_ids.push_back(frame_id);
    }
  }
  std::sort(all_frame_ids.begin(), all_frame_ids.end());

  // Partition frames into 3 clusters:
  // Cluster 0: first 25 frames (largest)
  // Cluster 1: next 5 frames
  // Cluster 2: last 5 frames (smallest)
  std::unordered_map<frame_t, int> frame_to_cluster;
  for (size_t i = 0; i < max_cluster_size; ++i) {
    frame_to_cluster[all_frame_ids[i]] = 0;
  }
  for (size_t i = max_cluster_size; i < max_cluster_size + other_cluster_size;
       ++i) {
    frame_to_cluster[all_frame_ids[i]] = 1;
  }
  for (size_t i = max_cluster_size + other_cluster_size;
       i < max_cluster_size + other_cluster_size * 2;
       ++i) {
    frame_to_cluster[all_frame_ids[i]] = 2;
  }

  // Partition the reconstruction so each cluster only sees its own 3D points.
  // Use keep_ratio=0.0 for complete separation between clusters.
  PartitionFramesIntoClusters(reconstruction, frame_to_cluster, 0.1);

  // All frames should still be registered before pruning.
  EXPECT_EQ(reconstruction.NumRegFrames(), initial_num_reg_frames);

  ReconstructionClusteringOptions options;
  const auto cluster_ids =
      ClusterReconstructionFrames(options, reconstruction);

  // Since the largest cluster is always assigned cluster ID 0, verify that
  // frames from the largest component have cluster ID 0.
  // For other two frames, each should only have a single image
  std::vector<std::vector<frame_t>> clusters;
  for (const auto& [frame_id, cluster_id] : cluster_ids) {
    if (clusters.size() <= static_cast<size_t>(cluster_id)) {
      clusters.resize(cluster_id + 1);
    }
    clusters[cluster_id].push_back(frame_id);
  }

  EXPECT_EQ(clusters[0].size(), max_cluster_size);
  EXPECT_EQ(clusters.size(), 3);
  for (size_t i = 1; i < clusters.size(); ++i) {
    EXPECT_EQ(clusters[i].size(), other_cluster_size);
  }

  // Verify frames from the largest cluster are assigned cluster ID 0.
  for (size_t i = 0; i < max_cluster_size; ++i) {
    EXPECT_EQ(cluster_ids.at(all_frame_ids[i]), 0);
  }
}

}  // namespace
}  // namespace colmap
