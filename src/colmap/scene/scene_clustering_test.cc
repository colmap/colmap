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

#include "colmap/scene/scene_clustering.h"

#include "colmap/scene/database.h"

#include <set>

#include <gtest/gtest.h>

namespace colmap {

TEST(SceneClustering, Empty) {
  const std::vector<std::pair<image_t, image_t>> image_pairs;
  const std::vector<int> num_inliers;
  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 0;
  options.leaf_max_num_images = 2;
  SceneClustering scene_clustering(options);
  EXPECT_TRUE(scene_clustering.GetRootCluster() == nullptr);
  scene_clustering.Partition(image_pairs, num_inliers);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids.size(), 0);
  EXPECT_EQ(scene_clustering.GetRootCluster()->child_clusters.size(), 0);
  EXPECT_EQ(scene_clustering.GetLeafClusters().size(), 1);
}

TEST(SceneClustering, OneLevel) {
  const std::vector<std::pair<image_t, image_t>> image_pairs = {{0, 1}};
  const std::vector<int> num_inliers = {10};
  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 0;
  options.leaf_max_num_images = 2;
  SceneClustering scene_clustering(options);
  EXPECT_TRUE(scene_clustering.GetRootCluster() == nullptr);
  scene_clustering.Partition(image_pairs, num_inliers);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids.size(), 2);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[0], 0);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[1], 1);
  EXPECT_EQ(scene_clustering.GetRootCluster()->child_clusters.size(), 0);
  EXPECT_EQ(scene_clustering.GetLeafClusters().size(), 1);
  EXPECT_EQ(scene_clustering.GetRootCluster(),
            scene_clustering.GetLeafClusters()[0]);
}

TEST(SceneClustering, ThreeFlatClusters) {
  const std::vector<std::pair<image_t, image_t>> image_pairs = {
      {0, 1}, {2, 3}, {4, 5}, {1, 2}, {3, 4}, {5, 0}, {0, 3}, {2, 5}, {4, 1}};
  const std::vector<int> num_inliers = {100, 100, 100, 10, 10, 10, 1, 1, 1};
  SceneClustering::Options options;
  options.branching = 3;
  options.image_overlap = 0;
  options.branching = 3;
  options.is_hierarchical = false;
  SceneClustering scene_clustering(options);
  EXPECT_TRUE(scene_clustering.GetRootCluster() == nullptr);
  scene_clustering.Partition(image_pairs, num_inliers);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids.size(), 6);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[0], 0);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[1], 1);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[2], 2);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[3], 3);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[4], 4);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[5], 5);
  EXPECT_EQ(scene_clustering.GetLeafClusters().size(), 3);
  EXPECT_EQ(scene_clustering.GetLeafClusters()[0]->image_ids.size(), 2);
  const std::set<image_t> image_ids0(
      scene_clustering.GetLeafClusters()[0]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[0]->image_ids.end());
  EXPECT_TRUE(image_ids0.count(0));
  EXPECT_TRUE(image_ids0.count(1));
  EXPECT_EQ(scene_clustering.GetLeafClusters()[1]->image_ids.size(), 2);
  const std::set<image_t> image_ids1(
      scene_clustering.GetLeafClusters()[1]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[1]->image_ids.end());
  EXPECT_TRUE(image_ids1.count(2));
  EXPECT_TRUE(image_ids1.count(3));
  EXPECT_EQ(scene_clustering.GetLeafClusters()[2]->image_ids.size(), 2);
  const std::set<image_t> image_ids2(
      scene_clustering.GetLeafClusters()[2]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[2]->image_ids.end());
  EXPECT_TRUE(image_ids2.count(4));
  EXPECT_TRUE(image_ids2.count(5));
}

TEST(SceneClustering, ThreeFlatClustersTwoOverlap) {
  const std::vector<std::pair<image_t, image_t>> image_pairs = {
      {0, 1}, {2, 3}, {4, 5}, {1, 2}, {3, 4}, {5, 0}, {0, 3}, {2, 5}, {4, 1}};
  const std::vector<int> num_inliers = {100, 100, 100, 10, 10, 10, 1, 1, 1};
  SceneClustering::Options options;
  options.branching = 3;
  options.image_overlap = 2;
  options.branching = 3;
  options.is_hierarchical = false;
  SceneClustering scene_clustering(options);
  EXPECT_TRUE(scene_clustering.GetRootCluster() == nullptr);
  scene_clustering.Partition(image_pairs, num_inliers);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids.size(), 6);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[0], 0);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[1], 1);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[2], 2);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[3], 3);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[4], 4);
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids[5], 5);
  EXPECT_EQ(scene_clustering.GetLeafClusters().size(), 3);
  EXPECT_EQ(scene_clustering.GetLeafClusters()[0]->image_ids.size(), 4);
  const std::set<image_t> image_ids0(
      scene_clustering.GetLeafClusters()[0]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[0]->image_ids.end());
  EXPECT_TRUE(image_ids0.count(0));
  EXPECT_TRUE(image_ids0.count(1));
  EXPECT_TRUE(image_ids0.count(2));
  EXPECT_TRUE(image_ids0.count(5));
  EXPECT_EQ(scene_clustering.GetLeafClusters()[1]->image_ids.size(), 4);
  const std::set<image_t> image_ids1(
      scene_clustering.GetLeafClusters()[1]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[1]->image_ids.end());
  EXPECT_TRUE(image_ids1.count(1));
  EXPECT_TRUE(image_ids1.count(2));
  EXPECT_TRUE(image_ids1.count(3));
  EXPECT_TRUE(image_ids1.count(4));
  EXPECT_EQ(scene_clustering.GetLeafClusters()[2]->image_ids.size(), 4);
  const std::set<image_t> image_ids2(
      scene_clustering.GetLeafClusters()[2]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[2]->image_ids.end());
  EXPECT_TRUE(image_ids2.count(0));
  EXPECT_TRUE(image_ids2.count(3));
  EXPECT_TRUE(image_ids2.count(4));
  EXPECT_TRUE(image_ids2.count(5));
}

}  // namespace colmap
