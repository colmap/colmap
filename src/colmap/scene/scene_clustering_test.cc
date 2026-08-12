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

#include "colmap/scene/scene_clustering.h"

#include <algorithm>
#include <initializer_list>
#include <set>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

struct SceneGraph {
  std::vector<std::pair<image_t, image_t>> image_pairs;
  std::vector<int> num_inliers;
};

// Image connectivity graph:
//
//               100           10
//        (0) -------- (1) -------- (2)
//         |                         |
//        10                        100
//         |                         |
//        (5) -------- (4) -------- (3)
//              100           10
//
// Weak cross-cluster connections (1 inlier):
//        (0) -------- (3)
//        (2) -------- (5)
//        (4) -------- (1)
SceneGraph MakeSceneGraphOneLevel() {
  static const SceneGraph graph = {
      {{0, 1}, {2, 3}, {4, 5}, {1, 2}, {3, 4}, {5, 0}, {0, 3}, {2, 5}, {4, 1}},
      {100, 100, 100, 10, 10, 10, 1, 1, 1}};
  return graph;
}

// Image connectivity graph:
//
//               100          50          100
//        (0) -------- (1) -------- (2) -------- (3)
//         |                                      |
//        10                                      10
//         |                                      |
//        (7) -------- (6) -------- (5) -------- (4)
//              100          50          100
//
// Weak cross-cluster connections (1 inlier):
//        (0) -------- (4)
//        (1) -------- (6)
//        (2) -------- (5)
//        (3) -------- (7)
SceneGraph MakeSceneGraphTwoLevel() {
  static const SceneGraph graph = {
      {{0, 1},
       {1, 2},
       {2, 3},
       {4, 5},
       {5, 6},
       {6, 7},
       {0, 7},
       {3, 4},
       {0, 4},
       {1, 6},
       {2, 5},
       {3, 7}},
      {100, 50, 100, 100, 50, 100, 10, 10, 1, 1, 1, 1}};
  return graph;
}

// Path graph with three levels of progressively weaker connections.
SceneGraph MakeSceneGraphThreeLevel() {
  static const SceneGraph graph = {{{0, 1},
                                    {1, 2},
                                    {2, 3},
                                    {3, 4},
                                    {4, 5},
                                    {5, 6},
                                    {6, 7},
                                    {7, 8},
                                    {8, 9},
                                    {9, 10},
                                    {10, 11},
                                    {11, 12},
                                    {12, 13},
                                    {13, 14},
                                    {14, 15}},
                                   {1000,
                                    100,
                                    1000,
                                    10,
                                    1000,
                                    100,
                                    1000,
                                    1,
                                    1000,
                                    100,
                                    1000,
                                    10,
                                    1000,
                                    100,
                                    1000}};
  return graph;
}

// The first six images form one top-level cluster and the last six images form
// the other. Image 6 overlaps the first cluster. Its strongest individual
// connection is to image 0, but its combined connection to images 3 and 4 is
// stronger.
SceneGraph MakeSceneGraphAggregateOverlap() {
  static const SceneGraph graph = {{{0, 1},
                                    {1, 2},
                                    {2, 3},
                                    {3, 4},
                                    {4, 5},
                                    {6, 7},
                                    {7, 8},
                                    {8, 9},
                                    {9, 10},
                                    {10, 11},
                                    {6, 0},
                                    {6, 3},
                                    {6, 4}},
                                   {10000,
                                    10000,
                                    1,
                                    10000,
                                    10000,
                                    20000,
                                    20000,
                                    20000,
                                    20000,
                                    20000,
                                    100,
                                    90,
                                    90}};
  return graph;
}

MATCHER_P(UnorderedClustersEqMatcher,
          expected_clusters,
          "is equal to the expected clusters (ignoring order): " +
              ::testing::PrintToString(expected_clusters)) {
  std::vector<std::set<image_t>> actual;
  for (const auto& cluster : arg) {
    actual.emplace_back(cluster.begin(), cluster.end());
  }
  std::vector<std::set<image_t>> expected;
  for (const auto& cluster : expected_clusters) {
    expected.emplace_back(cluster.begin(), cluster.end());
  }
  std::sort(actual.begin(), actual.end());
  std::sort(expected.begin(), expected.end());
  return actual == expected;
}

template <typename... ClusterTypes>
auto UnorderedClustersEq(std::initializer_list<ClusterTypes>... clusters) {
  std::vector<std::set<image_t>> expected;
  expected.reserve(sizeof...(clusters));
  (expected.emplace_back(clusters.begin(), clusters.end()), ...);
  return UnorderedClustersEqMatcher(expected);
}

std::vector<std::set<image_t>> GetLeafImageSets(
    const std::vector<const SceneClustering::Cluster*>& leaves) {
  std::vector<std::set<image_t>> leaf_image_sets;
  leaf_image_sets.reserve(leaves.size());
  for (const auto* leaf : leaves) {
    leaf_image_sets.emplace_back(leaf->image_ids.begin(),
                                 leaf->image_ids.end());
  }
  return leaf_image_sets;
}

std::vector<std::set<image_t>> GetChildImageSets(
    const SceneClustering::Cluster& cluster) {
  std::vector<std::set<image_t>> child_image_sets;
  child_image_sets.reserve(cluster.child_clusters.size());
  for (const auto& child : cluster.child_clusters) {
    child_image_sets.emplace_back(child.image_ids.begin(),
                                  child.image_ids.end());
  }
  return child_image_sets;
}

TEST(SceneClustering, Empty) {
  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 0;
  options.leaf_max_num_images = 2;
  SceneClustering scene_clustering(options);
  EXPECT_EQ(scene_clustering.GetRootCluster(), nullptr);
  scene_clustering.Partition({}, {});
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids.size(), 0);
  EXPECT_EQ(scene_clustering.GetRootCluster()->child_clusters.size(), 0);
  EXPECT_EQ(scene_clustering.GetLeafClusters().size(), 1);
}

TEST(SceneClustering, OneLevel) {
  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 0;
  options.leaf_max_num_images = 2;
  SceneClustering scene_clustering(options);
  EXPECT_EQ(scene_clustering.GetRootCluster(), nullptr);
  scene_clustering.Partition({{0, 1}}, {10});
  EXPECT_EQ(scene_clustering.GetRootCluster()->image_ids.size(), 2);
  EXPECT_THAT(scene_clustering.GetRootCluster()->image_ids,
              ::testing::UnorderedElementsAre(0, 1));
  EXPECT_EQ(scene_clustering.GetRootCluster()->child_clusters.size(), 0);
  EXPECT_EQ(scene_clustering.GetLeafClusters().size(), 1);
  EXPECT_EQ(scene_clustering.GetRootCluster(),
            scene_clustering.GetLeafClusters()[0]);
}

TEST(SceneClustering, ThreeFlatClusters) {
  const SceneGraph graph = MakeSceneGraphOneLevel();

  SceneClustering::Options options;
  options.branching = 3;
  options.image_overlap = 0;
  options.is_hierarchical = false;
  SceneClustering scene_clustering(options);
  EXPECT_EQ(scene_clustering.GetRootCluster(), nullptr);
  scene_clustering.Partition(graph.image_pairs, graph.num_inliers);

  EXPECT_EQ(scene_clustering.GetLeafClusters().size(), 3);
  EXPECT_THAT(GetLeafImageSets(scene_clustering.GetLeafClusters()),
              UnorderedClustersEq({0, 1}, {2, 3}, {4, 5}));
}

TEST(SceneClustering, ThreeFlatClustersTwoOverlap) {
  const SceneGraph graph = MakeSceneGraphOneLevel();

  SceneClustering::Options options;
  options.branching = 3;
  options.image_overlap = 2;
  options.is_hierarchical = false;
  SceneClustering scene_clustering(options);
  EXPECT_EQ(scene_clustering.GetRootCluster(), nullptr);
  scene_clustering.Partition(graph.image_pairs, graph.num_inliers);

  EXPECT_EQ(scene_clustering.GetLeafClusters().size(), 3);
  EXPECT_THAT(GetLeafImageSets(scene_clustering.GetLeafClusters()),
              UnorderedClustersEq({0, 1, 2, 5}, {1, 2, 3, 4}, {0, 3, 4, 5}));
}

TEST(SceneClustering, HierarchicalTwoLevelsNoOverlap) {
  const SceneGraph graph = MakeSceneGraphTwoLevel();

  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 0;
  options.leaf_max_num_images = 2;
  SceneClustering scene_clustering(options);
  EXPECT_EQ(scene_clustering.GetRootCluster(), nullptr);
  scene_clustering.Partition(graph.image_pairs, graph.num_inliers);

  EXPECT_EQ(scene_clustering.GetLeafClusters().size(), 4);
  EXPECT_THAT(GetLeafImageSets(scene_clustering.GetLeafClusters()),
              UnorderedClustersEq({0, 1}, {2, 3}, {4, 5}, {6, 7}));
}

TEST(SceneClustering, HierarchicalTwoLevelsWithOverlap) {
  const SceneGraph graph = MakeSceneGraphTwoLevel();

  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 2;
  options.leaf_max_num_images = 3;
  SceneClustering scene_clustering(options);
  EXPECT_EQ(scene_clustering.GetRootCluster(), nullptr);
  scene_clustering.Partition(graph.image_pairs, graph.num_inliers);

  EXPECT_EQ(scene_clustering.GetLeafClusters().size(), 4);
  EXPECT_THAT(
      GetLeafImageSets(scene_clustering.GetLeafClusters()),
      UnorderedClustersEq(
          {0, 1, 2, 3, 4}, {0, 1, 2, 4, 7}, {0, 3, 4, 5, 6}, {0, 4, 5, 6, 7}));
}

TEST(SceneClustering, HierarchicalThreeLevelsWithOverlap) {
  const SceneGraph graph = MakeSceneGraphThreeLevel();

  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 1;
  options.leaf_max_num_images = 3;
  SceneClustering scene_clustering(options);
  scene_clustering.Partition(graph.image_pairs, graph.num_inliers);

  // Top-level overlap images are partitioned twice after they are inherited,
  // exercising connectivity beyond the single generation covered above.
  EXPECT_EQ(scene_clustering.GetLeafClusters().size(), 8);
  EXPECT_THAT(GetLeafImageSets(scene_clustering.GetLeafClusters()),
              UnorderedClustersEq({0, 1, 2},
                                  {1, 2, 3, 4},
                                  {3, 4, 5, 6},
                                  {5, 6, 7, 8},
                                  {7, 8, 9, 10},
                                  {9, 10, 11, 12},
                                  {11, 12, 13, 14},
                                  {13, 14, 15}));
}

TEST(SceneClustering, HierarchicalOverlapUsesAllConnections) {
  const SceneGraph graph = MakeSceneGraphAggregateOverlap();

  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 1;
  options.leaf_max_num_images = 4;
  SceneClustering scene_clustering(options);
  scene_clustering.Partition(graph.image_pairs, graph.num_inliers);

  const SceneClustering::Cluster* target_cluster = nullptr;
  for (const auto& child : scene_clustering.GetRootCluster()->child_clusters) {
    const std::set<image_t> image_ids(child.image_ids.begin(),
                                      child.image_ids.end());
    if (image_ids.count(0) && image_ids.count(1) && image_ids.count(2) &&
        image_ids.count(3) && image_ids.count(4) && image_ids.count(5)) {
      target_cluster = &child;
      break;
    }
  }
  ASSERT_NE(target_cluster, nullptr);
  const auto child_image_sets = GetChildImageSets(*target_cluster);
  ASSERT_EQ(child_image_sets.size(), 2);

  // Image 6 does not simply follow its strongest connection to image 0. Its
  // full connectivity to images 3 and 4 participates in the graph cut. The
  // exact partition is implementation-dependent in METIS.
  EXPECT_TRUE(std::any_of(child_image_sets.begin(),
                          child_image_sets.end(),
                          [](const std::set<image_t>& image_ids) {
                            return image_ids.count(3) && image_ids.count(4) &&
                                   image_ids.count(6);
                          }));
}

}  // namespace
}  // namespace colmap
