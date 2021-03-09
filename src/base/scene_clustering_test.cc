// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#define TEST_NAME "base/scene_clustering"
#include "util/testing.h"

#include <set>

#include "base/database.h"
#include "base/scene_clustering.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  const std::vector<std::pair<image_t, image_t>> image_pairs;
  const std::vector<int> num_inliers;
  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 0;
  options.leaf_max_num_images = 2;
  SceneClustering scene_clustering(options);
  BOOST_CHECK(scene_clustering.GetRootCluster() == nullptr);
  scene_clustering.Partition(image_pairs, num_inliers);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids.size(), 0);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->child_clusters.size(),
                    0);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters().size(), 1);
}

BOOST_AUTO_TEST_CASE(TestOneLevel) {
  const std::vector<std::pair<image_t, image_t>> image_pairs = {{0, 1}};
  const std::vector<int> num_inliers = {10};
  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 0;
  options.leaf_max_num_images = 2;
  SceneClustering scene_clustering(options);
  BOOST_CHECK(scene_clustering.GetRootCluster() == nullptr);
  scene_clustering.Partition(image_pairs, num_inliers);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids.size(), 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[0], 0);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[1], 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->child_clusters.size(),
                    0);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters().size(), 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster(),
                    scene_clustering.GetLeafClusters()[0]);
}

BOOST_AUTO_TEST_CASE(TestTwoLevels) {
  const std::vector<std::pair<image_t, image_t>> image_pairs = {{0, 1}};
  const std::vector<int> num_inliers = {10};
  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 0;
  options.leaf_max_num_images = 1;
  SceneClustering scene_clustering(options);
  BOOST_CHECK(scene_clustering.GetRootCluster() == nullptr);
  scene_clustering.Partition(image_pairs, num_inliers);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids.size(), 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[0], 0);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[1], 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->child_clusters.size(),
                    2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters().size(), 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids[0], 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids[0], 0);
}

BOOST_AUTO_TEST_CASE(TestThreeLevels) {
  const std::vector<std::pair<image_t, image_t>> image_pairs = {{0, 1}, {0, 2}};
  const std::vector<int> num_inliers = {10, 11};
  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 0;
  options.leaf_max_num_images = 1;
  SceneClustering scene_clustering(options);
  BOOST_CHECK(scene_clustering.GetRootCluster() == nullptr);
  scene_clustering.Partition(image_pairs, num_inliers);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids.size(), 3);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[0], 0);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[1], 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[2], 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters().size(), 3);
  std::set<int> image_ids;
  image_ids.insert(scene_clustering.GetLeafClusters()[0]->image_ids[0]);
  image_ids.insert(scene_clustering.GetLeafClusters()[1]->image_ids[0]);
  image_ids.insert(scene_clustering.GetLeafClusters()[2]->image_ids[0]);
  BOOST_CHECK_EQUAL(image_ids.size(), 3);
  BOOST_CHECK(image_ids.count(0));
  BOOST_CHECK(image_ids.count(1));
  BOOST_CHECK(image_ids.count(2));
}

BOOST_AUTO_TEST_CASE(TestThreeLevelsMultipleImages) {
  const std::vector<std::pair<image_t, image_t>> image_pairs = {{0, 1}, {0, 2}};
  const std::vector<int> num_inliers = {10, 11};
  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 0;
  options.leaf_max_num_images = 2;
  SceneClustering scene_clustering(options);
  BOOST_CHECK(scene_clustering.GetRootCluster() == nullptr);
  scene_clustering.Partition(image_pairs, num_inliers);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids.size(), 3);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[0], 0);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[1], 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[2], 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters().size(), 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids.size(), 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids.size(), 2);
  std::set<int> image_ids;
  image_ids.insert(scene_clustering.GetLeafClusters()[0]->image_ids[0]);
  image_ids.insert(scene_clustering.GetLeafClusters()[1]->image_ids[0]);
  image_ids.insert(scene_clustering.GetLeafClusters()[1]->image_ids[1]);
  BOOST_CHECK_EQUAL(image_ids.size(), 3);
  BOOST_CHECK(image_ids.count(0));
  BOOST_CHECK(image_ids.count(1));
  BOOST_CHECK(image_ids.count(2));
}

BOOST_AUTO_TEST_CASE(TestOneOverlap) {
  const std::vector<std::pair<image_t, image_t>> image_pairs = {
      {0, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}, {2, 3}};
  const std::vector<int> num_inliers = {10, 11, 12, 13, 14, 15};
  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 1;
  options.leaf_max_num_images = 3;
  SceneClustering scene_clustering(options);
  BOOST_CHECK(scene_clustering.GetRootCluster() == nullptr);
  scene_clustering.Partition(image_pairs, num_inliers);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids.size(), 4);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[0], 0);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[1], 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[2], 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[3], 3);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters().size(), 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids.size(), 3);
  const std::set<image_t> image_ids0(
      scene_clustering.GetLeafClusters()[0]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[0]->image_ids.end());
  BOOST_CHECK(image_ids0.count(0));
  BOOST_CHECK(image_ids0.count(1) || image_ids0.count(2));
  BOOST_CHECK(image_ids0.count(3));
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids.size(), 3);
  const std::set<image_t> image_ids1(
      scene_clustering.GetLeafClusters()[1]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[1]->image_ids.end());
  BOOST_CHECK(image_ids1.count(1));
  BOOST_CHECK(image_ids0.count(0) || image_ids0.count(2));
  BOOST_CHECK(image_ids1.count(3));
}

BOOST_AUTO_TEST_CASE(TestTwoOverlap) {
  const std::vector<std::pair<image_t, image_t>> image_pairs = {
      {0, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}, {2, 3}};
  const std::vector<int> num_inliers = {10, 11, 12, 13, 14, 15};
  SceneClustering::Options options;
  options.branching = 2;
  options.image_overlap = 2;
  options.leaf_max_num_images = 2;
  SceneClustering scene_clustering(options);
  BOOST_CHECK(scene_clustering.GetRootCluster() == nullptr);
  scene_clustering.Partition(image_pairs, num_inliers);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids.size(), 4);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[0], 0);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[1], 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[2], 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[3], 3);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters().size(), 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids.size(), 4);
  const std::set<image_t> image_ids0(
      scene_clustering.GetLeafClusters()[0]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[0]->image_ids.end());
  BOOST_CHECK(image_ids0.count(0));
  BOOST_CHECK(image_ids0.count(1));
  BOOST_CHECK(image_ids0.count(2));
  BOOST_CHECK(image_ids0.count(3));
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids.size(), 4);
  const std::set<image_t> image_ids1(
      scene_clustering.GetLeafClusters()[1]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[1]->image_ids.end());
  BOOST_CHECK(image_ids1.count(0));
  BOOST_CHECK(image_ids1.count(1));
  BOOST_CHECK(image_ids1.count(2));
  BOOST_CHECK(image_ids1.count(3));
}

BOOST_AUTO_TEST_CASE(TestThreeFlatClusters) {
  const std::vector<std::pair<image_t, image_t>> image_pairs = {
      {0, 1}, {2, 3}, {4, 5}, {1, 2}, {3, 4}, {5, 0}, {0, 3}, {2, 5}, {4, 1}};
  const std::vector<int> num_inliers = {100, 100, 100, 10, 10, 10, 1, 1, 1};
  SceneClustering::Options options;
  options.branching = 3;
  options.image_overlap = 0;
  options.branching = 3;
  options.is_hierarchical = false;
  SceneClustering scene_clustering(options);
  BOOST_CHECK(scene_clustering.GetRootCluster() == nullptr);
  scene_clustering.Partition(image_pairs, num_inliers);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids.size(), 6);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[0], 0);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[1], 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[2], 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[3], 3);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[4], 4);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[5], 5);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters().size(), 3);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids.size(), 2);
  const std::set<image_t> image_ids0(
      scene_clustering.GetLeafClusters()[0]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[0]->image_ids.end());
  BOOST_CHECK(image_ids0.count(0));
  BOOST_CHECK(image_ids0.count(1));
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids.size(), 2);
  const std::set<image_t> image_ids1(
      scene_clustering.GetLeafClusters()[1]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[1]->image_ids.end());
  BOOST_CHECK(image_ids1.count(2));
  BOOST_CHECK(image_ids1.count(3));
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[2]->image_ids.size(), 2);
  const std::set<image_t> image_ids2(
      scene_clustering.GetLeafClusters()[2]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[2]->image_ids.end());
  BOOST_CHECK(image_ids2.count(4));
  BOOST_CHECK(image_ids2.count(5));
}

BOOST_AUTO_TEST_CASE(TestThreeFlatClustersTwoOverlap) {
  const std::vector<std::pair<image_t, image_t>> image_pairs = {
      {0, 1}, {2, 3}, {4, 5}, {1, 2}, {3, 4}, {5, 0}, {0, 3}, {2, 5}, {4, 1}};
  const std::vector<int> num_inliers = {100, 100, 100, 10, 10, 10, 1, 1, 1};
  SceneClustering::Options options;
  options.branching = 3;
  options.image_overlap = 2;
  options.branching = 3;
  options.is_hierarchical = false;
  SceneClustering scene_clustering(options);
  BOOST_CHECK(scene_clustering.GetRootCluster() == nullptr);
  scene_clustering.Partition(image_pairs, num_inliers);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids.size(), 6);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[0], 0);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[1], 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[2], 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[3], 3);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[4], 4);
  BOOST_CHECK_EQUAL(scene_clustering.GetRootCluster()->image_ids[5], 5);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters().size(), 3);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids.size(), 4);
  const std::set<image_t> image_ids0(
      scene_clustering.GetLeafClusters()[0]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[0]->image_ids.end());
  BOOST_CHECK(image_ids0.count(0));
  BOOST_CHECK(image_ids0.count(1));
  BOOST_CHECK(image_ids0.count(2));
  BOOST_CHECK(image_ids0.count(5));
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids.size(), 4);
  const std::set<image_t> image_ids1(
      scene_clustering.GetLeafClusters()[1]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[1]->image_ids.end());
  BOOST_CHECK(image_ids1.count(1));
  BOOST_CHECK(image_ids1.count(2));
  BOOST_CHECK(image_ids1.count(3));
  BOOST_CHECK(image_ids1.count(4));
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[2]->image_ids.size(), 4);
  const std::set<image_t> image_ids2(
      scene_clustering.GetLeafClusters()[2]->image_ids.begin(),
      scene_clustering.GetLeafClusters()[2]->image_ids.end());
  BOOST_CHECK(image_ids2.count(0));
  BOOST_CHECK(image_ids2.count(3));
  BOOST_CHECK(image_ids2.count(4));
  BOOST_CHECK(image_ids2.count(5));
}
