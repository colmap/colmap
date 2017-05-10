// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define TEST_NAME "base/scene_clustering"
#include "util/testing.h"

#include "base/database.h"
#include "base/scene_clustering.h"
#include "util/random.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  SetPRNGSeed(0);
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
  SetPRNGSeed(0);
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
  SetPRNGSeed(0);
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
  SetPRNGSeed(0);
  const std::vector<std::pair<image_t, image_t>> image_pairs = {{0, 1}, {0, 2}};
  const std::vector<int> num_inliers = {10, 10};
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
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids[0], 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids[0], 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[2]->image_ids[0], 0);
}

BOOST_AUTO_TEST_CASE(TestThreeLevelsMultipleImages) {
  SetPRNGSeed(0);
  const std::vector<std::pair<image_t, image_t>> image_pairs = {{0, 1}, {0, 2}};
  const std::vector<int> num_inliers = {10, 10};
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
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids[0], 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids.size(), 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids[0], 0);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids[1], 1);
}

BOOST_AUTO_TEST_CASE(TestOneOverlap) {
  SetPRNGSeed(0);
  const std::vector<std::pair<image_t, image_t>> image_pairs = {
      {0, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}, {2, 3}};
  const std::vector<int> num_inliers = {10, 10, 10, 10, 10, 10};
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
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids[0], 0);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids[1], 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids[2], 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids.size(), 3);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids[0], 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids[1], 3);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids[2], 0);
}

BOOST_AUTO_TEST_CASE(TestTwoOverlap) {
  SetPRNGSeed(0);
  const std::vector<std::pair<image_t, image_t>> image_pairs = {
      {0, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}, {2, 3}};
  const std::vector<int> num_inliers = {10, 10, 10, 10, 10, 10};
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
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids[0], 0);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids[1], 1);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids[2], 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[0]->image_ids[3], 3);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids.size(), 4);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids[0], 2);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids[1], 3);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids[2], 0);
  BOOST_CHECK_EQUAL(scene_clustering.GetLeafClusters()[1]->image_ids[3], 1);
}
