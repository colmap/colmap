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

#include "base/scene_clustering.h"

#include <set>

#include "base/database.h"
#include "base/graph_cut.h"
#include "util/random.h"

namespace colmap {

bool SceneClustering::Options::Check() const {
  CHECK_OPTION_GT(branching, 0);
  CHECK_OPTION_GE(num_overlapping_images, 0);
  CHECK_OPTION_GT(leaf_max_num_images, 0);
  return true;
}

SceneClustering::SceneClustering(const Options& options) : options_(options) {
  CHECK(options_.Check());
}

void SceneClustering::Partition(
    const std::vector<std::pair<image_pair_t, int>>& image_pairs) {
  CHECK(!root_cluster_);

  std::set<int> image_ids;
  std::vector<std::pair<int, int>> edges;
  std::vector<int> weights;
  edges.reserve(image_pairs.size());
  weights.reserve(image_pairs.size());
  for (const auto& image_pair : image_pairs) {
    image_t image_id1;
    image_t image_id2;
    Database::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
    image_ids.insert(image_id1);
    image_ids.insert(image_id2);
    edges.emplace_back(image_id1, image_id2);
    weights.push_back(image_pair.second);
  }

  root_cluster_.reset(new Cluster());
  root_cluster_->image_ids.insert(root_cluster_->image_ids.end(),
                                  image_ids.begin(), image_ids.end());
  PartitionCluster(edges, weights, root_cluster_.get());
}

void SceneClustering::PartitionCluster(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights, Cluster* cluster) {
  if (edges.size() == 0 ||
      cluster->image_ids.size() <= options_.leaf_max_num_images) {
    return;
  }

  const auto labels =
      ComputeNormalizedMinGraphCut(edges, weights, options_.branching);

  cluster->child_clusters.resize(options_.branching);
  for (const auto image_id : cluster->image_ids) {
    auto& child_cluster = cluster->child_clusters.at(labels.at(image_id));
    child_cluster.image_ids.push_back(image_id);
  }

  std::vector<std::vector<std::pair<int, int>>> child_edges(options_.branching);
  std::vector<std::vector<int>> child_weights(options_.branching);
  std::vector<std::vector<std::pair<int, int>>> overlapping_edges(
      options_.branching);
  for (size_t i = 0; i < edges.size(); ++i) {
    const int label1 = labels.at(edges[i].first);
    const int label2 = labels.at(edges[i].second);
    if (label1 == label2) {
      child_edges.at(label1).push_back(edges[i]);
      child_weights.at(label1).push_back(weights[i]);
    } else {
      overlapping_edges.at(label1).push_back(edges[i]);
      overlapping_edges.at(label2).push_back(edges[i]);
    }
  }

  for (size_t i = 0; i < options_.branching; ++i) {
    PartitionCluster(child_edges[i], child_weights[i],
                     &cluster->child_clusters[i]);
  }

  if (options_.num_overlapping_images > 0) {
    for (size_t i = 0; i < options_.branching; ++i) {
      Shuffle(overlapping_edges[i].size(), &overlapping_edges[i]);
      std::set<int> overlapping_image_ids;
      for (const auto& edge : overlapping_edges[i]) {
        if (labels.at(edge.first) == i) {
          overlapping_image_ids.insert(edge.second);
        } else {
          overlapping_image_ids.insert(edge.first);
        }
        if (overlapping_image_ids.size() >= options_.num_overlapping_images) {
          break;
        }
      }
      cluster->child_clusters[i].image_ids.insert(
          cluster->child_clusters[i].image_ids.end(),
          overlapping_image_ids.begin(), overlapping_image_ids.end());
    }
  }
}

const SceneClustering::Cluster* SceneClustering::GetRootCluster() const {
  return root_cluster_.get();
}

std::vector<const SceneClustering::Cluster*> SceneClustering::GetLeafClusters()
    const {
  std::vector<const Cluster*> leaf_clusters;

  if (!root_cluster_) {
    return leaf_clusters;
  } else if (root_cluster_->child_clusters.empty()) {
    leaf_clusters.push_back(root_cluster_.get());
    return leaf_clusters;
  }

  std::vector<const Cluster*> non_leaf_clusters;
  non_leaf_clusters.push_back(root_cluster_.get());

  while (!non_leaf_clusters.empty()) {
    const auto cluster = non_leaf_clusters.back();
    non_leaf_clusters.pop_back();

    for (const auto& child_cluster : cluster->child_clusters) {
      if (child_cluster.child_clusters.empty()) {
        leaf_clusters.push_back(&child_cluster);
      } else {
        non_leaf_clusters.push_back(&child_cluster);
      }
    }
  }

  return leaf_clusters;
}

}  // namespace colmap
