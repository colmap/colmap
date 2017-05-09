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
  CHECK_OPTION_GE(image_overlap, 0);
  CHECK_OPTION_LE(image_overlap, 1);
  CHECK_OPTION_GE(min_image_overlap, 0);
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
  CHECK_EQ(edges.size(), weights.size());

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
  std::vector<std::vector<int>> overlapping_weights(options_.branching);
  for (size_t i = 0; i < edges.size(); ++i) {
    const int label1 = labels.at(edges[i].first);
    const int label2 = labels.at(edges[i].second);
    if (label1 == label2) {
      child_edges.at(label1).push_back(edges[i]);
      child_weights.at(label1).push_back(weights[i]);
    } else {
      overlapping_edges.at(label1).push_back(edges[i]);
      overlapping_edges.at(label2).push_back(edges[i]);
      overlapping_weights.at(label1).push_back(weights[i]);
      overlapping_weights.at(label2).push_back(weights[i]);
    }
  }

  const size_t num_overlapping_images = std::max(
      static_cast<size_t>(options_.min_image_overlap),
      static_cast<size_t>(options_.image_overlap * cluster->image_ids.size()));
  if (num_overlapping_images > 0) {
    for (int i = 0; i < options_.branching; ++i) {
      // Make sure selection of adding overlapping images is random.
      Shuffle(overlapping_edges[i].size(), &overlapping_edges[i]);

      // Select overlapping edges at random and add image to cluster.
      std::set<int> overlapping_image_ids;
      for (const auto& edge : overlapping_edges[i]) {
        if (labels.at(edge.first) == i) {
          overlapping_image_ids.insert(edge.second);
        } else {
          overlapping_image_ids.insert(edge.first);
        }
        if (overlapping_image_ids.size() >= num_overlapping_images) {
          break;
        }
      }

      // Append the overlapping images to the cluster.
      cluster->child_clusters[i].image_ids.insert(
          cluster->child_clusters[i].image_ids.end(),
          overlapping_image_ids.begin(), overlapping_image_ids.end());

      // Append all edges connected to the overlapping images to the cluster.
      for (size_t j = 0; j < overlapping_edges[i].size(); ++j) {
        const auto& edge = overlapping_edges[i][j];
        if (overlapping_image_ids.count(edge.first) > 0 ||
            overlapping_image_ids.count(edge.second) > 0) {
          child_edges[i].push_back(edge);
          child_weights[i].push_back(overlapping_weights[i][j]);
        }
      }
    }
  }

  for (size_t i = 0; i < options_.branching; ++i) {
    if (cluster->image_ids.size() >
        cluster->child_clusters[i].image_ids.size()) {
      PartitionCluster(child_edges[i], child_weights[i],
                       &cluster->child_clusters[i]);
    }  // Else, the clustering does not converge and the overlap constraint
       // cannot be satisfied, so do not further cluster the child.
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
