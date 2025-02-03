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

#include "colmap/math/graph_cut.h"
#include "colmap/math/random.h"

#include <set>

namespace colmap {

bool SceneClustering::Options::Check() const {
  CHECK_OPTION_GT(branching, 0);
  CHECK_OPTION_GE(image_overlap, 0);
  return true;
}

SceneClustering::SceneClustering(const Options& options) : options_(options) {
  THROW_CHECK(options_.Check());
}

void SceneClustering::Partition(
    const std::vector<std::pair<image_t, image_t>>& image_pairs,
    const std::vector<int>& num_inliers) {
  THROW_CHECK(!root_cluster_);
  THROW_CHECK_EQ(image_pairs.size(), num_inliers.size());

  std::set<image_t> image_ids;
  std::vector<std::pair<int, int>> edges;
  edges.reserve(image_pairs.size());
  for (const auto& image_pair : image_pairs) {
    image_ids.insert(image_pair.first);
    image_ids.insert(image_pair.second);
    edges.emplace_back(image_pair.first, image_pair.second);
  }

  root_cluster_ = std::make_unique<Cluster>();
  root_cluster_->image_ids.insert(
      root_cluster_->image_ids.end(), image_ids.begin(), image_ids.end());
  if (options_.is_hierarchical) {
    PartitionHierarchicalCluster(edges, num_inliers, root_cluster_.get());
  } else {
    PartitionFlatCluster(edges, num_inliers);
  }
}

void SceneClustering::PartitionHierarchicalCluster(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights,
    Cluster* cluster) {
  THROW_CHECK_EQ(edges.size(), weights.size());

  // If the cluster is small enough, we return from the recursive clustering.
  if (edges.empty() || cluster->image_ids.size() <=
                           static_cast<size_t>(options_.leaf_max_num_images)) {
    return;
  }

  // Partition the cluster using a normalized cut on the scene graph.
  const auto labels =
      ComputeNormalizedMinGraphCut(edges, weights, options_.branching);

  // Assign the images to the clustered child clusters.
  cluster->child_clusters.resize(options_.branching);
  for (const auto image_id : cluster->image_ids) {
    if (labels.count(image_id)) {
      auto& child_cluster = cluster->child_clusters.at(labels.at(image_id));
      child_cluster.image_ids.push_back(image_id);
    } else {
      LOG(WARNING) << "Graph cut failed to assign cluster label to image "
                   << image_id << "; assigning to cluster 0";
      cluster->child_clusters.at(0).image_ids.push_back(image_id);
    }
  }

  // Collect the edges based on whether they are inter or intra child clusters.
  std::vector<std::vector<std::pair<int, int>>> child_edges(options_.branching);
  std::vector<std::vector<int>> child_weights(options_.branching);
  std::vector<std::vector<std::pair<std::pair<int, int>, int>>>
      overlapping_edges(options_.branching);
  for (size_t i = 0; i < edges.size(); ++i) {
    const int label1 = labels.at(edges[i].first);
    const int label2 = labels.at(edges[i].second);
    if (label1 == label2) {
      child_edges.at(label1).push_back(edges[i]);
      child_weights.at(label1).push_back(weights[i]);
    } else {
      overlapping_edges.at(label1).emplace_back(edges[i], weights[i]);
      overlapping_edges.at(label2).emplace_back(edges[i], weights[i]);
    }
  }

  // Recursively partition all the child clusters.
  for (int i = 0; i < options_.branching; ++i) {
    // Skip empty clusters or clusters where the current cluster has as many
    // images as its child to avoid infinite loops. This can happen because
    // the normalized cut sometimes decides to put all images into one
    // cluster.
    if (cluster->child_clusters[i].image_ids.empty() ||
        cluster->child_clusters[i].image_ids.size() ==
            cluster->image_ids.size()) {
      continue;
    }

    PartitionHierarchicalCluster(
        child_edges[i], child_weights[i], &cluster->child_clusters[i]);
  }

  // Remove empty clusters.
  cluster->child_clusters.erase(
      std::remove_if(cluster->child_clusters.begin(),
                     cluster->child_clusters.end(),
                     [](const Cluster& childCluster) {
                       return childCluster.image_ids.empty();
                     }),
      cluster->child_clusters.end());

  // If the child cluster is the same as the current cluster, it is redundant
  // and we can remove it.
  if (cluster->child_clusters.size() == 1 &&
      cluster->image_ids.size() ==
          cluster->child_clusters[0].image_ids.size()) {
    cluster->child_clusters = {};
  }

  if (options_.image_overlap > 0) {
    for (int i = 0; i < options_.branching; ++i) {
      // Sort the overlapping edges by the number of inlier matches, such
      // that we add overlapping images with many common observations.
      std::sort(overlapping_edges[i].begin(),
                overlapping_edges[i].end(),
                [](const std::pair<std::pair<int, int>, int>& edge1,
                   const std::pair<std::pair<int, int>, int>& edge2) {
                  return edge1.second > edge2.second;
                });

      // Select overlapping edges at random and add image to cluster.
      std::set<int> overlapping_image_ids;
      for (const auto& edge : overlapping_edges[i]) {
        if (labels.at(edge.first.first) == i) {
          overlapping_image_ids.insert(edge.first.second);
        } else {
          overlapping_image_ids.insert(edge.first.first);
        }
        if (overlapping_image_ids.size() >=
            static_cast<size_t>(options_.image_overlap)) {
          break;
        }
      }

      // Recursively append the overlapping images to cluster and its children.
      std::function<void(Cluster*)> InsertOverlappingImageIds =
          [&](Cluster* cluster) {
            cluster->image_ids.insert(cluster->image_ids.end(),
                                      overlapping_image_ids.begin(),
                                      overlapping_image_ids.end());
            for (auto& child_cluster : cluster->child_clusters) {
              InsertOverlappingImageIds(&child_cluster);
            }
          };

      InsertOverlappingImageIds(&cluster->child_clusters[i]);
    }
  }
}

void SceneClustering::PartitionFlatCluster(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights) {
  THROW_CHECK_EQ(edges.size(), weights.size());

  // Partition the cluster using a normalized cut on the scene graph.
  const auto labels =
      ComputeNormalizedMinGraphCut(edges, weights, options_.branching);

  // Assign the images to the clustered child clusters.
  root_cluster_->child_clusters.resize(options_.branching);
  for (const auto image_id : root_cluster_->image_ids) {
    if (labels.count(image_id)) {
      auto& child_cluster =
          root_cluster_->child_clusters.at(labels.at(image_id));
      child_cluster.image_ids.push_back(image_id);
    }
  }

  // Sort child clusters by descending size of images and secondarily by lowest
  // image id.
  std::sort(root_cluster_->child_clusters.begin(),
            root_cluster_->child_clusters.end(),
            [](const Cluster& first, const Cluster& second) {
              return first.image_ids.size() >= second.image_ids.size() &&
                     *std::min_element(first.image_ids.begin(),
                                       first.image_ids.end()) <
                         *std::min_element(second.image_ids.begin(),
                                           second.image_ids.end());
            });

  // For each image find all related images with their weights
  std::unordered_map<int, std::vector<std::pair<int, int>>> related_images;
  for (size_t i = 0; i < edges.size(); ++i) {
    related_images[edges[i].first].emplace_back(edges[i].second, weights[i]);
    related_images[edges[i].second].emplace_back(edges[i].first, weights[i]);
  }

  // Sort related images by decreasing weights
  for (auto& image : related_images) {
    std::sort(image.second.begin(),
              image.second.end(),
              [](const std::pair<int, int>& first,
                 const std::pair<int, int>& second) {
                return first.second > second.second;
              });
  }

  // For each cluster add as many of the needed matching images up to
  // the max image overal allowance
  // We do the process sequentially for each image to ensure that at
  // least we get the best matches firat
  for (int i = 0; i < options_.branching; ++i) {
    auto& orig_image_ids = root_cluster_->child_clusters[i].image_ids;
    std::set<int> cluster_images(
        root_cluster_->child_clusters[i].image_ids.begin(),
        root_cluster_->child_clusters[i].image_ids.end());
    const size_t max_size = cluster_images.size() + options_.image_overlap;
    // check up to all the desired matches
    for (size_t j = 0; j < static_cast<size_t>(options_.num_image_matches) &&
                       cluster_images.size() < max_size;
         ++j) {
      for (const image_t image_id : orig_image_ids) {
        const auto& images = related_images[image_id];
        if (j >= images.size()) {
          continue;
        }
        // image not exists in cluster so we add it in the overlap set
        const int related_id = images[j].first;
        if (cluster_images.count(related_id) == 0) {
          cluster_images.insert(related_id);
        }
        if (cluster_images.size() >= max_size) {
          break;
        }
      }
    }
    orig_image_ids.clear();
    orig_image_ids.insert(
        orig_image_ids.end(), cluster_images.begin(), cluster_images.end());
  }
}

const SceneClustering::Cluster* SceneClustering::GetRootCluster() const {
  return root_cluster_.get();
}

std::vector<const SceneClustering::Cluster*> SceneClustering::GetLeafClusters()
    const {
  THROW_CHECK_NOTNULL(root_cluster_);

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

SceneClustering SceneClustering::Create(const Options& options,
                                        const Database& database) {
  LOG(INFO) << "Reading scene graph...";
  const std::vector<std::pair<image_pair_t, int>> pair_ids_and_num_inliers =
      database.ReadTwoViewGeometryNumInliers();

  std::vector<std::pair<image_t, image_t>> all_image_pairs;
  all_image_pairs.reserve(pair_ids_and_num_inliers.size());
  std::vector<int> all_num_inliers;
  all_num_inliers.reserve(pair_ids_and_num_inliers.size());
  for (const auto& [pair_id, num_inliers] : pair_ids_and_num_inliers) {
    all_image_pairs.push_back(Database::PairIdToImagePair(pair_id));
    all_num_inliers.push_back(num_inliers);
  }

  LOG(INFO) << "Partitioning scene graph...";
  SceneClustering scene_clustering(options);
  scene_clustering.Partition(all_image_pairs, all_num_inliers);
  return scene_clustering;
}

}  // namespace colmap
