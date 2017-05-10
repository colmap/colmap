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

#ifndef COLMAP_SRC_BASE_SCENE_CLUSTERING_H_
#define COLMAP_SRC_BASE_SCENE_CLUSTERING_H_

#include <list>
#include <vector>

#include "util/types.h"

namespace colmap {

// Scene clustering approach using normalized cuts on the scene graph. The scene
// is hierarchically partitioned into overlapping clusters until a maximum
// number of images is in a leaf node.
class SceneClustering {
 public:
  struct Options {
    // The branching factor of the hierarchical clustering.
    int branching = 2;

    // The number of overlapping images between child clusters.
    int image_overlap = 50;

    // The maximum number of images in a leaf node cluster, otherwise the
    // cluster is further partitioned using the given branching factor. Note
    // that a cluster leaf node will have at most `leaf_max_num_images +
    // overlap` images to satisfy the overlap constraint.
    int leaf_max_num_images = 500;

    bool Check() const;
  };

  struct Cluster {
    std::vector<image_t> image_ids;
    std::vector<Cluster> child_clusters;
  };

  SceneClustering(const Options& options);

  void Partition(const std::vector<std::pair<image_t, image_t>>& image_pairs,
                 const std::vector<int>& num_inliers);

  const Cluster* GetRootCluster() const;
  std::vector<const Cluster*> GetLeafClusters() const;

 private:
  void PartitionCluster(const std::vector<std::pair<int, int>>& edges,
                        const std::vector<int>& weights, Cluster* cluster);

  const Options options_;
  std::unique_ptr<Cluster> root_cluster_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_SCENE_CLUSTERING_H_
