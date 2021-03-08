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

#ifndef COLMAP_SRC_BASE_SCENE_CLUSTERING_H_
#define COLMAP_SRC_BASE_SCENE_CLUSTERING_H_

#include <list>
#include <vector>

#include "base/database.h"
#include "util/types.h"

namespace colmap {

// Scene clustering approach using normalized cuts on the scene graph. The scene
// is hierarchically partitioned into overlapping clusters until a maximum
// number of images is in a leaf node.
class SceneClustering {
 public:
  struct Options {
    // Flag for hierarchical vs flat clustering
    bool is_hierarchical = true;

    // The branching factor of the hierarchical clustering.
    int branching = 2;

    // The number of overlapping images between child clusters.
    int image_overlap = 50;

    // The max related images matches to look for in a flat cluster
    int num_image_matches = 20;

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

  static SceneClustering Create(const Options& options,
                                const Database& database);

 private:
  void PartitionHierarchicalCluster(
      const std::vector<std::pair<int, int>>& edges,
      const std::vector<int>& weights, Cluster* cluster);

  void PartitionFlatCluster(const std::vector<std::pair<int, int>>& edges,
                            const std::vector<int>& weights);

  const Options options_;
  std::unique_ptr<Cluster> root_cluster_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_SCENE_CLUSTERING_H_
