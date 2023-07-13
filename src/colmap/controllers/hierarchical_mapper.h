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

#pragma once

#include "colmap/base/reconstruction_manager.h"
#include "colmap/base/scene_clustering.h"
#include "colmap/controllers/incremental_mapper.h"
#include "colmap/util/threading.h"

#include <memory>

namespace colmap {

// Hierarchical mapping first hierarchically partitions the scene into multiple
// overlapping clusters, then reconstructs them separately using incremental
// mapping, and finally merges them all into a globally consistent
// reconstruction. This is especially useful for larger-scale scenes, since
// incremental mapping becomes slow with an increasing number of images.
class HierarchicalMapperController : public Thread {
 public:
  struct Options {
    // The path to the image folder which are used as input.
    std::string image_path;

    // The path to the database file which is used as input.
    std::string database_path;

    // The maximum number of trials to initialize a cluster.
    int init_num_trials = 10;

    // The number of workers used to reconstruct clusters in parallel.
    int num_workers = -1;

    // Options for clustering the scene graph.
    SceneClustering::Options clustering_options;

    // Options used to reconstruction each cluster individually.
    IncrementalMapperOptions incremental_options;

    bool Check() const;
  };

  HierarchicalMapperController(
      const Options& options,
      std::shared_ptr<ReconstructionManager> reconstruction_manager);

 private:
  void Run() override;

  const Options options_;
  const SceneClustering::Options clustering_options_;
  const std::shared_ptr<const IncrementalMapperOptions> mapper_options_;
  std::shared_ptr<ReconstructionManager> reconstruction_manager_;
};

}  // namespace colmap
