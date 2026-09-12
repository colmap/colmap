// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/scene/scene_clustering.h"
#include "colmap/util/base_controller.h"

#include <filesystem>
#include <memory>

namespace colmap {

struct HierarchicalPipelineOptions {
  // The image path at which to find the images to extract point colors.
  // If not specified, all point colors will be black.
  std::filesystem::path image_path;

  // The maximum number of trials to initialize a cluster.
  int init_num_trials = 10;

  // The total number of threads for the hierarchical pipeline. This budget
  // is divided across workers to avoid thread oversubscription.
  int num_threads = -1;

  // The number of workers used to reconstruct clusters in parallel.
  int num_workers = -1;

  // Options for clustering the scene graph.
  SceneClustering::Options clustering_options;

  // Options used to reconstruction each cluster individually.
  IncrementalPipelineOptions incremental_options;

  bool Check() const;
};

// Hierarchical mapping first hierarchically partitions the scene into multiple
// overlapping clusters, then reconstructs them separately using incremental
// mapping, and finally merges them all into a globally consistent
// reconstruction. This is especially useful for larger-scale scenes, since
// incremental mapping becomes slow with an increasing number of images.
class HierarchicalPipeline : public BaseController {
 public:
  HierarchicalPipeline(
      const HierarchicalPipelineOptions& options,
      std::shared_ptr<Database> database,
      std::shared_ptr<ReconstructionManager> reconstruction_manager);

  void Run() override;

 private:
  const HierarchicalPipelineOptions options_;
  std::shared_ptr<DatabaseCache> database_cache_;
  std::shared_ptr<ReconstructionManager> reconstruction_manager_;
};

}  // namespace colmap
