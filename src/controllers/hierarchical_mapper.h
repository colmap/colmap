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

#ifndef COLMAP_SRC_CONTROLLERS_HIERARCHICAL_MAPPER_H_
#define COLMAP_SRC_CONTROLLERS_HIERARCHICAL_MAPPER_H_

#include "base/reconstruction_manager.h"
#include "base/scene_clustering.h"
#include "controllers/incremental_mapper.h"
#include "util/threading.h"

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

    bool Check() const;
  };

  HierarchicalMapperController(
      const Options& options,
      const SceneClustering::Options& clustering_options,
      const IncrementalMapperController::Options& mapper_options,
      ReconstructionManager* reconstruction_manager);

 private:
  void Run() override;

  const Options options_;
  const SceneClustering::Options clustering_options_;
  const IncrementalMapperController::Options mapper_options_;
  ReconstructionManager* reconstruction_manager_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_CONTROLLERS_HIERARCHICAL_MAPPER_H_
