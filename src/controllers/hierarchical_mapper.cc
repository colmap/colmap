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

#include "controllers/hierarchical_mapper.h"

#include "base/scene_clustering.h"
#include "util/misc.h"

namespace colmap {
namespace {

void MergeClusters(
    const SceneClustering::Cluster& cluster,
    std::unordered_map<const SceneClustering::Cluster*, ReconstructionManager>*
        reconstruction_managers) {
  // Extract all reconstructions from all child clusters.
  std::vector<Reconstruction*> reconstructions;
  for (const auto& child_cluster : cluster.child_clusters) {
    if (!child_cluster.child_clusters.empty()) {
      MergeClusters(child_cluster, reconstruction_managers);
    }

    auto& reconstruction_manager = reconstruction_managers->at(&child_cluster);
    for (size_t i = 0; i < reconstruction_manager.Size(); ++i) {
      reconstructions.push_back(&reconstruction_manager.Get(i));
    }
  }

  // Try to merge all child cluster reconstruction.
  while (reconstructions.size() > 1) {
    bool merge_success = false;
    for (size_t i = 0; i < reconstructions.size(); ++i) {
      for (size_t j = 0; j < i; ++j) {
        const int kMinCommonImages = 3;
        if (reconstructions[i]->Merge(*reconstructions[j], kMinCommonImages)) {
          reconstructions.erase(reconstructions.begin() + j);
          merge_success = true;
          break;
        }
      }

      if (merge_success) {
        break;
      }
    }

    if (!merge_success) {
      break;
    }
  }

  // Insert a new reconstruction manager for merged cluster.
  auto& reconstruction_manager = (*reconstruction_managers)[&cluster];
  for (const auto& reconstruction : reconstructions) {
    reconstruction_manager.Add();
    reconstruction_manager.Get(reconstruction_manager.Size() - 1) =
        *reconstruction;
  }

  // Delete all merged child cluster reconstruction managers.
  for (const auto& child_cluster : cluster.child_clusters) {
    reconstruction_managers->erase(&child_cluster);
  }
}

}  // namespace

bool HierarchicalMapperController::Options::Check() const {
  CHECK_OPTION_GT(init_num_trials, -1);
  CHECK_OPTION_GE(num_workers, -1);
  return true;
}

HierarchicalMapperController::HierarchicalMapperController(
    const Options& options, const SceneClustering::Options& clustering_options,
    const IncrementalMapperController::Options& mapper_options,
    ReconstructionManager* reconstruction_manager)
    : options_(options),
      clustering_options_(clustering_options),
      mapper_options_(mapper_options),
      reconstruction_manager_(reconstruction_manager) {
  CHECK(options_.Check());
  CHECK(clustering_options_.Check());
  CHECK(mapper_options_.Check());
  CHECK_EQ(clustering_options_.branching, 2);
}

void HierarchicalMapperController::Run() {
  PrintHeading1("Partitioning the scene");

  //////////////////////////////////////////////////////////////////////////////
  // Cluster scene
  //////////////////////////////////////////////////////////////////////////////

  SceneClustering scene_clustering(clustering_options_);

  std::unordered_map<image_t, std::string> image_id_to_name;

  {
    Database database(options_.database_path);

    std::cout << "Reading images..." << std::endl;
    const auto images = database.ReadAllImages();
    for (const auto& image : images) {
      image_id_to_name.emplace(image.ImageId(), image.Name());
    }

    std::cout << "Reading scene graph..." << std::endl;
    std::vector<std::pair<image_t, image_t>> image_pairs;
    std::vector<int> num_inliers;
    database.ReadInlierMatchesGraph(&image_pairs, &num_inliers);

    std::cout << "Partitioning scene graph..." << std::endl;
    scene_clustering.Partition(image_pairs, num_inliers);
  }

  auto leaf_clusters = scene_clustering.GetLeafClusters();

  size_t total_num_images = 0;
  for (size_t i = 0; i < leaf_clusters.size(); ++i) {
    total_num_images += leaf_clusters[i]->image_ids.size();
    std::cout << StringPrintf("  Cluster %d with %d images", i + 1,
                              leaf_clusters[i]->image_ids.size())
              << std::endl;
  }

  std::cout << StringPrintf("Clusters have %d images", total_num_images)
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Reconstruct clusters
  //////////////////////////////////////////////////////////////////////////////

  PrintHeading1("Reconstructing clusters");

  // Determine the number of workers and threads per worker.
  const int kMaxNumThreads = -1;
  const int num_eff_threads = GetEffectiveNumThreads(kMaxNumThreads);
  const int kDefaultNumWorkers = 8;
  const int num_eff_workers =
      options_.num_workers < 1
          ? std::min(static_cast<int>(leaf_clusters.size()),
                     std::min(kDefaultNumWorkers, num_eff_threads))
          : options_.num_workers;
  const int num_threads_per_worker =
      std::max(1, num_eff_threads / num_eff_workers);

  // Function to reconstruct one cluster using incremental mapping.
  auto ReconstructCluster = [&, this](
                                const SceneClustering::Cluster& cluster,
                                ReconstructionManager* reconstruction_manager) {
    if (cluster.image_ids.empty()) {
      return;
    }

    IncrementalMapperController::Options custom_options = mapper_options_;
    custom_options.max_model_overlap = 3;
    custom_options.init_num_trials = options_.init_num_trials;
    custom_options.num_threads = num_threads_per_worker;

    for (const auto image_id : cluster.image_ids) {
      custom_options.image_names.insert(image_id_to_name.at(image_id));
    }

    IncrementalMapperController mapper(&custom_options, options_.image_path,
                                       options_.database_path,
                                       reconstruction_manager);
    mapper.Start();
    mapper.Wait();
  };

  ThreadPool thread_pool(num_eff_workers);

  // Start reconstructing the bigger clusters first for resource usage.
  std::sort(leaf_clusters.begin(), leaf_clusters.end(),
            [](const SceneClustering::Cluster* cluster1,
               const SceneClustering::Cluster* cluster2) {
              return cluster1->image_ids.size() > cluster2->image_ids.size();
            });

  // Start the reconstruction workers.

  std::unordered_map<const SceneClustering::Cluster*, ReconstructionManager>
      reconstruction_managers;
  reconstruction_managers.reserve(leaf_clusters.size());
  for (const auto& cluster : leaf_clusters) {
    thread_pool.AddTask(ReconstructCluster, *cluster,
                        &reconstruction_managers[cluster]);
  }
  thread_pool.Wait();

  //////////////////////////////////////////////////////////////////////////////
  // Merge clusters
  //////////////////////////////////////////////////////////////////////////////

  PrintHeading1("Merging clusters");

  MergeClusters(*scene_clustering.GetRootCluster(), &reconstruction_managers);

  CHECK_EQ(reconstruction_managers.size(), 1);
  *reconstruction_manager_ = std::move(reconstruction_managers.begin()->second);

  std::cout << std::endl;
  GetTimer().PrintMinutes();
}

}  // namespace colmap
