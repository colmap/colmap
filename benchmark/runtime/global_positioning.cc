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

#include "glomap/estimators/global_positioning.h"

#include "colmap/math/random.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include "glomap/scene/pose_graph.h"

#include <algorithm>
#include <atomic>

#include <benchmark/benchmark.h>
#include <glog/logging.h>

using namespace colmap;
using namespace glomap;

static void BM_GlobalPositioning(benchmark::State& state) {
  FLAGS_minloglevel = 2;  // Suppress INFO and WARNING logs.
  SetPRNGSeed(42);

  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = state.range(0);
  dataset_options.num_cameras_per_rig = state.range(1);
  dataset_options.num_frames_per_rig = state.range(2);
  dataset_options.num_points3D = state.range(3);

  // Compute sparsity from target number of neighbors per image.
  // sparsity ≈ 1 - num_neighbors / (num_images - 1)
  const int num_neighbors = state.range(4);
  const int num_images = dataset_options.num_rigs *
                         dataset_options.num_cameras_per_rig *
                         dataset_options.num_frames_per_rig;
  dataset_options.match_sparsity = std::max(
      0.0, 1.0 - static_cast<double>(num_neighbors) / (num_images - 1));
  dataset_options.match_config = SyntheticDatasetOptions::MatchConfig::SPARSE;
  dataset_options.two_view_geometry_has_relative_pose = true;

  static std::atomic<int> counter{0};
  auto temp_dir = std::filesystem::temp_directory_path() /
                  ("colmap_benchmark_gp_" + std::to_string(counter++));
  std::filesystem::remove_all(temp_dir);  // Clean up any leftover from previous run.
  std::filesystem::create_directories(temp_dir);

  auto database = Database::Open(temp_dir / "database.db");
  Reconstruction gt_reconstruction;
  SynthesizeDataset(dataset_options, &gt_reconstruction, database.get());

  DatabaseCache database_cache;
  DatabaseCache::Options cache_options;
  database_cache.Load(*database, cache_options);
  database.reset();  // Close database connection.

  PoseGraph pose_graph;
  pose_graph.Load(*database_cache.CorrespondenceGraph());

  Reconstruction reconstruction = gt_reconstruction;
  for (const auto& [frame_id, _] : reconstruction.Frames()) {
    Frame& frame = reconstruction.Frame(frame_id);
    frame.SetRigFromWorld(Rigid3d(frame.RigFromWorld().rotation(),
                                  Eigen::Vector3d::Zero()));
  }

  GlobalPositionerOptions base_options;
  base_options.use_gpu = false;
  base_options.random_seed = 42;
  base_options.solver_options.max_num_iterations = 50;
  base_options.solver_options.minimizer_progress_to_stdout = false;

  for (auto _ : state) {
    state.PauseTiming();
    Reconstruction reconstruction_copy = reconstruction;
    GlobalPositionerOptions options = base_options;
    state.ResumeTiming();

    GlobalPositioner positioner(options);
    positioner.Solve(pose_graph, reconstruction_copy);
  }

  state.counters["imgs"] = reconstruction.NumRegImages();
  state.counters["rigs"] = reconstruction.NumRigs();
  state.counters["cams"] = reconstruction.NumCameras();
  state.counters["frms"] = reconstruction.NumRegFrames();
  state.counters["pnts"] = reconstruction.NumPoints3D();
  state.counters["nbrs"] = num_neighbors;

  std::filesystem::remove_all(temp_dir);
}

static void GenerateArguments(benchmark::Benchmark* b) {
  // Args: {num_rigs, num_cameras_per_rig, num_frames_per_rig, num_points3D,
  //        num_neighbors}
  for (const int num_rigs : {1, 5}) {
    for (const int num_cameras_per_rig : {1, 3}) {
      for (const int num_frames_per_rig : {10, 50}) {
        for (const int num_points3D : {1000, 10000}) {
          for (const int num_neighbors : {10, 20}) {
            b->Args({num_rigs,
                     num_cameras_per_rig,
                     num_frames_per_rig,
                     num_points3D,
                     num_neighbors});
          }
        }
      }
    }
  }
}

BENCHMARK(BM_GlobalPositioning)
    ->Apply(GenerateArguments)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();
