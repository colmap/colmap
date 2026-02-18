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

#include "colmap/estimators/global_positioning.h"

#include "colmap/math/random.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <algorithm>
#include <array>

#include <benchmark/benchmark.h>
#include <glog/logging.h>

using namespace colmap;

struct CachedData {
  std::array<int64_t, 5> dataset_args;
  Reconstruction reconstruction;
  colmap::PoseGraph pose_graph;
};

static void BM_GlobalPositioning(benchmark::State& state) {
  FLAGS_minloglevel = 2;  // Suppress INFO and WARNING logs.

  const std::array<int64_t, 5> dataset_args = {state.range(0),
                                               state.range(1),
                                               state.range(2),
                                               state.range(3),
                                               state.range(4)};
  const bool use_parameter_block_ordering = state.range(5);

  // Cache dataset to avoid resynthesizing for different ordering options.
  static std::unique_ptr<CachedData> cached;
  if (!cached || cached->dataset_args != dataset_args) {
    SetPRNGSeed(42);

    SyntheticDatasetOptions dataset_options;
    dataset_options.num_rigs = dataset_args[0];
    dataset_options.num_cameras_per_rig = dataset_args[1];
    dataset_options.num_frames_per_rig = dataset_args[2];
    dataset_options.num_points3D = dataset_args[3];

    // Compute sparsity from target number of neighbors per image.
    // sparsity ≈ 1 - num_neighbors / (num_images - 1)
    const int num_neighbors = dataset_args[4];
    const int num_images = dataset_options.num_rigs *
                           dataset_options.num_cameras_per_rig *
                           dataset_options.num_frames_per_rig;
    dataset_options.match_sparsity = std::max(
        0.0, 1.0 - static_cast<double>(num_neighbors) / (num_images - 1));
    dataset_options.match_config = SyntheticDatasetOptions::MatchConfig::SPARSE;
    dataset_options.two_view_geometry_has_relative_pose = true;

    cached = std::make_unique<CachedData>();
    cached->dataset_args = dataset_args;

    auto database = Database::Open(kInMemorySqliteDatabasePath);
    Reconstruction gt_reconstruction;
    SynthesizeDataset(dataset_options, &gt_reconstruction, database.get());

    DatabaseCache database_cache;
    DatabaseCache::Options cache_options;
    database_cache.Load(*database, cache_options);
    database.reset();  // Close database connection.

    cached->pose_graph.Load(*database_cache.CorrespondenceGraph());

    cached->reconstruction = gt_reconstruction;
    for (const auto& [frame_id, _] : cached->reconstruction.Frames()) {
      Frame& frame = cached->reconstruction.Frame(frame_id);
      frame.SetRigFromWorld(
          Rigid3d(frame.RigFromWorld().rotation(), Eigen::Vector3d::Zero()));
    }
  }

  const Reconstruction& reconstruction = cached->reconstruction;
  const colmap::PoseGraph& pose_graph = cached->pose_graph;
  const int num_neighbors = dataset_args[4];

  colmap::GlobalPositionerOptions base_options;
  base_options.use_gpu = false;
  base_options.random_seed = 42;
  base_options.solver_options.max_num_iterations = 50;
  base_options.solver_options.minimizer_progress_to_stdout = false;

  for (auto _ : state) {
    state.PauseTiming();
    Reconstruction reconstruction_copy = reconstruction;
    colmap::GlobalPositionerOptions options = base_options;
    options.use_parameter_block_ordering = use_parameter_block_ordering;
    state.ResumeTiming();

    colmap::GlobalPositioner positioner(options);
    positioner.Solve(pose_graph, reconstruction_copy);
  }
  state.counters["ord"] = use_parameter_block_ordering;
  state.counters["imgs"] = reconstruction.NumRegImages();
  state.counters["rigs"] = reconstruction.NumRigs();
  state.counters["cams"] = reconstruction.NumCameras();
  state.counters["frms"] = reconstruction.NumRegFrames();
  state.counters["pnts"] = reconstruction.NumPoints3D();
  state.counters["nbrs"] = num_neighbors;
}

static void GenerateArguments(benchmark::Benchmark* b) {
  // Args: {num_rigs, num_cameras_per_rig, num_frames_per_rig, num_points3D,
  //        num_neighbors, use_parameter_block_ordering}
  for (const int num_rigs : {1, 5}) {
    for (const int num_cameras_per_rig : {1, 3}) {
      for (const int num_frames_per_rig : {10, 50}) {
        for (const int num_points3D : {1000, 10000}) {
          for (const int num_neighbors : {10, 20}) {
            for (const bool use_parameter_block_ordering : {true, false}) {
              b->Args({num_rigs,
                       num_cameras_per_rig,
                       num_frames_per_rig,
                       num_points3D,
                       num_neighbors,
                       use_parameter_block_ordering});
            }
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
