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

// End-to-end incremental mapping benchmark used to measure the runtime and
// peak-memory impact of the hash map backend (see src/colmap/util/containers.h
// and the COLMAP_HASH_MAP_BACKEND CMake option). Build one binary per backend
// (via benchmark/runtime/run_hash_map_experiment.sh) and compare.
//
// Two families of benchmarks:
//   * BM_IncrementalMapping/Synthetic: parametrized synthetic scenes generated
//     with colmap::SynthesizeDataset into an in-memory database.
//   * BM_IncrementalMapping/RealDatabase: a full run on a real database. The
//     path defaults to ~/data/south-building/database.db and can be overridden
//     with the COLMAP_BENCHMARK_DATABASE_PATH environment variable. The
//     benchmark is skipped (not failed) if the database does not exist.

#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/math/random.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/logging.h"
#include "colmap/util/memory.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

#include <benchmark/benchmark.h>

using namespace colmap;

namespace {

constexpr unsigned kSeed = 42;

// Deterministic mapper options so all backends perform identical work; only the
// container implementation varies between builds.
std::shared_ptr<IncrementalPipelineOptions> MakeOptions() {
  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->random_seed = kSeed;
  // Single-threaded to isolate container performance and keep timings stable.
  options->num_threads = 1;
  options->extract_colors = false;
  return options;
}

// Runs one full reconstruction from the given database, returns the largest
// reconstruction (or nullptr if none). Reports scene-size and peak-memory
// counters on the benchmark state.
void RunPipelineAndReport(benchmark::State& state,
                          const std::shared_ptr<Database>& database) {
  std::shared_ptr<const Reconstruction> largest;
  for (auto _ : state) {
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    IncrementalPipeline mapper(
        MakeOptions(), database, reconstruction_manager);
    mapper.Run();

    state.PauseTiming();
    size_t best_size = 0;
    largest = nullptr;
    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
      const auto& reconstruction = reconstruction_manager->Get(i);
      if (reconstruction->NumRegImages() >= best_size) {
        best_size = reconstruction->NumRegImages();
        largest = reconstruction;
      }
    }
    state.ResumeTiming();
  }

  state.counters["peak_rss_mb"] =
      static_cast<double>(GetPeakRSSBytes()) / (1024.0 * 1024.0);
  if (largest != nullptr) {
    state.counters["num_reg_images"] = largest->NumRegImages();
    state.counters["num_points3d"] = largest->NumPoints3D();
    state.counters["mean_track_length"] = largest->ComputeMeanTrackLength();
  } else {
    state.SkipWithError("No reconstruction was produced");
  }
}

// -----------------------------------------------------------------------------
// Synthetic scenes.
// -----------------------------------------------------------------------------

void AddSyntheticArguments(::benchmark::internal::Benchmark* b) {
  b->ArgNames({"num_frames_per_rig", "num_points3D"});
  // Scale the number of images (num_frames_per_rig, since num_rigs =
  // num_cameras_per_rig = 1). A SPARSE view graph (see BM body) keeps the
  // per-image matching/observation work bounded so the image count can scale
  // without the O(n^2) exhaustive-matching blow-up, isolating the effect of
  // image-count scaling on the container-heavy registration paths.
  // Keep num_points3D small: the synthetic generator makes every point visible
  // in every image (dense tracks that neither track_length nor match_config
  // reduce), so total observations = num_points3D * num_images. A small point
  // count keeps bundle adjustment bounded while the image count scales.
  for (const int num_frames_per_rig : {50, 100, 200, 400}) {
    b->Args({num_frames_per_rig, /*num_points3D=*/1500});
  }
}

void BM_IncrementalMapping_Synthetic(benchmark::State& state) {
  SetPRNGSeed(kSeed);

  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 1;
  dataset_options.num_frames_per_rig = static_cast<int>(state.range(0));
  dataset_options.num_points3D = static_cast<int>(state.range(1));
  // Sparse but connected view graph so per-image work stays bounded as the
  // image count grows (0 = fully connected/exhaustive, 1 = empty).
  dataset_options.match_config = SyntheticDatasetOptions::MatchConfig::SPARSE;
  dataset_options.match_sparsity = 0.9;

  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction gt_reconstruction;
  SynthesizeDataset(dataset_options, &gt_reconstruction, database.get());

  // Inject realistic observation noise so the mapper performs representative
  // bundle-adjustment, triangulation and outlier-filtering work. point2D_stddev
  // perturbs the 2D keypoints written to the database (i.e. what the mapper
  // actually reads); the point3D/pose noise perturbs the ground-truth
  // reconstruction only and is included to match the bundle_adjustment
  // benchmark.
  SyntheticNoiseOptions noise_options;
  noise_options.point2D_stddev = 1.0;
  noise_options.point3D_stddev = 0.05;
  noise_options.rig_from_world_translation_stddev = 0.01;
  noise_options.rig_from_world_rotation_stddev = 1.0;
  SynthesizeNoise(noise_options, &gt_reconstruction, database.get());

  RunPipelineAndReport(state, database);
}

BENCHMARK(BM_IncrementalMapping_Synthetic)
    ->Apply(AddSyntheticArguments)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(1)
    ->UseRealTime();

// -----------------------------------------------------------------------------
// Real database (defaults to the south-building dataset).
// -----------------------------------------------------------------------------

std::filesystem::path RealDatabasePath() {
  if (const char* env = std::getenv("COLMAP_BENCHMARK_DATABASE_PATH");
      env != nullptr && env[0] != '\0') {
    return std::filesystem::path(env);
  }
  const char* home = std::getenv("HOME");
  const std::filesystem::path home_path(home != nullptr ? home : "");
  return home_path / "data" / "south-building" / "database.db";
}

void BM_IncrementalMapping_RealDatabase(benchmark::State& state) {
  SetPRNGSeed(kSeed);

  const std::filesystem::path database_path = RealDatabasePath();
  if (!std::filesystem::exists(database_path)) {
    state.SkipWithError(
        ("Database not found: " + database_path.string() +
         " (set COLMAP_BENCHMARK_DATABASE_PATH to override)")
            .c_str());
    return;
  }

  // Copy to a temporary writable location: the reconstruction opens the
  // database read-write, and this avoids mutating the user's original database
  // (and sidesteps read-only/WAL state on the source file).
  const std::filesystem::path tmp_path =
      std::filesystem::temp_directory_path() / "colmap_benchmark_database.db";
  std::filesystem::remove(tmp_path);
  std::filesystem::copy_file(
      database_path, tmp_path, std::filesystem::copy_options::overwrite_existing);

  auto database = Database::Open(tmp_path);
  RunPipelineAndReport(state, database);

  database.reset();
  std::filesystem::remove(tmp_path);
}

BENCHMARK(BM_IncrementalMapping_RealDatabase)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(3)
    ->UseRealTime();

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
