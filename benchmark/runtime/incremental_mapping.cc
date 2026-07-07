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

// End-to-end incremental mapping benchmark on synthetic data. Measures the
// wall-clock runtime of a full reconstruction (registration, triangulation,
// bundle adjustment) and is useful for evaluating the impact of changes to the
// mapper or bundle adjustment. Parametrized over camera model and scene size;
// the camera model determines which cost-function path bundle adjustment
// exercises. All runs are deterministic (fixed seed, single thread) so the
// produced reconstruction is stable and only the runtime varies across
// configurations.

#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/math/random.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"
#include "colmap/util/logging.h"

#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

using namespace colmap;

namespace {

constexpr unsigned kSeed = 42;

// Deterministic mapper options so all configurations perform identical work;
// only the bundle-adjustment cost-function implementation varies between
// builds.
std::shared_ptr<IncrementalPipelineOptions> MakeOptions() {
  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->random_seed = kSeed;
  // Single-threaded to keep timings stable and isolate per-residual cost.
  options->num_threads = 1;
  options->extract_colors = false;
  return options;
}

void RunPipelineAndReport(benchmark::State& state,
                          const std::shared_ptr<Database>& database) {
  std::shared_ptr<const Reconstruction> largest;
  for (auto _ : state) {
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    IncrementalPipeline mapper(MakeOptions(), database, reconstruction_manager);
    mapper.Run();

    state.PauseTiming();
    largest = nullptr;
    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
      const auto& reconstruction = reconstruction_manager->Get(i);
      if (largest == nullptr ||
          reconstruction->NumRegImages() > largest->NumRegImages()) {
        largest = reconstruction;
      }
    }
    state.ResumeTiming();
  }

  if (largest != nullptr) {
    state.counters["num_reg_images"] = largest->NumRegImages();
    state.counters["num_points3d"] = largest->NumPoints3D();
    state.counters["mean_track_length"] = largest->ComputeMeanTrackLength();
  } else {
    state.SkipWithError("No reconstruction was produced");
  }
}

std::vector<double> DefaultCameraParams(const CameraModelId model_id) {
  // Nominal intrinsics for a 1024x768 image with moderate distortion.
  switch (model_id) {
    case PinholeCameraModel::model_id:
      return {1280, 1280, 512, 384};
    case OpenCVCameraModel::model_id:
      return {1280, 1280, 512, 384, 0.05, 0.01, 0.001, 0.001};
    default:
      LOG(FATAL) << "Unsupported camera model";
      return {};
  }
}

void BM_IncrementalMapping(benchmark::State& state,
                           const CameraModelId camera_model_id) {
  SetPRNGSeed(kSeed);

  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 1;
  dataset_options.num_frames_per_rig = static_cast<int>(state.range(0));
  dataset_options.num_points3D = static_cast<int>(state.range(1));
  dataset_options.camera_model_id = camera_model_id;
  dataset_options.camera_params = DefaultCameraParams(camera_model_id);
  // Sparse but connected view graph so per-image work stays bounded as the
  // image count grows.
  dataset_options.match_config = SyntheticDatasetOptions::MatchConfig::SPARSE;
  dataset_options.match_sparsity = 0.9;

  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction gt_reconstruction;
  SynthesizeDataset(dataset_options, &gt_reconstruction, database.get());

  // Inject realistic observation noise so bundle adjustment performs
  // representative work.
  SyntheticNoiseOptions noise_options;
  noise_options.point2D_stddev = 1.0;
  noise_options.point3D_stddev = 0.05;
  noise_options.rig_from_world_translation_stddev = 0.01;
  noise_options.rig_from_world_rotation_stddev = 1.0;
  SynthesizeNoise(noise_options, &gt_reconstruction, database.get());

  RunPipelineAndReport(state, database);
}

void AddArguments(::benchmark::Benchmark* b) {
  b->ArgNames({"num_frames", "num_points3D"});
  for (const int num_frames : {25, 50, 100}) {
    b->Args({num_frames, /*num_points3D=*/1000});
  }
  b->Unit(benchmark::kMillisecond)
      ->Iterations(1)
      ->Repetitions(5)
      ->ReportAggregatesOnly(true)
      ->UseRealTime();
}

}  // namespace

BENCHMARK_CAPTURE(BM_IncrementalMapping, PINHOLE, PinholeCameraModel::model_id)
    ->Apply(AddArguments);
BENCHMARK_CAPTURE(BM_IncrementalMapping, OPENCV, OpenCVCameraModel::model_id)
    ->Apply(AddArguments);

BENCHMARK_MAIN();
