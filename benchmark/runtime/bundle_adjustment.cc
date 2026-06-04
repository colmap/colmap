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

#include "colmap/estimators/bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <benchmark/benchmark.h>

using namespace colmap;

static void AddArguments(::benchmark::Benchmark* b) {
  for (const int track_length : {5, 20, 100}) {
    for (const int num_rigs : {1, 5}) {
      for (const int num_cameras_per_rig : {1, 3}) {
        for (const int num_frames_per_rig : {10, 50}) {
          const int num_images =
              num_rigs * num_cameras_per_rig * num_frames_per_rig;
          if (track_length > num_images) continue;
          for (const int num_points3D : {1000, 10000}) {
            b->Args({track_length,
                     num_rigs,
                     num_cameras_per_rig,
                     num_frames_per_rig,
                     num_points3D});
          }
        }
      }
    }
  }
}

class BM_BundleAdjustment : public benchmark::Fixture {
 public:
  void SetUp(::benchmark::State& state) {
    SetPRNGSeed(42);

    SyntheticDatasetOptions dataset_options;
    dataset_options.track_length = state.range(0);
    dataset_options.num_rigs = state.range(1);
    dataset_options.num_cameras_per_rig = state.range(2);
    dataset_options.num_frames_per_rig = state.range(3);
    dataset_options.num_points3D = state.range(4);

    reconstruction_ = std::make_unique<Reconstruction>();
    SynthesizeDataset(dataset_options, reconstruction_.get());

    SyntheticNoiseOptions noise_options;
    noise_options.point2D_stddev = 1.0;
    noise_options.point3D_stddev = 0.05;
    noise_options.rig_from_world_translation_stddev = 0.01;
    noise_options.rig_from_world_rotation_stddev = 1.0;
    SynthesizeNoise(noise_options, reconstruction_.get());

    for (const image_t image_id : reconstruction_->RegImageIds()) {
      config_.AddImage(image_id);
    }
    config_.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

    options_.print_summary = false;
  }

  void TearDown(::benchmark::State& /*state*/) {
    reconstruction_.reset();
    options_ = BundleAdjustmentOptions();
    config_ = BundleAdjustmentConfig();
  }

 protected:
  void ReportSceneCounters(benchmark::State& state) const {
    state.counters["track_length"] = reconstruction_->ComputeMeanTrackLength();
    state.counters["num_images"] = reconstruction_->NumRegImages();
    state.counters["num_rigs"] = reconstruction_->NumRigs();
    state.counters["num_cameras"] = reconstruction_->NumCameras();
    state.counters["num_frames"] = reconstruction_->NumRegFrames();
    state.counters["num_points3d"] = reconstruction_->NumPoints3D();
  }

  std::unique_ptr<Reconstruction> reconstruction_;
  BundleAdjustmentConfig config_;
  BundleAdjustmentOptions options_;
};

// Time column reports wall-clock time (ms) per full BA solve.
BENCHMARK_DEFINE_F(BM_BundleAdjustment, Ceres)(benchmark::State& state) {
  BundleAdjustmentOptions opts = options_;
  opts.backend = BundleAdjustmentBackend::CERES;

  int total_lm_steps = 0;
  double total_ceres_time_s = 0.0;

  for (auto _ : state) {
    state.PauseTiming();
    Reconstruction copy = *reconstruction_;
    state.ResumeTiming();

    auto ba = CreateDefaultBundleAdjuster(opts, config_, copy);
    const auto summary = ba->Solve();

    state.PauseTiming();
    if (summary->termination_type ==
        BundleAdjustmentTerminationType::NO_CONVERGENCE) {
      state.SkipWithError("Bundle adjustment did not converge");
      break;
    }
    const auto* ceres_sum =
        dynamic_cast<const CeresBundleAdjustmentSummary*>(summary.get());
    if (ceres_sum == nullptr) {
      state.SkipWithError("Unexpected summary type for Ceres backend");
      break;
    }
    total_lm_steps += ceres_sum->ceres_summary.num_successful_steps +
                      ceres_sum->ceres_summary.num_unsuccessful_steps;
    total_ceres_time_s += ceres_sum->ceres_summary.total_time_in_seconds;
    state.ResumeTiming();
  }

  state.PauseTiming();
  ReportSceneCounters(state);
  const int64_t num_iters = state.iterations();
  if (num_iters > 0) {
    state.counters["avg_lm_steps"] =
        static_cast<double>(total_lm_steps) / num_iters;
  }
  if (total_lm_steps > 0) {
    state.counters["avg_ms_per_lm_step"] =
        total_ceres_time_s * 1000.0 / total_lm_steps;
  }
  state.ResumeTiming();
}

#ifdef CASPAR_ENABLED
// Time column reports wall-clock time (ms) per full BA solve.
BENCHMARK_DEFINE_F(BM_BundleAdjustment, Caspar)(benchmark::State& state) {
  BundleAdjustmentOptions opts = options_;
  opts.backend = BundleAdjustmentBackend::CASPAR;

  for (auto _ : state) {
    state.PauseTiming();
    Reconstruction copy = *reconstruction_;
    state.ResumeTiming();

    auto ba = CreateDefaultBundleAdjuster(opts, config_, copy);
    const auto summary = ba->Solve();

    state.PauseTiming();
    if (summary->termination_type ==
        BundleAdjustmentTerminationType::NO_CONVERGENCE) {
      state.SkipWithError("Bundle adjustment did not converge");
      break;
    }
    state.ResumeTiming();
  }

  state.PauseTiming();
  ReportSceneCounters(state);
  state.ResumeTiming();
}
#endif

BENCHMARK_REGISTER_F(BM_BundleAdjustment, Ceres)
    ->Apply(AddArguments)
    ->Unit(benchmark::kMillisecond);

#ifdef CASPAR_ENABLED
BENCHMARK_REGISTER_F(BM_BundleAdjustment, Caspar)
    ->Apply(AddArguments)
    ->Unit(benchmark::kMillisecond);
#endif

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
