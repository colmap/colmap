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

#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <benchmark/benchmark.h>

using namespace colmap;

void GenerateArguments(benchmark::internal::Benchmark* b) {
  for (const int num_rigs : {1, 5}) {
    for (const int num_cameras_per_rig : {1, 3}) {
      for (const int num_frames_per_rig : {10, 50}) {
        for (const int num_points3D : {1000, 10000}) {
          b->Args({num_rigs,
                   num_cameras_per_rig,
                   num_frames_per_rig,
                   num_points3D});
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
    dataset_options.num_rigs = state.range(0);
    dataset_options.num_cameras_per_rig = state.range(1);
    dataset_options.num_frames_per_rig = state.range(2);
    dataset_options.num_points3D = state.range(3);

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

    // Set up BA options.
    options_.print_summary = false;
    options_.solver_options.max_num_iterations = 50;
  }

  void TearDown(::benchmark::State& state) {
    reconstruction_.reset();
    options_ = BundleAdjustmentOptions();
    config_ = BundleAdjustmentConfig();
  }

 protected:
  std::unique_ptr<Reconstruction> reconstruction_;
  BundleAdjustmentConfig config_;
  BundleAdjustmentOptions options_;
};

BENCHMARK_DEFINE_F(BM_BundleAdjustment, Solve)(benchmark::State& state) {
  int num_iterations = 0;
  for (auto _ : state) {
    state.PauseTiming();
    // Copy the reconstruction for each iteration since BA modifies it.
    Reconstruction reconstruction_copy = *reconstruction_;
    state.ResumeTiming();

    auto bundle_adjuster =
        CreateDefaultBundleAdjuster(options_, config_, reconstruction_copy);
    const ceres::Solver::Summary summary = bundle_adjuster->Solve();

    // Stop timing and check if BA converged.
    state.PauseTiming();
    num_iterations += summary.num_successful_steps;
    if (summary.termination_type == ceres::NO_CONVERGENCE) {
      state.SkipWithError("Bundle adjustment did not converge");
    }
    state.ResumeTiming();
  }

  state.PauseTiming();
  // Report custom counters.
  state.counters["imgs"] = reconstruction_->NumRegImages();
  state.counters["rigs"] = reconstruction_->NumRigs();
  state.counters["cams"] = reconstruction_->NumCameras();
  state.counters["frms"] = reconstruction_->NumRegFrames();
  state.counters["pnts"] = reconstruction_->NumPoints3D();
  state.counters["itrs"] = num_iterations;
  state.ResumeTiming();
}

BENCHMARK_REGISTER_F(BM_BundleAdjustment, Solve)
    ->Apply(GenerateArguments)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();
