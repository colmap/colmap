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

// Outputs per-LM-iteration MSE convergence as CSV to stdout.
// Columns: solver,iteration,time_ms,mse_px2
//
// MSE convention matches the existing COLMAP reporting (sqrt(cost/n) = RMSE):
//   Ceres:  mse = cost / num_residuals
//   Caspar: mse = score_best / num_residuals
// Usage:
//   bundle_adjustment_convergence \
//       [--track_length=N] [--num_frames=N] [--num_points3D=N] \
//       [--max_ceres_iterations=N] [--max_caspar_iterations=N]

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <iostream>
#include <string>
#include <string_view>

using namespace colmap;

namespace {

int ParseIntFlag(int argc, char** argv, std::string_view prefix, int default_val) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg.substr(0, prefix.size()) == prefix) {
      return std::stoi(std::string(arg.substr(prefix.size())));
    }
  }
  return default_val;
}

}  // namespace

int main(int argc, char** argv) {
  // -1 means "use solver default". Override only when flag is provided.
  const int max_ceres_iterations =
      ParseIntFlag(argc, argv, "--max_ceres_iterations=", -1);
  const int max_caspar_iterations =
      ParseIntFlag(argc, argv, "--max_caspar_iterations=", -1);
  const int track_length = ParseIntFlag(argc, argv, "--track_length=", 20);
  const int num_frames = ParseIntFlag(argc, argv, "--num_frames=", 50);
  const int num_points3D = ParseIntFlag(argc, argv, "--num_points3D=", 5000);

  SetPRNGSeed(42);

  SyntheticDatasetOptions dataset_options;
  dataset_options.track_length = track_length;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 1;
  dataset_options.num_frames_per_rig = num_frames;
  dataset_options.num_points3D = num_points3D;

  Reconstruction reconstruction;
  SynthesizeDataset(dataset_options, &reconstruction);

  SyntheticNoiseOptions noise_options;
  noise_options.point2D_stddev = 1.0;
  noise_options.point3D_stddev = 0.05;
  noise_options.rig_from_world_translation_stddev = 0.01;
  noise_options.rig_from_world_rotation_stddev = 1.0;
  SynthesizeNoise(noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  BundleAdjustmentOptions options;
  options.print_summary = false;
  options.refine_sensor_from_rig = false;
  if (max_ceres_iterations >= 0) {
    options.ceres->solver_options.max_num_iterations = max_ceres_iterations;
  }
  if (max_caspar_iterations >= 0) {
    options.caspar->solver_iter_max = max_caspar_iterations;
  }
  options.caspar->collect_iteration_data = true;

  std::cerr << "Scene: track_length=" << track_length
            << " num_frames=" << num_frames
            << " num_points3D=" << num_points3D
            << " max_ceres_iterations="
            << options.ceres->solver_options.max_num_iterations
            << " max_caspar_iterations=" << options.caspar->solver_iter_max
            << "\n";

  std::cout << "solver,iteration,time_ms,mse_px2\n";

  // Ceres
  {
    Reconstruction copy = reconstruction;
    BundleAdjustmentOptions opts = options;
    opts.backend = BundleAdjustmentBackend::CERES;
    auto ba = CreateDefaultBundleAdjuster(opts, config, copy);
    const auto summary = ba->Solve();
    const auto* s = dynamic_cast<const CeresBundleAdjustmentSummary*>(summary.get());
    if (!s) {
      std::cerr << "ERROR: unexpected Ceres summary type\n";
      return 1;
    }
    const double n = static_cast<double>(s->num_residuals);
    const double t0_s = s->ceres_summary.iterations.empty()
                            ? 0.0
                            : s->ceres_summary.iterations[0].cumulative_time_in_seconds;
    for (const auto& iter : s->ceres_summary.iterations) {
      const double mse = iter.cost / n;
      const double time_ms = (iter.cumulative_time_in_seconds - t0_s) * 1000.0;
      std::cout << "ceres," << iter.iteration << "," << time_ms << "," << mse << "\n";
    }
    std::cerr << "Ceres: " << s->ceres_summary.iterations.size()
              << " iterations, final mse="
              << s->ceres_summary.final_cost / n << " px^2\n";
  }

#ifdef CASPAR_ENABLED
  // Caspar
  {
    Reconstruction copy = reconstruction;
    BundleAdjustmentOptions opts = options;
    opts.backend = BundleAdjustmentBackend::CASPAR;
    auto ba = CreateDefaultBundleAdjuster(opts, config, copy);
    const auto summary = ba->Solve();
    const auto* s = dynamic_cast<const CasparBundleAdjustmentSummary*>(summary.get());
    if (!s) {
      std::cerr << "ERROR: unexpected Caspar summary type\n";
      return 1;
    }
    const double n = static_cast<double>(s->num_residuals);
    // Emit initial state (iteration -1) at t=0
    std::cout << "caspar,-1,0," << (s->initial_score / n) << "\n";
    for (const auto& iter : s->iterations) {
      const double mse = iter.score_best / n;
      const double time_ms = iter.dt_tot * 1000.0;
      std::cout << "caspar," << iter.solver_iter << "," << time_ms << "," << mse << "\n";
    }
    const double final_mse = s->iterations.empty()
                                 ? s->initial_score / n
                                 : s->iterations.back().score_best / n;
    std::cerr << "Caspar: " << s->iteration_count << " iterations, final mse="
              << final_mse << " px^2\n";
  }
#endif

  return 0;
}
