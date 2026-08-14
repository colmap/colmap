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

#include "colmap/controllers/base_option_manager.h"
#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"

#include <chrono>
#include <iostream>
#include <string>

using namespace colmap;

int main(int argc, char** argv) {
  // -1 means "use solver default". Override only when flag is provided.
  int max_ceres_iterations = -1;
  int max_caspar_iterations = -1;
  int track_length = 20;
  int num_frames = 50;
  int num_points3D = 5000;
  int num_cameras_per_rig = 1;
  bool refine_focal_length = false;
  std::string label = "default";

  BaseOptionManager args(/*add_project_options=*/false);
  args.AddDefaultOption("max_ceres_iterations", &max_ceres_iterations);
  args.AddDefaultOption("max_caspar_iterations", &max_caspar_iterations);
  args.AddDefaultOption("track_length", &track_length);
  args.AddDefaultOption("num_frames", &num_frames);
  args.AddDefaultOption("num_points3D", &num_points3D);
  args.AddDefaultOption("num_cameras_per_rig", &num_cameras_per_rig);
  args.AddDefaultOption("refine_focal_length", &refine_focal_length);
  args.AddDefaultOption("label", &label);
  if (!args.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  SetPRNGSeed(42);

  SyntheticDatasetOptions dataset_options;
  dataset_options.track_length = track_length;
  dataset_options.num_cameras_per_rig = num_cameras_per_rig;
  dataset_options.num_points3D = num_points3D;
  // Use PINHOLE so fx≠fy refinement is testable without distortion coupling.
  dataset_options.camera_model_id = PinholeCameraModel::model_id;
  dataset_options.camera_params = {1280, 1280, 512, 384};

  dataset_options.num_rigs = 1;
  dataset_options.num_frames_per_rig = num_frames;

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
  options.refine_focal_length = refine_focal_length;
  options.refine_sensor_from_rig = false;
  if (max_ceres_iterations >= 0) {
    options.ceres->solver_options.max_num_iterations = max_ceres_iterations;
  }
  if (max_caspar_iterations >= 0) {
    options.caspar->solver_iter_max = max_caspar_iterations;
  }
  options.caspar->collect_iteration_data = true;

  std::cerr << "Scene: track_length=" << track_length
            << " num_frames=" << num_frames << " num_points3D=" << num_points3D
            << " num_cameras_per_rig=" << num_cameras_per_rig
            << " refine_focal_length=" << refine_focal_length
            << " label=" << label << " max_ceres_iterations="
            << options.ceres->solver_options.max_num_iterations
            << " max_caspar_iterations=" << options.caspar->solver_iter_max
            << "\n";

  std::cout << "solver,scenario,iteration,time_ms,mse_px2\n";
  // Ceres
  {
    Reconstruction copy = reconstruction;
    BundleAdjustmentOptions opts = options;
    opts.backend = BundleAdjustmentBackend::CERES;

    // Measure setup/build time.
    const auto t_setup_start = std::chrono::steady_clock::now();
    auto ba = CreateDefaultBundleAdjuster(opts, config, copy);
    const auto t_setup_end = std::chrono::steady_clock::now();

    const double setup_time_ms =
        std::chrono::duration<double, std::milli>(t_setup_end - t_setup_start)
            .count();

    // Measure total Solve() wall time.
    const auto t_solve_start = std::chrono::steady_clock::now();
    const auto summary = ba->Solve();
    const auto t_solve_end = std::chrono::steady_clock::now();

    const double solve_wall_ms =
        std::chrono::duration<double, std::milli>(t_solve_end - t_solve_start)
            .count();

    const auto* s =
        dynamic_cast<const CeresBundleAdjustmentSummary*>(summary.get());
    if (!s) {
      std::cerr << "ERROR: unexpected Ceres summary type\n";
      return 1;
    }

    const double n = static_cast<double>(s->num_residuals);

    // Ceres cumulative iteration time reported internally.
    const double iter_total_ms =
        s->ceres_summary.iterations.empty()
            ? 0.0
            : s->ceres_summary.iterations.back().cumulative_time_in_seconds *
                  1000.0;

    // Solve() wall time not accounted for by iteration timing.
    const double solve_overhead_ms =
        std::max(0.0, solve_wall_ms - iter_total_ms);

    // Emit t=0 anchor so the plot origin is consistent with Caspar.
    if (!s->ceres_summary.iterations.empty()) {
      std::cout << "ceres," << label << ",-1,0,"
                << s->ceres_summary.iterations.front().cost / n << "\n";
    }

    for (const auto& iter : s->ceres_summary.iterations) {
      const double mse = iter.cost / n;

      // Include:
      //   setup time
      // + Solve() dispatch/overhead
      // + cumulative iteration time (includes initial eval — same convention
      //   as Caspar's dt_tot which includes DoResJacFirst)
      const double time_ms = setup_time_ms + solve_overhead_ms +
                             iter.cumulative_time_in_seconds * 1000.0;

      std::cout << "ceres," << label << "," << iter.iteration << "," << time_ms
                << "," << mse << "\n";
    }

    std::cerr << "Ceres: " << s->ceres_summary.iterations.size()
              << " iterations, final mse=" << s->ceres_summary.final_cost / n
              << " px^2"
              << ", setup_ms=" << setup_time_ms
              << ", solve_overhead_ms=" << solve_overhead_ms
              << ", iter_ms=" << iter_total_ms
              << ", solve_wall_ms=" << solve_wall_ms << "\n";
  }
#ifdef CASPAR_ENABLED
  // Caspar
  {
    Reconstruction copy = reconstruction;
    BundleAdjustmentOptions opts = options;
    opts.backend = BundleAdjustmentBackend::CASPAR;

    // Measure setup/build time.
    const auto t_setup_start = std::chrono::steady_clock::now();
    auto ba = CreateDefaultBundleAdjuster(opts, config, copy);
    const auto t_setup_end = std::chrono::steady_clock::now();

    const double setup_time_ms =
        std::chrono::duration<double, std::milli>(t_setup_end - t_setup_start)
            .count();

    // Measure total Solve() wall time.
    const auto t_solve_start = std::chrono::steady_clock::now();
    const auto summary = ba->Solve();
    const auto t_solve_end = std::chrono::steady_clock::now();

    const double solve_wall_ms =
        std::chrono::duration<double, std::milli>(t_solve_end - t_solve_start)
            .count();

    const auto* s =
        dynamic_cast<const CasparBundleAdjustmentSummary*>(summary.get());
    if (!s) {
      std::cerr << "ERROR: unexpected Caspar summary type\n";
      return 1;
    }

    const double n = static_cast<double>(s->num_residuals);

    // Caspar internal measured iteration time.
    const double iter_total_ms =
        s->iterations.empty() ? 0.0 : s->iterations.back().dt_tot * 1000.0;

    // Everything inside Solve() that is NOT accounted for by iteration timing.
    const double solve_overhead_ms =
        std::max(0.0, solve_wall_ms - iter_total_ms);

    // Emit initial state (iteration -1) at t=0
    std::cout << "caspar," << label << ",-1,0," << (s->initial_score / n)
              << "\n";

    for (const auto& iter : s->iterations) {
      const double mse = iter.score_best / n;

      // Include:
      //   setup time
      // + Solve() dispatch/overhead
      // + cumulative iteration time
      const double time_ms =
          setup_time_ms + solve_overhead_ms + iter.dt_tot * 1000.0;

      std::cout << "caspar," << label << "," << iter.solver_iter << ","
                << time_ms << "," << mse << "\n";
    }

    const double final_mse = s->iterations.empty()
                                 ? s->initial_score / n
                                 : s->iterations.back().score_best / n;

    std::cerr << "Caspar: " << s->iteration_count
              << " iterations, final mse=" << final_mse << " px^2"
              << ", setup_ms=" << setup_time_ms
              << ", solve_overhead_ms=" << solve_overhead_ms
              << ", iter_ms=" << iter_total_ms
              << ", solve_wall_ms=" << solve_wall_ms << "\n";
  }
#endif

  return 0;
}
