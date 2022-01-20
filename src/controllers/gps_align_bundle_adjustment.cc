// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "controllers/gps_align_bundle_adjustment.h"

#include <ceres/ceres.h>

#include "optim/bundle_adjustment.h"
#include "util/misc.h"

#include "base/gps.h"

namespace colmap {
namespace {

// Callback functor called after each bundle adjustment iteration.
class GPSAlignBundleAdjustmentIterationCallback
    : public ceres::IterationCallback {
 public:
  explicit GPSAlignBundleAdjustmentIterationCallback(Thread* thread)
      : thread_(thread) {}

  virtual ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) {
    CHECK_NOTNULL(thread_);
    thread_->BlockIfPaused();
    if (thread_->IsStopped()) {
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    } else {
      return ceres::SOLVER_CONTINUE;
    }
  }

 private:
  Thread* thread_;
};

}  // namespace

GPSAlignBundleAdjustmentController::GPSAlignBundleAdjustmentController(
    const OptionManager& options, Reconstruction* reconstruction)
    : options_(options), reconstruction_(reconstruction) {}

void GPSAlignBundleAdjustmentController::Run() {
  CHECK_NOTNULL(reconstruction_);

  PrintHeading1("Global bundle adjustment with GPS Alignment");

  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  if (reg_image_ids.size() < 2) {
    std::cout << "ERROR: Need at least two views." << std::endl;
    return;
  }

  // Avoid degeneracies in bundle adjustment.
  reconstruction_->FilterObservationsWithNegativeDepth();

  BundleAdjustmentOptions ba_options = *options_.bundle_adjustment;
  ba_options.solver_options.minimizer_progress_to_stdout = true;

  GPSAlignBundleAdjustmentIterationCallback iteration_callback(this);
  ba_options.solver_options.callbacks.push_back(&iteration_callback);

  // Configure bundle adjustment.
  // bool is_gauge_fixed = true;
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  std::cout << "\n\nGPS-SfM Details -- Pre-BA : \n";
  std::cout << ">>> Num Images : " << reconstruction_->NumImages() << "\n";
  std::cout << ">>> Num 2D-3D obs : " << reconstruction_->ComputeNumObservations() << "\n";
  std::cout << ">>> Num 3D Points : " << reconstruction_->NumPoints3D() << "\n\n";

  // Run bundle adjustment.
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  bundle_adjuster.Solve(reconstruction_);

  // Filter GPS-SfM Results.
  reconstruction_->FilterObservationsWithNegativeDepth();
  reconstruction_->FilterAllPoints3D(3.0, 0.0);

  std::cout << "\n\nGPS-SfM Details -- Post-BA : \n";
  std::cout << ">>> Num Images : " << reconstruction_->NumImages() << "\n";
  std::cout << ">>> Num 2D-3D obs : " << reconstruction_->ComputeNumObservations() << "\n";
  std::cout << ">>> Num 3D Points : " << reconstruction_->NumPoints3D() << "\n\n";

  GetTimer().PrintMinutes();
}

}  // namespace colmap
