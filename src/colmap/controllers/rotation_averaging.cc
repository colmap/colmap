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

#include "colmap/controllers/rotation_averaging.h"

#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include "glomap/io/colmap_io.h"
#include "glomap/processors/view_graph_manipulation.h"

namespace colmap {

RotationAveragingController::RotationAveragingController(
    const RotationAveragingControllerOptions& options,
    std::shared_ptr<Database> database,
    std::shared_ptr<Reconstruction> reconstruction)
    : options_(options),
      database_(std::move(THROW_CHECK_NOTNULL(database))),
      reconstruction_(std::move(THROW_CHECK_NOTNULL(reconstruction))) {}

void RotationAveragingController::Run() {
  // Initialize view graph from database.
  glomap::ViewGraph view_graph;
  glomap::InitializeGlomapFromDatabase(
      *database_, *reconstruction_, view_graph);

  // Read pose priors from database.
  std::vector<PosePrior> pose_priors = database_->ReadAllPosePriors();

  if (view_graph.image_pairs.empty()) {
    LOG(ERROR) << "Cannot continue without image pairs";
    return;
  }

  Timer run_timer;
  run_timer.Start();

  // Step 0: Preprocessing
  LOG(INFO) << "----- Running preprocessing -----";
  glomap::ViewGraphManipulator::UpdateImagePairsConfig(
      view_graph, *reconstruction_);
  glomap::ViewGraphManipulator::DecomposeRelPose(view_graph, *reconstruction_);

  // Step 1: View graph calibration
  LOG(INFO) << "----- Running view graph calibration -----";
  glomap::ViewGraphCalibrator calibrator(options_.view_graph_calibration);
  if (!calibrator.Solve(view_graph, *reconstruction_)) {
    LOG(ERROR) << "Failed to solve view graph calibration";
    return;
  }

  // Step 2: Relative pose estimation
  LOG(INFO) << "----- Running relative pose estimation -----";
  glomap::EstimateRelativePoses(
      view_graph, *reconstruction_, options_.relative_pose_estimation);

  glomap::ImagePairsInlierCount(
      view_graph, *reconstruction_, options_.inlier_thresholds, true);

  view_graph.FilterByNumInliers(options_.inlier_thresholds.min_inlier_num);
  view_graph.FilterByInlierRatio(options_.inlier_thresholds.min_inlier_ratio);

  if (view_graph.KeepLargestConnectedComponents(*reconstruction_) == 0) {
    LOG(ERROR) << "No connected components found";
    return;
  }

  // Step 3: Rotation averaging
  LOG(INFO) << "----- Running rotation averaging -----";
  if (!glomap::SolveRotationAveraging(
          view_graph, *reconstruction_, pose_priors, options_.rotation_estimation)) {
    LOG(ERROR) << "Failed to solve rotation averaging";
    return;
  }

  LOG(INFO) << "Rotation averaging done in " << run_timer.ElapsedSeconds()
            << " seconds";
}

}  // namespace colmap
