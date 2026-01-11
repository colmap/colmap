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

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/pose.h"
#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include "glomap/estimators/gravity_refinement.h"
#include "glomap/estimators/rotation_averaging.h"
#include "glomap/sfm/global_mapper.h"

#include <limits>

namespace colmap {

RotationAveragingController::RotationAveragingController(
    const RotationAveragingControllerOptions& options,
    std::shared_ptr<Database> database,
    std::shared_ptr<Reconstruction> reconstruction)
    : options_(options),
      reconstruction_(std::move(THROW_CHECK_NOTNULL(reconstruction))) {
  THROW_CHECK_NOTNULL(database);
  if (options_.decompose_relative_pose) {
    MaybeDecomposeAndWriteRelativePoses(database.get());
  }
  DatabaseCache::Options database_cache_options;
  database_cache_options.min_num_matches = options_.min_num_matches;
  database_cache_options.ignore_watermarks = options_.ignore_watermarks;
  database_cache_options.image_names = {options_.image_names.begin(),
                                        options_.image_names.end()};
  database_cache_ = DatabaseCache::Create(*database, database_cache_options);
}

void RotationAveragingController::Run() {
  // Propagate options to component options.
  RotationAveragingControllerOptions options = options_;
  options.rotation_estimation.random_seed = options.random_seed;
  options.gravity_refiner.solver_options.num_threads = options.num_threads;

  Timer run_timer;
  run_timer.Start();

  // Create a global mapper instance to setup pose graph and reconstruction.
  glomap::GlobalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction_);

  if (mapper.PoseGraph()->Empty()) {
    LOG(ERROR) << "Cannot continue without image pairs";
    return;
  }

  // Get a mutable copy of pose priors.
  std::vector<PosePrior> pose_priors = database_cache_->PosePriors();

  // Skip MST initialization if gravity priors exist (we initialize from gravity
  // instead). Otherwise, use MST initialization.
  options.rotation_estimation.skip_initialization = !pose_priors.empty();

  // Initialize frame rotations from gravity priors.
  if (!pose_priors.empty()) {
    const Eigen::Vector3d kUnknownTranslation =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    for (const auto& pose_prior : pose_priors) {
      const auto& image = reconstruction_->Image(pose_prior.pose_prior_id);
      if (!image.IsRefInFrame()) {
        continue;
      }
      reconstruction_->Frame(image.FrameId())
          .SetRigFromWorld(Rigid3d(
              Eigen::Quaterniond(GravityAlignedRotation(pose_prior.gravity)),
              kUnknownTranslation));
    }
  }

  // Optionally refine gravity priors (only if gravity priors exist).
  if (options.refine_gravity && !pose_priors.empty()) {
    // Compute largest connected component and invalidate pairs before gravity
    // refinement.
    const std::unordered_set<frame_t> active_frame_ids =
        mapper.PoseGraph()->ComputeLargestConnectedFrameComponent(
            *reconstruction_,
            /*filter_unregistered=*/false);
    std::unordered_set<image_t> active_image_ids;
    for (const auto& [image_id, image] : reconstruction_->Images()) {
      if (active_frame_ids.count(image.FrameId())) {
        active_image_ids.insert(image_id);
      }
    }
    mapper.PoseGraph()->InvalidatePairsOutsideActiveImageIds(active_image_ids);

    LOG(INFO) << "----- Running gravity refinement -----";
    glomap::RunGravityRefinement(options.gravity_refiner,
                                 *mapper.PoseGraph(),
                                 *reconstruction_,
                                 pose_priors);
  }

  LOG(INFO) << "----- Running rotation averaging -----";
  if (!glomap::SolveRotationAveraging(options.rotation_estimation,
                                      *mapper.PoseGraph(),
                                      *reconstruction_,
                                      pose_priors)) {
    LOG(ERROR) << "Failed to solve rotation averaging";
    return;
  }

  LOG(INFO) << "Rotation averaging done in " << run_timer.ElapsedSeconds()
            << "s";
}

}  // namespace colmap
