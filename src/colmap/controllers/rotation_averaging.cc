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

#include "glomap/sfm/global_mapper.h"

namespace colmap {

RotationAveragingController::RotationAveragingController(
    const RotationAveragingControllerOptions& options,
    std::shared_ptr<Database> database,
    std::shared_ptr<Reconstruction> reconstruction)
    : options_(options),
      database_(std::move(THROW_CHECK_NOTNULL(database))),
      reconstruction_(std::move(THROW_CHECK_NOTNULL(reconstruction))) {}

void RotationAveragingController::Run() {
  // Propagate options to component options.
  RotationAveragingControllerOptions options = options_;
  if (options.random_seed >= 0) {
    options.rotation_estimation.random_seed = options.random_seed;
  }

  Timer run_timer;
  run_timer.Start();

  // Create a global mapper instance
  glomap::GlobalMapper mapper(database_);
  mapper.BeginReconstruction(reconstruction_);

  if (mapper.ViewGraph()->Empty()) {
    LOG(ERROR) << "Cannot continue without image pairs";
    return;
  }

  LOG(INFO) << "----- Running rotation averaging -----";
  if (!mapper.RotationAveraging(options.rotation_estimation,
                                options.max_rotation_error_deg)) {
    LOG(ERROR) << "Failed to solve rotation averaging";
    return;
  }

  LOG(INFO) << "Rotation averaging done in " << run_timer.ElapsedSeconds()
            << "s";
}

}  // namespace colmap
