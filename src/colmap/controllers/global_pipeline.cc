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

#include "colmap/controllers/global_pipeline.h"

#include "colmap/estimators/alignment.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/scene/database_cache.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include "glomap/sfm/global_mapper.h"

namespace colmap {

GlobalPipeline::GlobalPipeline(
    const GlobalPipelineOptions& options,
    std::shared_ptr<Database> database,
    std::shared_ptr<colmap::ReconstructionManager> reconstruction_manager)
    : options_(options),
      database_(std::move(THROW_CHECK_NOTNULL(database))),
      reconstruction_manager_(
          std::move(THROW_CHECK_NOTNULL(reconstruction_manager))) {
  if (options_.decompose_relative_pose) {
    MaybeDecomposeAndWriteRelativePoses(database_.get());
  }
}

void GlobalPipeline::Run() {
  if (!options_.skip_view_graph_calibration) {
    LOG_HEADING1("Running view graph calibration");
    Timer run_timer;
    run_timer.Start();
    ViewGraphCalibrationOptions vgc_options = options_.view_graph_calibration;
    vgc_options.random_seed = options_.random_seed;
    vgc_options.solver_options.num_threads = options_.num_threads;
    if (!CalibrateViewGraph(vgc_options, database_.get())) {
      LOG(ERROR) << "View graph calibration failed";
      return;
    }
    LOG(INFO) << "View graph calibration done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Create database cache with relative poses for pose graph.
  DatabaseCache::Options database_cache_options;
  database_cache_options.min_num_matches = options_.min_num_matches;
  database_cache_options.ignore_watermarks = options_.ignore_watermarks;
  database_cache_options.image_names = {options_.image_names.begin(),
                                        options_.image_names.end()};
  auto database_cache =
      DatabaseCache::Create(*database_, database_cache_options);

  auto reconstruction = std::make_shared<Reconstruction>();

  // Prepare mapper options with top-level options.
  glomap::GlobalMapperOptions mapper_options = options_.mapper;
  mapper_options.image_path = options_.image_path;
  mapper_options.num_threads = options_.num_threads;
  mapper_options.random_seed = options_.random_seed;

  glomap::GlobalMapper global_mapper(database_cache);
  global_mapper.BeginReconstruction(reconstruction);

  Timer run_timer;
  run_timer.Start();
  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(mapper_options, cluster_ids);
  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  // Align reconstruction to the original metric scales in rig extrinsics.
  AlignReconstructionToOrigRigScales(database_cache->Rigs(),
                                     reconstruction.get());

  // Output the reconstruction.
  Reconstruction& output_reconstruction =
      *reconstruction_manager_->Get(reconstruction_manager_->Add());
  output_reconstruction = *reconstruction;
  if (!options_.image_path.empty()) {
    LOG(INFO) << "Extracting colors ...";
    output_reconstruction.ExtractColorsForAllImages(options_.image_path);
  }
}

}  // namespace colmap
