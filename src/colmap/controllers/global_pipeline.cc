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

#include "colmap/util/timer.h"

#include "glomap/io/colmap_io.h"
#include "glomap/sfm/global_mapper.h"

namespace colmap {

GlobalPipeline::GlobalPipeline(
    const GlobalPipelineOptions& options,
    std::shared_ptr<Database> database,
    std::shared_ptr<colmap::ReconstructionManager> reconstruction_manager)
    : options_(options),
      database_(std::move(THROW_CHECK_NOTNULL(database))),
      reconstruction_manager_(
          std::move(THROW_CHECK_NOTNULL(reconstruction_manager))) {}

void GlobalPipeline::Run() {
  // Run view graph calibration on the database before loading into mapper.
  if (!options_.skip_view_graph_calibration) {
    LOG(INFO) << "----- Running view graph calibration -----";
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

  auto reconstruction = std::make_shared<Reconstruction>();

  glomap::GlobalMapper global_mapper(database_);
  global_mapper.BeginReconstruction(reconstruction);

  if (global_mapper.ViewGraph()->Empty()) {
    LOG(ERROR) << "Cannot continue without image pairs";
    return;
  }

  // Prepare mapper options with top-level options.
  glomap::GlobalMapperOptions mapper_options = options_.mapper;
  mapper_options.image_path = options_.image_path;
  mapper_options.num_threads = options_.num_threads;
  mapper_options.random_seed = options_.random_seed;

  Timer run_timer;
  run_timer.Start();
  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(mapper_options, cluster_ids);
  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  int max_cluster_id = -1;
  for (const auto& [frame_id, cluster_id] : cluster_ids) {
    if (cluster_id > max_cluster_id) {
      max_cluster_id = cluster_id;
    }
  }

  // If it is not separated into several clusters, then output them as whole.
  if (max_cluster_id == -1) {
    Reconstruction& output_reconstruction =
        *reconstruction_manager_->Get(reconstruction_manager_->Add());
    output_reconstruction = *reconstruction;
    if (!options_.image_path.empty()) {
      LOG(INFO) << "Extracting colors ...";
      output_reconstruction.ExtractColorsForAllImages(options_.image_path);
    }
  } else {
    for (int comp = 0; comp <= max_cluster_id; comp++) {
      Reconstruction& output_reconstruction =
          *reconstruction_manager_->Get(reconstruction_manager_->Add());
      output_reconstruction = glomap::SubReconstructionByClusterId(
          *reconstruction, cluster_ids, comp);
      if (!options_.image_path.empty()) {
        output_reconstruction.ExtractColorsForAllImages(options_.image_path);
      }
    }
    LOG(INFO) << "Exported " << max_cluster_id + 1 << " reconstructions";
  }
}

}  // namespace colmap
