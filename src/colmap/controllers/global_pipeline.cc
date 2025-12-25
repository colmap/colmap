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

#include "glomap/io/colmap_converter.h"
#include "glomap/sfm/global_mapper.h"

namespace colmap {

GlobalPipeline::GlobalPipeline(
    const glomap::GlobalMapperOptions& options,
    const std::string& image_path,
    const std::string& database_path,
    std::shared_ptr<colmap::ReconstructionManager> reconstruction_manager)
    : options_(options),
      image_path_(image_path),
      database_path_(database_path),
      reconstruction_manager_(std::move(reconstruction_manager)) {}

void GlobalPipeline::Run() {
  auto database = Database::Open(database_path_);

  glomap::ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<frame_t, glomap::Frame> frames;
  std::unordered_map<image_t, glomap::Image> images;
  std::unordered_map<point3D_t, Point3D> tracks;
  glomap::ConvertDatabaseToGlomap(
      *database, view_graph, rigs, cameras, frames, images);
  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();

  if (view_graph.image_pairs.empty()) {
    LOG(ERROR) << "Cannot continue without image pairs";
    return;
  }

  Timer run_timer;
  run_timer.Start();
  glomap::GlobalMapper global_mapper(options_);
  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(database.get(),
                      view_graph,
                      rigs,
                      cameras,
                      frames,
                      images,
                      tracks,
                      pose_priors,
                      cluster_ids);
  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  int largest_component_num = -1;
  for (const auto& [frame_id, cluster_id] : cluster_ids) {
    if (cluster_id > largest_component_num) largest_component_num = cluster_id;
  }

  // If it is not separated into several clusters, then output them as whole.
  if (largest_component_num == -1) {
    colmap::Reconstruction& reconstruction =
        *reconstruction_manager_->Get(reconstruction_manager_->Add());
    glomap::ConvertGlomapToColmap(
        rigs, cameras, frames, images, tracks, reconstruction, cluster_ids);
    // Read in colors
    if (image_path_ != "") {
      LOG(INFO) << "Extracting colors ...";
      reconstruction.ExtractColorsForAllImages(image_path_);
    }
  } else {
    for (int comp = 0; comp <= largest_component_num; comp++) {
      std::cout << "\r Exporting reconstruction " << comp + 1 << " / "
                << largest_component_num + 1 << std::flush;
      colmap::Reconstruction& reconstruction =
          *reconstruction_manager_->Get(reconstruction_manager_->Add());
      glomap::ConvertGlomapToColmap(rigs,
                                    cameras,
                                    frames,
                                    images,
                                    tracks,
                                    reconstruction,
                                    cluster_ids,
                                    comp);
      // Read in colors
      if (image_path_ != "") {
        reconstruction.ExtractColorsForAllImages(image_path_);
      }
    }
    std::cout << '\n';
  }
}

}  // namespace colmap
