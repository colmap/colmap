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

#include "colmap/controllers/reconstruction_pruning.h"

#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include "glomap/processors/reconstruction_pruning.h"

namespace colmap {
namespace {

// Extract a subset of the reconstruction for a specific cluster.
// Returns a new Reconstruction containing only frames/images/points from the
// specified cluster.
Reconstruction ExtractClusterReconstruction(
    const Reconstruction& reconstruction,
    const std::unordered_map<glomap::frame_t, int>& cluster_ids,
    int cluster_id) {
  // Helper to get cluster id for a frame
  auto get_cluster_id =
      [&cluster_ids](glomap::frame_t frame_id) -> int {
    auto it = cluster_ids.find(frame_id);
    return it != cluster_ids.end() ? it->second : -1;
  };

  // Make a copy of the reconstruction
  Reconstruction filtered = reconstruction;

  // Collect frames to deregister (those not in this cluster)
  std::vector<glomap::frame_t> frames_to_deregister;
  for (const auto& [frame_id, frame] : filtered.Frames()) {
    if (!frame.HasPose() || get_cluster_id(frame_id) != cluster_id) {
      frames_to_deregister.push_back(frame_id);
    }
  }

  // Deregister frames not in this cluster
  // This also removes point observations from those frames' images
  for (glomap::frame_t frame_id : frames_to_deregister) {
    if (filtered.Frame(frame_id).HasPose()) {
      filtered.DeRegisterFrame(frame_id);
    }
  }

  filtered.UpdatePoint3DErrors();
  return filtered;
}

}  // namespace

ReconstructionPruningController::ReconstructionPruningController(
    const ReconstructionPruningOptions& options,
    std::shared_ptr<Reconstruction> reconstruction)
    : options_(options), reconstruction_(std::move(reconstruction)) {}

void ReconstructionPruningController::Run() {
  THROW_CHECK_NOTNULL(reconstruction_);

  LOG_HEADING1("Pruning weakly connected frames");
  Timer timer;
  timer.Start();
  std::unordered_map<glomap::frame_t, int> cluster_ids =
      glomap::PruneWeaklyConnectedFrames(options_.pruning, *reconstruction_);
  LOG(INFO) << "Pruning done in " << timer.ElapsedSeconds() << " seconds";

  LOG(INFO) << "Number of frames after pruning: "
            << reconstruction_->NumRegFrames();

  // Find max cluster id
  int max_cluster_id = -1;
  for (const auto& [frame_id, cluster_id] : cluster_ids) {
    if (cluster_id > max_cluster_id) {
      max_cluster_id = cluster_id;
    }
  }

  LOG_HEADING1("Writing pruned model(s)");

  // If no clusters (or single cluster), output as single reconstruction
  if (max_cluster_id <= 0) {
    reconstruction_->Write(options_.output_path);
  } else {
    // Split by cluster and output multiple reconstructions
    for (int comp = 0; comp <= max_cluster_id; comp++) {
      Reconstruction cluster_reconstruction =
          ExtractClusterReconstruction(*reconstruction_, cluster_ids, comp);
      const auto reconstruction_path =
          options_.output_path / std::to_string(comp);
      CreateDirIfNotExists(reconstruction_path);
      cluster_reconstruction.Write(reconstruction_path);
    }
    LOG(INFO) << "Exported " << max_cluster_id + 1 << " reconstructions";
  }
}

}  // namespace colmap
