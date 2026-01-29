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

#include "colmap/controllers/reconstruction_clustering.h"

#include "colmap/scene/reconstruction_clustering.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

namespace colmap {
namespace {

// Extract a subset of the reconstruction for a specific cluster.
// Returns a new Reconstruction containing only frames/images/points from the
// specified cluster.
std::shared_ptr<Reconstruction> SubReconstructionByClusterId(
    const Reconstruction& reconstruction,
    const std::unordered_map<frame_t, int>& cluster_ids,
    int cluster_id) {
  // Helper to get cluster id for a frame
  auto get_cluster_id = [&cluster_ids](frame_t frame_id) -> int {
    auto it = cluster_ids.find(frame_id);
    return it != cluster_ids.end() ? it->second : -1;
  };

  // Make a copy of the reconstruction
  auto filtered = std::make_shared<Reconstruction>(reconstruction);

  // Collect frames to deregister (those not in this cluster)
  std::vector<frame_t> frames_to_deregister;
  for (const auto& [frame_id, frame] : filtered->Frames()) {
    if (!frame.HasPose() || get_cluster_id(frame_id) != cluster_id) {
      frames_to_deregister.push_back(frame_id);
    }
  }

  // Deregister frames not in this cluster
  // This also removes point observations from those frames' images
  for (frame_t frame_id : frames_to_deregister) {
    if (filtered->Frame(frame_id).HasPose()) {
      filtered->DeRegisterFrame(frame_id);
    }
  }

  filtered->UpdatePoint3DErrors();
  return filtered;
}

}  // namespace

ReconstructionClustererController::ReconstructionClustererController(
    const ReconstructionClusteringOptions& options,
    std::shared_ptr<Reconstruction> reconstruction,
    std::shared_ptr<ReconstructionManager> reconstruction_manager)
    : options_(options),
      reconstruction_(std::move(reconstruction)),
      reconstruction_manager_(std::move(reconstruction_manager)) {}

void ReconstructionClustererController::Run() {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(reconstruction_manager_);

  LOG_HEADING1("Pruning weakly connected frames");
  Timer timer;
  timer.Start();
  std::unordered_map<frame_t, int> cluster_ids =
      ClusterReconstructionFrames(options_, *reconstruction_);
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

  // Clear any existing reconstructions
  reconstruction_manager_->Clear();

  // If no clusters (or single cluster), add the single reconstruction
  // Note that cluster_id start from 0, so max_cluster_id of -1 means no
  // clusters
  if (max_cluster_id < 0) {
    if (reconstruction_->NumRegFrames() >=
        static_cast<size_t>(options_.min_num_reg_frames)) {
      size_t idx = reconstruction_manager_->Add();
      *reconstruction_manager_->Get(idx) = *reconstruction_;
    } else {
      LOG(WARNING) << "Reconstruction has only "
                   << reconstruction_->NumRegFrames()
                   << " registered frames, below minimum threshold of "
                   << options_.min_num_reg_frames;
    }
  } else {
    // For invalid frames, clusters ids are -1 and are skipped automatically
    // Split by cluster and add multiple reconstructions
    for (int comp = 0; comp <= max_cluster_id; comp++) {
      std::shared_ptr<Reconstruction> cluster_reconstruction =
          SubReconstructionByClusterId(*reconstruction_, cluster_ids, comp);
      THROW_CHECK_GE(
          cluster_reconstruction->NumRegFrames(),
          static_cast<size_t>(
              options_.min_num_reg_frames));  // Should always be true
      const size_t num_reg_frames = cluster_reconstruction->NumRegFrames();
      size_t idx = reconstruction_manager_->Add();
      reconstruction_manager_->Get(idx) = std::move(cluster_reconstruction);
      LOG(INFO) << "Added cluster " << comp << " with " << num_reg_frames
                << " registered frames";
    }
    LOG(INFO) << "Created " << reconstruction_manager_->Size()
              << " cluster reconstructions";
  }
}

}  // namespace colmap
