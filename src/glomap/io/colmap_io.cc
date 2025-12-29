#include "glomap/io/colmap_io.h"

#include "colmap/feature/utils.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

namespace {

void WriteReconstruction(const colmap::Reconstruction& reconstruction,
                         const std::string& path,
                         const std::string& output_format,
                         const std::string& image_path) {
  colmap::Reconstruction recon_copy = reconstruction;
  if (!image_path.empty()) {
    LOG(INFO) << "Extracting colors ...";
    recon_copy.ExtractColorsForAllImages(image_path);
  }
  colmap::CreateDirIfNotExists(path, true);
  if (output_format == "txt") {
    recon_copy.WriteText(path);
  } else if (output_format == "bin") {
    recon_copy.WriteBinary(path);
  } else {
    LOG(ERROR) << "Unsupported output type";
  }
}

}  // namespace

namespace glomap {

colmap::Reconstruction SubReconstructionByClusterId(
    const colmap::Reconstruction& reconstruction,
    const std::unordered_map<frame_t, int>& cluster_ids,
    int cluster_id) {
  // If no filtering needed, return a copy
  if (cluster_id == -1 || cluster_ids.empty()) {
    return reconstruction;
  }

  // Helper to get cluster id for a frame
  auto get_cluster_id = [&cluster_ids](frame_t frame_id) -> int {
    auto it = cluster_ids.find(frame_id);
    return it != cluster_ids.end() ? it->second : -1;
  };

  // Make a copy of the reconstruction
  colmap::Reconstruction filtered = reconstruction;

  // Collect frames to deregister (those not in this cluster)
  std::vector<frame_t> frames_to_deregister;
  for (const auto& [frame_id, frame] : filtered.Frames()) {
    if (!frame.HasPose() || get_cluster_id(frame_id) != cluster_id) {
      frames_to_deregister.push_back(frame_id);
    }
  }

  // Deregister frames not in this cluster
  // This also removes point observations from those frames' images
  for (frame_t frame_id : frames_to_deregister) {
    if (filtered.Frame(frame_id).HasPose()) {
      filtered.DeRegisterFrame(frame_id);
    }
  }

  filtered.UpdatePoint3DErrors();
  return filtered;
}

void WriteReconstructionsByClusters(
    const std::string& reconstruction_path,
    const colmap::Reconstruction& reconstruction,
    const std::unordered_map<frame_t, int>& cluster_ids,
    const std::string& output_format,
    const std::string& image_path) {
  // Find the maximum cluster id to determine if we have multiple clusters
  int max_cluster_id = -1;
  for (const auto& [frame_id, cluster_id] : cluster_ids) {
    if (cluster_id > max_cluster_id) {
      max_cluster_id = cluster_id;
    }
  }

  // If no clusters, output as single reconstruction
  if (max_cluster_id == -1) {
    WriteReconstruction(
        reconstruction, reconstruction_path + "/0", output_format, image_path);
  } else {
    // Export each cluster separately
    for (int comp = 0; comp <= max_cluster_id; comp++) {
      colmap::Reconstruction cluster_recon =
          SubReconstructionByClusterId(reconstruction, cluster_ids, comp);
      WriteReconstruction(cluster_recon,
                          reconstruction_path + "/" + std::to_string(comp),
                          output_format,
                          image_path);
    }
    LOG(INFO) << "Exported " << max_cluster_id + 1 << " reconstructions";
  }
}

}  // namespace glomap
