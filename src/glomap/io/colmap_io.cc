#include "glomap/io/colmap_io.h"

#include "colmap/util/file.h"
#include "colmap/util/misc.h"

#include "glomap/io/colmap_converter.h"

namespace glomap {

void WriteGlomapReconstruction(
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

  // Helper to write a single reconstruction
  auto write_reconstruction = [&](const colmap::Reconstruction& recon,
                                  const std::string& path) {
    colmap::Reconstruction recon_copy = recon;
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
  };

  // If no clusters, output as single reconstruction
  if (max_cluster_id == -1) {
    write_reconstruction(reconstruction, reconstruction_path + "/0");
  } else {
    // Export each cluster separately
    for (int comp = 0; comp <= max_cluster_id; comp++) {
      std::cout << "\r Exporting reconstruction " << comp + 1 << " / "
                << max_cluster_id + 1 << std::flush;

      colmap::Reconstruction cluster_recon =
          ExtractCluster(reconstruction, cluster_ids, comp);
      write_reconstruction(cluster_recon,
                           reconstruction_path + "/" + std::to_string(comp));
    }
    std::cout << '\n';
  }
}

void WriteColmapReconstruction(const std::string& reconstruction_path,
                               const colmap::Reconstruction& reconstruction,
                               const std::string& output_format) {
  colmap::CreateDirIfNotExists(reconstruction_path, true);
  if (output_format == "txt") {
    reconstruction.WriteText(reconstruction_path);
  } else if (output_format == "bin") {
    reconstruction.WriteBinary(reconstruction_path);
  } else {
    LOG(ERROR) << "Unsupported output type";
  }
}

}  // namespace glomap
