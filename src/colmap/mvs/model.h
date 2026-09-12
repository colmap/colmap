// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/image.h"
#include "colmap/util/hash_containers.h"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace colmap {
namespace mvs {

// Simple sparse model class.
struct Model {
  struct Point {
    float x = 0;
    float y = 0;
    float z = 0;
    std::vector<int> track;
  };

  // Read the model from different data formats.
  void Read(const std::filesystem::path& path, const std::string& format);
  void ReadFromCOLMAP(const std::filesystem::path& path,
                      const std::filesystem::path& sparse_path = "sparse",
                      const std::filesystem::path& images_path = "images");
  void ReadFromPMVS(const std::filesystem::path& path);

  // Get the image index for the given image name.
  int GetImageIdx(const std::string& name) const;
  const std::string& GetImageName(int image_idx) const;

  // For each image, determine the maximally overlapping images, sorted based on
  // the number of shared points subject to a minimum robust average
  // triangulation angle of the points.
  std::vector<std::vector<int>> GetMaxOverlappingImages(
      size_t num_images, double min_triangulation_angle) const;

  // Get the overlapping images defined in the vis.dat file.
  const std::vector<std::vector<int>>& GetMaxOverlappingImagesFromPMVS() const;

  // Compute the robust minimum and maximum depths from the sparse point cloud.
  std::vector<std::pair<float, float>> ComputeDepthRanges() const;

  // Compute the number of shared points between all overlapping images.
  std::vector<std::map<int, int>> ComputeSharedPoints() const;

  // Compute the median triangulation angles between all overlapping images.
  std::vector<std::map<int, float>> ComputeTriangulationAngles(
      float percentile = 50) const;

  // Note that in case the data is read from a COLMAP reconstruction, the index
  // of an image or point does not correspond to its original identifier in the
  // reconstruction, but it corresponds to the position in the
  // images.bin/points3D.bin files. This is mainly done for more efficient
  // access to the data, which is required during the stereo fusion stage.
  std::vector<Image> images;
  std::vector<Point> points;

 private:
  bool ReadFromBundlerPMVS(const std::filesystem::path& path);
  bool ReadFromRawPMVS(const std::filesystem::path& path);

  std::vector<std::string> image_names_;
  NodeHashMap<std::string, int> image_name_to_idx_;

  std::vector<std::vector<int>> pmvs_vis_dat_;
};

}  // namespace mvs
}  // namespace colmap
