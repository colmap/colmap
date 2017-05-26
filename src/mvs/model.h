// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_MVS_MODEL_H_
#define COLMAP_SRC_MVS_MODEL_H_

#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "mvs/depth_map.h"
#include "mvs/image.h"
#include "mvs/normal_map.h"

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
  void Read(const std::string& path, const std::string& format);
  void ReadFromCOLMAP(const std::string& path);
  void ReadFromPMVS(const std::string& path);

  // Get the image identifier for the given image name.
  int GetImageId(const std::string& name) const;
  std::string GetImageName(const int image_id) const;

  // For each image, determine the maximally overlapping images, sorted based on
  // the number of shared points subject to a minimum robust average
  // triangulation angle of the points.
  std::vector<std::vector<int>> GetMaxOverlappingImages(
      const size_t num_images, const double min_triangulation_angle) const;

  // Compute the robust minimum and maximum depths from the sparse point cloud.
  std::vector<std::pair<float, float>> ComputeDepthRanges() const;

  // Compute the number of shared points between all overlapping images.
  std::vector<std::map<int, int>> ComputeSharedPoints() const;

  // Compute the median triangulation angles between all overlapping images.
  std::vector<std::map<int, float>> ComputeTriangulationAngles(
      const float percentile = 50) const;

  std::vector<Image> images;
  std::vector<Point> points;

 private:
  std::vector<std::string> image_names_;
  std::unordered_map<std::string, int> image_name_to_id_;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_MODEL_H_
