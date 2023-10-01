// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#pragma once

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/normal_map.h"

#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
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
  void Read(const std::string& path, const std::string& format);
  void ReadFromCOLMAP(const std::string& path,
                      const std::string& sparse_path = "sparse",
                      const std::string& images_path = "images");
  void ReadFromPMVS(const std::string& path);

  // Get the image index for the given image name.
  int GetImageIdx(const std::string& name) const;
  std::string GetImageName(int image_idx) const;

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
  bool ReadFromBundlerPMVS(const std::string& path);
  bool ReadFromRawPMVS(const std::string& path);

  std::vector<std::string> image_names_;
  std::unordered_map<std::string, int> image_name_to_idx_;

  std::vector<std::vector<int>> pmvs_vis_dat_;
};

}  // namespace mvs
}  // namespace colmap
