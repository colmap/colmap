// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#ifndef COLMAP_SRC_MVS_FUSION_H_
#define COLMAP_SRC_MVS_FUSION_H_

#include <vector>

#include <Eigen/Core>

#include "mvs/depth_map.h"
#include "mvs/image.h"
#include "mvs/mat.h"
#include "mvs/normal_map.h"
#include "util/math.h"

namespace colmap {
namespace mvs {

struct FusedPoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float nx = 0.0f;
  float ny = 0.0f;
  float nz = 0.0f;
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
};

struct FusionOptions {
  // Minimum number of fused pixels to produce a point.
  int min_num_pixels = 3;

  // Maximum number of pixels to fuse into a single point.
  int max_num_pixels = 1000;

  // Maximum depth in consistency graph traversal.
  int max_traversal_depth = 100;

  // Maximum relative difference between measured and projected pixel.
  float max_reproj_error = 2.0f;

  // Maximum relative difference between measured and projected depth.
  float max_depth_error = 0.01f;

  // Maximum difference between normals of pixels to be fused.
  float max_normal_error = 10.0f;

  // Check the options for validity.
  void Check() const;

  // Print the options to stdout.
  void Print() const;
};

// Fuse the multi-view stereo depth and normal maps into a consistent global
// dense point cloud with normal information.
std::vector<FusedPoint> StereoFusion(
    const FusionOptions& options,
    const std::vector<uint8_t>& used_image_mask,
    const std::vector<Image>& images, const std::vector<DepthMap>& depth_maps,
    const std::vector<NormalMap>& normal_maps,
    const std::vector<std::vector<int>>& consistency_graph);

// Write the point cloud to PLY file.
void WritePlyText(const std::string& path,
                  const std::vector<FusedPoint>& points);
void WritePlyBinary(const std::string& path,
                    const std::vector<FusedPoint>& points);

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_FUSION_H_
