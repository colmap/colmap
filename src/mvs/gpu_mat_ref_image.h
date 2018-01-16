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

#ifndef COLMAP_SRC_MVS_GPU_MAT_REF_IMAGE_H_
#define COLMAP_SRC_MVS_GPU_MAT_REF_IMAGE_H_

#include <memory>

#include "mvs/cuda_array_wrapper.h"
#include "mvs/gpu_mat.h"

namespace colmap {
namespace mvs {

class GpuMatRefImage {
 public:
  GpuMatRefImage(const size_t width, const size_t height);

  // Filter image using sum convolution kernel to compute local sum of
  // intensities. The filtered images can then be used for repeated, efficient
  // NCC computation.
  void Filter(const uint8_t* image_data, const size_t window_radius,
              const size_t window_step, const float sigma_spatial,
              const float sigma_color);

  // Image intensities.
  std::unique_ptr<GpuMat<uint8_t>> image;

  // Local sum of image intensities.
  std::unique_ptr<GpuMat<float>> sum_image;

  // Local sum of squared image intensities.
  std::unique_ptr<GpuMat<float>> squared_sum_image;

 private:
  const static size_t kBlockDimX = 16;
  const static size_t kBlockDimY = 12;

  size_t width_;
  size_t height_;
};

struct BilateralWeightComputer {
  __device__ BilateralWeightComputer(const float sigma_spatial,
                                     const float sigma_color)
      : spatial_normalization_(1.0f / (2.0f * sigma_spatial * sigma_spatial)),
        color_normalization_(1.0f / (2.0f * sigma_color * sigma_color)) {}

  __device__ inline float Compute(const float row_diff, const float col_diff,
                                  const float color1,
                                  const float color2) const {
    const float spatial_dist_squared =
        row_diff * row_diff + col_diff * col_diff;
    const float color_dist = color1 - color2;
    return exp(-spatial_dist_squared * spatial_normalization_ -
               color_dist * color_dist * color_normalization_);
  }

 private:
  const float spatial_normalization_;
  const float color_normalization_;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_GPU_MAT_REF_IMAGE_H_
