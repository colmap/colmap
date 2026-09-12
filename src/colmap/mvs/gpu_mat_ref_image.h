// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/cuda_texture.h"
#include "colmap/mvs/gpu_mat.h"

#include <memory>

namespace colmap {
namespace mvs {

class GpuMatRefImage {
 public:
  GpuMatRefImage(const size_t width, const size_t height);

  // Filter image using sum convolution kernel to compute local sum of
  // intensities. The filtered images can then be used for repeated, efficient
  // NCC computation.
  void Filter(const uint8_t* image_data,
              const size_t window_radius,
              const size_t window_step,
              const float sigma_spatial,
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

  const size_t width_;
  const size_t height_;
};

struct BilateralWeightComputer {
  __device__ BilateralWeightComputer(const float sigma_spatial,
                                     const float sigma_color)
      : spatial_normalization_(1.0f / (2.0f * sigma_spatial * sigma_spatial)),
        color_normalization_(1.0f / (2.0f * sigma_color * sigma_color)) {}

  __device__ inline float Compute(const float row_diff,
                                  const float col_diff,
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
