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

#include "mvs/gpu_mat_ref_image.h"

#include <iostream>

#include "util/cudacc.h"

namespace colmap {
namespace mvs {
namespace {

texture<uint8_t, cudaTextureType2D, cudaReadModeNormalizedFloat> image_texture;

template <int kWindowRadius, int kWindowStep>
__global__ void FilterKernel(GpuMat<uint8_t> image, GpuMat<float> sum_image,
                             GpuMat<float> squared_sum_image,
                             const float sigma_spatial,
                             const float sigma_color) {
  const size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t col = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= image.GetHeight() || col >= image.GetWidth()) {
    return;
  }

  const float center_color = tex2D(image_texture, col, row);

  float sum = 0.0f;
  float squared_sum = 0.0f;
  float bilateral_weight_sum = 0.0f;
  for (int win_col = -kWindowRadius; win_col <= kWindowRadius;
       win_col += kWindowStep) {
    float sum_col = 0.0f;
    float squared_sum_col = 0.0f;
    float bilateral_weight_sum_col = 0.0f;
    for (int win_row = -kWindowRadius; win_row <= kWindowRadius;
         win_row += kWindowStep) {
      const float color = tex2D(image_texture, col + win_col, row + win_row);
      const float bilateral_weight =
          ComputeBilateralWeight(0.0f, 0.0f, win_col, win_row, center_color,
                                 color, sigma_spatial, sigma_color);
      sum_col += bilateral_weight * color;
      squared_sum_col += bilateral_weight * color * color;
      bilateral_weight_sum_col += bilateral_weight;
    }
    sum += sum_col;
    squared_sum += squared_sum_col;
    bilateral_weight_sum += bilateral_weight_sum_col;
  }

  sum /= bilateral_weight_sum;
  squared_sum /= bilateral_weight_sum;

  image.Set(row, col, static_cast<uint8_t>(255.0f * center_color));
  sum_image.Set(row, col, sum);
  squared_sum_image.Set(row, col, squared_sum);
}

}  // namespace

GpuMatRefImage::GpuMatRefImage(const size_t width, const size_t height)
    : height_(height), width_(width) {
  image.reset(new GpuMat<uint8_t>(width, height));
  sum_image.reset(new GpuMat<float>(width, height));
  squared_sum_image.reset(new GpuMat<float>(width, height));
}

void GpuMatRefImage::Filter(const uint8_t* image_data,
                            const size_t window_radius,
                            const size_t window_step, const float sigma_spatial,
                            const float sigma_color) {
#define SWITCH_WINDOW_RADIUS(window_radius, window_step)          \
  case window_radius:                                             \
    Filter<window_radius, window_step>(image_data, sigma_spatial, \
                                       sigma_color);              \
    break;

#define CASE_WINDOW_STEP(window_step)                                 \
  case window_step:                                                   \
    switch (window_radius) {                                          \
      SWITCH_WINDOW_RADIUS(1, window_step)                            \
      SWITCH_WINDOW_RADIUS(2, window_step)                            \
      SWITCH_WINDOW_RADIUS(3, window_step)                            \
      SWITCH_WINDOW_RADIUS(4, window_step)                            \
      SWITCH_WINDOW_RADIUS(5, window_step)                            \
      SWITCH_WINDOW_RADIUS(6, window_step)                            \
      SWITCH_WINDOW_RADIUS(7, window_step)                            \
      SWITCH_WINDOW_RADIUS(8, window_step)                            \
      SWITCH_WINDOW_RADIUS(9, window_step)                            \
      SWITCH_WINDOW_RADIUS(10, window_step)                           \
      SWITCH_WINDOW_RADIUS(11, window_step)                           \
      SWITCH_WINDOW_RADIUS(12, window_step)                           \
      SWITCH_WINDOW_RADIUS(13, window_step)                           \
      SWITCH_WINDOW_RADIUS(14, window_step)                           \
      SWITCH_WINDOW_RADIUS(15, window_step)                           \
      SWITCH_WINDOW_RADIUS(16, window_step)                           \
      SWITCH_WINDOW_RADIUS(17, window_step)                           \
      SWITCH_WINDOW_RADIUS(18, window_step)                           \
      SWITCH_WINDOW_RADIUS(19, window_step)                           \
      SWITCH_WINDOW_RADIUS(20, window_step)                           \
      default: {                                                      \
        std::cerr << "Error: Window size not supported" << std::endl; \
        break;                                                        \
      }                                                               \
    }                                                                 \
    break;

  switch (window_step) {
    CASE_WINDOW_STEP(1)
    CASE_WINDOW_STEP(2)
    default: {
      std::cerr << "Error: Window step not supported" << std::endl;
      break;
    }
  }

#undef SWITCH_WINDOW_RADIUS
#undef SWITCH_WINDOW_RADIUS
}

template <int kWindowRadius, int kWindowStep>
void GpuMatRefImage::Filter(const uint8_t* image_data,
                            const float sigma_spatial,
                            const float sigma_color) {
  CudaArrayWrapper<uint8_t> image_array(width_, height_, 1);
  image_array.CopyToDevice(image_data);
  image_texture.addressMode[0] = cudaAddressModeBorder;
  image_texture.addressMode[1] = cudaAddressModeBorder;
  image_texture.addressMode[2] = cudaAddressModeBorder;
  image_texture.filterMode = cudaFilterModePoint;
  image_texture.normalized = false;

  const dim3 block_size(kBlockDimX, kBlockDimY);
  const dim3 grid_size((width_ - 1) / block_size.x + 1,
                       (height_ - 1) / block_size.y + 1);

  CUDA_SAFE_CALL(cudaBindTextureToArray(image_texture, image_array.GetPtr()));
  FilterKernel<kWindowRadius, kWindowStep><<<grid_size, block_size>>>(
      *image, *sum_image, *squared_sum_image, sigma_spatial, sigma_color);
  CUDA_SYNC_AND_CHECK();
  CUDA_SAFE_CALL(cudaUnbindTexture(image_texture));
}

}  // namespace mvs
}  // namespace colmap
