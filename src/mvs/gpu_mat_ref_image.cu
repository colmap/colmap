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

template <int kWindowRadius>
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
  for (int win_col = -kWindowRadius; win_col <= kWindowRadius; win_col++) {
    float sum_row = 0.0f;
    float squared_sum_row = 0.0f;
    float bilateral_weight_sum_row = 0.0f;
    for (int win_row = -kWindowRadius; win_row <= kWindowRadius; win_row++) {
      const float color = tex2D(image_texture, col + win_col, row + win_row);
      const float bilateral_weight =
          ComputeBilateralWeight(0.0f, 0.0f, win_col, win_row, center_color,
                                 color, sigma_spatial, sigma_color);
      sum_row += bilateral_weight * color;
      squared_sum_row += bilateral_weight * color * color;
      bilateral_weight_sum_row += bilateral_weight;
    }
    sum += sum_row;
    squared_sum += squared_sum_row;
    bilateral_weight_sum += bilateral_weight_sum_row;
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
                            const float sigma_spatial,
                            const float sigma_color) {
#define CALL_FILTER_FUNC(window_radius)                            \
  case window_radius:                                              \
    Filter<window_radius>(image_data, sigma_spatial, sigma_color); \
    break;

  switch (window_radius) {
    CALL_FILTER_FUNC(1)
    CALL_FILTER_FUNC(2)
    CALL_FILTER_FUNC(3)
    CALL_FILTER_FUNC(4)
    CALL_FILTER_FUNC(5)
    CALL_FILTER_FUNC(6)
    CALL_FILTER_FUNC(7)
    CALL_FILTER_FUNC(8)
    CALL_FILTER_FUNC(9)
    CALL_FILTER_FUNC(10)
    CALL_FILTER_FUNC(11)
    CALL_FILTER_FUNC(12)
    CALL_FILTER_FUNC(13)
    CALL_FILTER_FUNC(14)
    CALL_FILTER_FUNC(15)
    CALL_FILTER_FUNC(16)
    CALL_FILTER_FUNC(17)
    CALL_FILTER_FUNC(18)
    CALL_FILTER_FUNC(19)
    CALL_FILTER_FUNC(20)
    CALL_FILTER_FUNC(21)
    CALL_FILTER_FUNC(22)
    CALL_FILTER_FUNC(23)
    CALL_FILTER_FUNC(24)
    CALL_FILTER_FUNC(25)
    CALL_FILTER_FUNC(26)
    CALL_FILTER_FUNC(27)
    CALL_FILTER_FUNC(28)
    CALL_FILTER_FUNC(29)
    CALL_FILTER_FUNC(30)
    default:
      std::cerr << "Error: Window size not supported" << std::endl;
      exit(EXIT_FAILURE);
      break;
  }

#undef CALL_FILTER_FUNC
}

template <int kWindowRadius>
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
  FilterKernel<kWindowRadius><<<grid_size, block_size>>>(
      *image, *sum_image, *squared_sum_image, sigma_spatial, sigma_color);
  CUDA_CHECK_ERROR();
  CUDA_SAFE_CALL(cudaUnbindTexture(image_texture));
}

}  // namespace mvs
}  // namespace colmap
