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

__global__ void FilterKernel(GpuMat<uint8_t> image, GpuMat<float> sum_image,
                             GpuMat<float> squared_sum_image,
                             const int window_radius, const int window_step,
                             const float sigma_spatial,
                             const float sigma_color) {
  const size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t col = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= image.GetHeight() || col >= image.GetWidth()) {
    return;
  }

  BilateralWeightComputer bilateral_weight_computer(sigma_spatial, sigma_color);

  const float center_color = tex2D(image_texture, col, row);

  float color_sum = 0.0f;
  float color_squared_sum = 0.0f;
  float bilateral_weight_sum = 0.0f;

  for (int window_row = -window_radius; window_row <= window_radius;
       window_row += window_step) {
    for (int window_col = -window_radius; window_col <= window_radius;
         window_col += window_step) {
      const float color =
          tex2D(image_texture, col + window_col, row + window_row);
      const float bilateral_weight = bilateral_weight_computer.Compute(
          window_row, window_col, center_color, color);
      color_sum += bilateral_weight * color;
      color_squared_sum += bilateral_weight * color * color;
      bilateral_weight_sum += bilateral_weight;
    }
  }

  color_sum /= bilateral_weight_sum;
  color_squared_sum /= bilateral_weight_sum;

  image.Set(row, col, static_cast<uint8_t>(255.0f * center_color));
  sum_image.Set(row, col, color_sum);
  squared_sum_image.Set(row, col, color_squared_sum);
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
  FilterKernel<<<grid_size, block_size>>>(
      *image, *sum_image, *squared_sum_image, window_radius, window_step,
      sigma_spatial, sigma_color);
  CUDA_SYNC_AND_CHECK();
  CUDA_SAFE_CALL(cudaUnbindTexture(image_texture));
}

}  // namespace mvs
}  // namespace colmap
