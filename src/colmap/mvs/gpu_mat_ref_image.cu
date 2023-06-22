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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/mvs/gpu_mat_ref_image.h"
#include "colmap/util/cudacc.h"

#include <iostream>

namespace colmap {
namespace mvs {
namespace {

__global__ void FilterKernel(const cudaTextureObject_t image_texture,
                             GpuMat<uint8_t> image,
                             GpuMat<float> sum_image,
                             GpuMat<float> squared_sum_image,
                             const int window_radius,
                             const int window_step,
                             const float sigma_spatial,
                             const float sigma_color) {
  const size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t col = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= image.GetHeight() || col >= image.GetWidth()) {
    return;
  }

  BilateralWeightComputer bilateral_weight_computer(sigma_spatial, sigma_color);

  const float center_color = tex2D<float>(image_texture, col, row);

  float color_sum = 0.0f;
  float color_squared_sum = 0.0f;
  float bilateral_weight_sum = 0.0f;

  for (int window_row = -window_radius; window_row <= window_radius;
       window_row += window_step) {
    for (int window_col = -window_radius; window_col <= window_radius;
         window_col += window_step) {
      const float color =
          tex2D<float>(image_texture, col + window_col, row + window_row);
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
                            const size_t window_step,
                            const float sigma_spatial,
                            const float sigma_color) {
  cudaTextureDesc texture_desc;
  memset(&texture_desc, 0, sizeof(texture_desc));
  texture_desc.addressMode[0] = cudaAddressModeBorder;
  texture_desc.addressMode[1] = cudaAddressModeBorder;
  texture_desc.addressMode[2] = cudaAddressModeBorder;
  texture_desc.filterMode = cudaFilterModePoint;
  texture_desc.readMode = cudaReadModeNormalizedFloat;
  texture_desc.normalizedCoords = false;
  auto image_texture = CudaArrayLayeredTexture<uint8_t>::FromHostArray(
      texture_desc, width_, height_, 1, image_data);

  const dim3 block_size(kBlockDimX, kBlockDimY);
  const dim3 grid_size((width_ - 1) / block_size.x + 1,
                       (height_ - 1) / block_size.y + 1);

  FilterKernel<<<grid_size, block_size>>>(image_texture->GetObj(),
                                          *image,
                                          *sum_image,
                                          *squared_sum_image,
                                          window_radius,
                                          window_step,
                                          sigma_spatial,
                                          sigma_color);
  CUDA_SYNC_AND_CHECK();
}

}  // namespace mvs
}  // namespace colmap
