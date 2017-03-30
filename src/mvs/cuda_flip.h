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

#ifndef COLMAP_SRC_MVS_CUDA_FLIP_H_
#define COLMAP_SRC_MVS_CUDA_FLIP_H_

#include <cuda_runtime.h>

namespace colmap {
namespace mvs {

// Flip the input matrix horizontally.
template <typename T>
void CudaFlipHorizontal(const T* input, T* output, const int width,
                        const int height, const int pitch_input,
                        const int pitch_output);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

// TILE_DIM_FLIP must divide by BLOCK_ROWS. Do not change these values.
#define TILE_DIM_FLIP 32
#define BLOCK_ROWS_FLIP 8

namespace internal {

template <typename T>
__global__ void CudaFlipHorizontalKernel(T* output_data, const T* input_data,
                                         const int width, const int height,
                                         const int input_pitch,
                                         const int output_pitch) {
  int x_index = blockIdx.x * TILE_DIM_FLIP + threadIdx.x;
  const int y_index = blockIdx.y * TILE_DIM_FLIP + threadIdx.y;

  __shared__ T tile[TILE_DIM_FLIP][TILE_DIM_FLIP + 1];
  const int tile_x = min(threadIdx.x, width - 1 - blockIdx.x * TILE_DIM_FLIP);
  const int tile_y = min(threadIdx.y, height - 1 - blockIdx.y * TILE_DIM_FLIP);

  for (int i = 0; i < TILE_DIM_FLIP; i += BLOCK_ROWS_FLIP) {
    const int x = min(x_index, width - 1);
    const int y = min(y_index, height - i - 1);
    const int index = y * input_pitch + x + i * input_pitch;
    tile[tile_y + i][tile_x] = input_data[index];
  }

  __syncthreads();

  x_index = width - 1 - (blockIdx.x * TILE_DIM_FLIP + threadIdx.x);
  if (x_index < width) {
    const int index = x_index + y_index * output_pitch;
    for (int i = 0; i < TILE_DIM_FLIP; i += BLOCK_ROWS_FLIP) {
      if (y_index + i < height) {
        output_data[index + i * output_pitch] =
            tile[threadIdx.y + i][threadIdx.x];
      }
    }
  }
}

}  // namespace internal

template <typename T>
void CudaFlipHorizontal(const T* input, T* output, const int width,
                        const int height, const int pitch_input,
                        const int pitch_output) {
  dim3 block_dim(TILE_DIM_FLIP, BLOCK_ROWS_FLIP, 1);
  dim3 grid_dim;
  grid_dim.x = (width - 1) / TILE_DIM_FLIP + 1;
  grid_dim.y = (height - 1) / TILE_DIM_FLIP + 1;

  internal::CudaFlipHorizontalKernel<<<grid_dim, block_dim>>>(
      output, input, width, height, pitch_input / sizeof(T),
      pitch_output / sizeof(T));
}

#undef TILE_DIM_FLIP
#undef BLOCK_ROWS_FLIP

#endif  // __CUDACC__

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_CUDA_FLIP_H_
