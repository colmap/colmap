// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/cuda_to_hip.h"

namespace colmap {
namespace mvs {

// Flip the input matrix horizontally.
template <typename T>
void CudaFlipHorizontal(const T* input,
                        T* output,
                        const int width,
                        const int height,
                        const int pitch_input,
                        const int pitch_output);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDACC__) || defined(__HIPCC__)

// TILE_DIM_FLIP must divide by BLOCK_ROWS. Do not change these values.
#define TILE_DIM_FLIP 32
#define BLOCK_ROWS_FLIP 8

namespace internal {

template <typename T>
__global__ void CudaFlipHorizontalKernel(T* output_data,
                                         const T* input_data,
                                         const int width,
                                         const int height,
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
    tile[tile_y + i][tile_x] = *(
        (T*)((char*)input_data + static_cast<size_t>(y + i) * input_pitch) + x);
  }

  __syncthreads();

  x_index = width - 1 - (blockIdx.x * TILE_DIM_FLIP + threadIdx.x);
  if (x_index < width) {
    for (int i = 0; i < TILE_DIM_FLIP; i += BLOCK_ROWS_FLIP) {
      if (y_index + i < height) {
        *((T*)((char*)output_data +
               static_cast<size_t>(y_index + i) * output_pitch) +
          x_index) = tile[threadIdx.y + i][threadIdx.x];
      }
    }
  }
}

}  // namespace internal

template <typename T>
void CudaFlipHorizontal(const T* input,
                        T* output,
                        const int width,
                        const int height,
                        const int pitch_input,
                        const int pitch_output) {
  dim3 block_dim(TILE_DIM_FLIP, BLOCK_ROWS_FLIP, 1);
  dim3 grid_dim;
  grid_dim.x = (width - 1) / TILE_DIM_FLIP + 1;
  grid_dim.y = (height - 1) / TILE_DIM_FLIP + 1;

  internal::CudaFlipHorizontalKernel<<<grid_dim, block_dim>>>(
      output, input, width, height, pitch_input, pitch_output);
}

#undef TILE_DIM_FLIP
#undef BLOCK_ROWS_FLIP

#endif  // defined(__CUDACC__) || defined(__HIPCC__)

}  // namespace mvs
}  // namespace colmap
