// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#ifndef COLMAP_SRC_MVS_CUDA_TRANSPOSE_H_
#define COLMAP_SRC_MVS_CUDA_TRANSPOSE_H_

#include <cuda_runtime.h>

namespace colmap {
namespace mvs {

// Transpose the input matrix.
template <typename T>
void CudaTranspose(const T* input, T* output, const int width, const int height,
                   const int pitch_input, const int pitch_output);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

// TILE_DIM_TRANSPOSE must divide by BLOCK_ROWS. Do not change these values.
#define TILE_DIM_TRANSPOSE 32
#define BLOCK_ROWS_TRANSPOSE 8

namespace internal {

template <typename T>
__global__ void CudaTransposeKernel(T* output_data, const T* input_data,
                                    const int width, const int height,
                                    const int input_pitch,
                                    const int output_pitch) {
  int x_index = blockIdx.x * TILE_DIM_TRANSPOSE + threadIdx.x;
  int y_index = blockIdx.y * TILE_DIM_TRANSPOSE + threadIdx.y;

  __shared__ T tile[TILE_DIM_TRANSPOSE][TILE_DIM_TRANSPOSE + 1];
  const int tile_x =
      min(threadIdx.x, width - 1 - blockIdx.x * TILE_DIM_TRANSPOSE);
  const int tile_y =
      min(threadIdx.y, height - 1 - blockIdx.y * TILE_DIM_TRANSPOSE);

  for (int i = 0; i < TILE_DIM_TRANSPOSE; i += BLOCK_ROWS_TRANSPOSE) {
    const int x = min(x_index, width - 1);
    const int y = min(y_index, height - i - 1);
    tile[tile_y + i][tile_x] =
        *((T*)((char*)input_data + y * input_pitch + i * input_pitch) + x);
  }

  __syncthreads();

  x_index = blockIdx.y * TILE_DIM_TRANSPOSE + threadIdx.x;
  if (x_index < height) {
    y_index = blockIdx.x * TILE_DIM_TRANSPOSE + threadIdx.y;
    for (int i = 0; i < TILE_DIM_TRANSPOSE; i += BLOCK_ROWS_TRANSPOSE) {
      if (y_index + i < width) {
        *((T*)((char*)output_data + y_index * output_pitch + i * output_pitch) +
          x_index) = tile[threadIdx.x][threadIdx.y + i];
      }
    }
  }
}

}  // namespace internal

template <typename T>
void CudaTranspose(const T* input, T* output, const int width, const int height,
                   const int pitch_input, const int pitch_output) {
  dim3 block_dim(TILE_DIM_TRANSPOSE, BLOCK_ROWS_TRANSPOSE, 1);
  dim3 grid_dim;
  grid_dim.x = (width - 1) / TILE_DIM_TRANSPOSE + 1;
  grid_dim.y = (height - 1) / TILE_DIM_TRANSPOSE + 1;

  internal::CudaTransposeKernel<<<grid_dim, block_dim>>>(
      output, input, width, height, pitch_input, pitch_output);
}

#undef TILE_DIM_TRANSPOSE
#undef BLOCK_ROWS_TRANSPOSE

#endif  // __CUDACC__

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_CUDA_TRANSPOSE_H_
