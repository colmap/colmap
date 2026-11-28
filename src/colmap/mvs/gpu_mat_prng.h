// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/gpu_mat.h"

#include <cstdint>

namespace colmap {
namespace mvs {
namespace internal {
// Original full-image block-major seed, independent of caller launch geometry.
// Preconditions: positive width and valid original-image row/column.
__host__ __device__ inline size_t GpuMatPRNGSeed(size_t row,
                                                 size_t col,
                                                 size_t width) {
  const size_t blocks_x = (width - 1) / kGpuMatBlockDimX + 1;
  // CUDA's original block-index arithmetic wraps at 32 bits before widening.
  const uint32_t block_index = static_cast<uint32_t>(
      (row / kGpuMatBlockDimY) * blocks_x + col / kGpuMatBlockDimX);
  return static_cast<size_t>(block_index) * kGpuMatBlockDimX *
             kGpuMatBlockDimY +
         (row % kGpuMatBlockDimY) * kGpuMatBlockDimX + col % kGpuMatBlockDimX;
}
}  // namespace internal

class GpuMatPRNG : public GpuMat<curandState> {
 public:
  GpuMatPRNG(const int width, const int height);

 private:
  void InitRandomState();
};

}  // namespace mvs
}  // namespace colmap
