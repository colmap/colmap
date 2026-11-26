// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/gpu_mat_prng.h"

namespace colmap {
namespace mvs {
namespace {

__global__ void InitRandomStateKernel(GpuMatView<curandState> output) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  // Preserve each original pixel seed, with sequence zero and offset zero.
  if (col < output.GetWidth() && row < output.GetHeight()) {
    curand_init(internal::GpuMatPRNGSeed(row, col, output.width),
                0,
                0,
                &output.GetRef(row, col));
  }
}

}  // namespace

GpuMatPRNG::GpuMatPRNG(const int width, const int height)
    : GpuMat(width, height) {
  InitRandomStateKernel<<<gridSize_, blockSize_>>>(View());
}

}  // namespace mvs
}  // namespace colmap
