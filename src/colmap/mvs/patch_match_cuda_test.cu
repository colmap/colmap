// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/gpu_mat.h"
#include "colmap/mvs/patch_match_geometry.cuh"
#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <array>

namespace colmap {
namespace mvs {
namespace {
__global__ void ProbePropagatedDepthKernel(const float4 inv_K,
                                           const float depth,
                                           const float3 normal,
                                           const float row1,
                                           const float row2,
                                           const float col,
                                           float* output) {
  const float n[3] = {normal.x, normal.y, normal.z};
  const float calibration[4] = {inv_K.x, inv_K.y, inv_K.z, inv_K.w};
  *output = internal::PropagateDepth(calibration, depth, n, row1, row2, col);
}

}  // namespace
namespace internal {
float ProbePropagatedDepth(const std::array<float, 4>& K,
                           const float depth,
                           const std::array<float, 3>& normal,
                           const float row1,
                           const float row2,
                           const float col) {
  const float inv_K[4] = {1 / K[0], -K[1] / K[0], 1 / K[2], -K[3] / K[2]};
  GpuMat<float> output(1, 1);
  ProbePropagatedDepthKernel<<<1, 1>>>(
      make_float4(inv_K[0], inv_K[1], inv_K[2], inv_K[3]),
      depth,
      make_float3(normal[0], normal[1], normal[2]),
      row1,
      row2,
      col,
      output.GetPtr());
  CUDA_SYNC_AND_CHECK();
  return output.CopyToMat().GetData()[0];
}

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
