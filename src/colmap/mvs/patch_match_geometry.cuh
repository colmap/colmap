// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cfloat>
#include <cmath>

namespace colmap {
namespace mvs {
namespace internal {

// Transfer the plane depth between rows of the same image column. Retain
// depth1 for nearly parallel rays or intersections without finite positive
// depth; that fallback need not lie on the original plane at row2.
__device__ inline float PropagateDepth(const float inv_K[4],
                                       const float depth1,
                                       const float normal1[3],
                                       const float row1,
                                       const float row2,
                                       const float col) {
  const float ray_x = inv_K[0] * col + inv_K[1];
  const float ray_y1 = inv_K[2] * row1 + inv_K[3];
  const float ray_y2 = inv_K[2] * row2 + inv_K[3];
  // Along one image column, x = ray_x * z. The plane's effective z
  // coefficient therefore includes the x component of its normal.
  const float normal_z = normal1[0] * ray_x + normal1[2];
  const float denom = normal1[1] * ray_y2 + normal_z;
  constexpr float kEps = 1e-5f;
  if (abs(denom) < kEps) {
    return depth1;
  }
  const float depth2 = depth1 * ((normal1[1] * ray_y1 + normal_z) / denom);
  // A plane horizon can put the adjacent intersection behind the camera.
  // Retain the valid previous hypothesis when no finite forward depth exists.
  return depth2 > 0.0f && depth2 <= FLT_MAX ? depth2 : depth1;
}

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
