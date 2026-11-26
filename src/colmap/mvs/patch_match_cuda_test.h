// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/mat.h"

namespace colmap {
namespace mvs {
class PatchMatchCuda;
namespace internal {

struct SweepTraceResult {
  Mat<int> winners;
  Mat<int> support;
  // Four slices each for the incumbent and propagated candidate plane.
  Mat<float> hypotheses;
};

SweepTraceResult ProbeSweep(PatchMatchCuda& patch_match,
                            unsigned long long seed);

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
