// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/mat.h"

#include <cstdint>
#include <vector>

namespace colmap {
namespace mvs {
class PatchMatchCuda;
namespace internal {

struct SweepTraceResult {
  Mat<int> draws;
  Mat<int> winners;
  Mat<int> support;
  // Four slices per hypothesis: depth followed by three normal components.
  Mat<float> hypotheses;
  Mat<float> hypothesis_costs;
  Mat<float> forward;
  Mat<float> initial_cost;
  Mat<float> cost;
  Mat<float> depth;
  Mat<float> normal;
  Mat<float> selection;
  Mat<uint32_t> continuation;
};

SweepTraceResult ProbeSweep(PatchMatchCuda& patch_match,
                            bool cooperative,
                            unsigned long long seed);

struct SweepSequenceResult {
  bool initialization_matches;
  std::vector<bool> continuation_matches;
  std::vector<SweepTraceResult> sweeps;
};

// Preserves production PHOTO/GEO initialization and rotates eight real sweeps.
SweepSequenceResult ProbeSweepSequence(PatchMatchCuda& patch_match,
                                       bool cooperative);

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
