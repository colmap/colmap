// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cfloat>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define COLMAP_SAMPLING_HD __host__ __device__
#else
#define COLMAP_SAMPLING_HD
#endif

namespace colmap::mvs {

// Return false only for an all-zero PDF. Empty support stays finite and does
// not invent a uniform distribution. Other inputs retain the normalization
// arithmetic; this is not validation or a new nonfinite-input policy.
COLMAP_SAMPLING_HD inline bool TransformPDFToCDF(float* probs,
                                                 int num_probs,
                                                 int source_stride) {
  float prob_sum = 0.0f;
  bool all_zero = true;
  for (int i = 0; i < num_probs; ++i) {
    prob_sum += probs[i * source_stride];
    all_zero &= probs[i * source_stride] == 0.0f;
  }
  if (all_zero) return false;
  const float inv_prob_sum = 1.0f / prob_sum;
  float cum_prob = 0.0f;
  for (int i = 0; i < num_probs; ++i) {
    const float prob = probs[i * source_stride] * inv_prob_sum;
    cum_prob += prob;
    probs[i * source_stride] = cum_prob;
  }
  return true;
}

// uniform is the caller's single CURAND draw in (0, 1]. Preserve the existing
// upper-endpoint adjustment without selecting a zero-mass prefix near zero.
COLMAP_SAMPLING_HD inline int SampleSource(const float* cdf,
                                           int num_sources,
                                           int source_stride,
                                           float uniform) {
  const float shifted = uniform - FLT_EPSILON;
  const float threshold = shifted > 0.0f ? shifted : 0.0f;
  for (int i = 0; i < num_sources; ++i) {
    if (cdf[i * source_stride] > threshold) return i;
  }
  return -1;
}

}  // namespace colmap::mvs

#undef COLMAP_SAMPLING_HD
