// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cfloat>
#include <cstdint>

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

#if defined(__CUDACC__) || defined(__HIPCC__)
namespace internal {

template <int kSampleWindowSize, int kBlockWidth>
__device__ __forceinline__ int AppendSampledSource(
    const int source,
    const bool use_source_mask,
    uint32_t& sampled_sources,
    int& num_unique_sources,
    int (&sources)[kSampleWindowSize][kBlockWidth],
    const int lane) {
  int job_slot = -1;
  if (use_source_mask) {
    job_slot = source;
    if (job_slot >= 0)
      sampled_sources |= uint32_t{1} << static_cast<uint32_t>(job_slot);
  } else if (source >= 0) {
    job_slot = 0;
    while (job_slot < num_unique_sources && sources[job_slot][lane] != source) {
      ++job_slot;
    }
    if (job_slot == num_unique_sources) {
      sources[job_slot][lane] = source;
      ++num_unique_sources;
    }
  }
  return job_slot;
}

template <int kSampleWindowSize, int kBlockWidth>
__device__ __forceinline__ void PackSampledSources(
    const bool use_source_mask,
    const uint32_t sampled_sources,
    int num_unique_sources,
    const int sample_count,
    int (&sources)[kSampleWindowSize][kBlockWidth],
    int (&aliases)[kSampleWindowSize][kBlockWidth],
    const int lane) {
  if (use_source_mask) {
    uint32_t remaining_sources = sampled_sources;
    while (remaining_sources != 0) {
      const int source = __ffs(remaining_sources) - 1;
      sources[num_unique_sources++][lane] = source;
      remaining_sources &= remaining_sources - 1;
    }
    for (int slot = 0; slot < sample_count; ++slot) {
      const int source = aliases[slot][lane];
      if (source >= 0) {
        aliases[slot][lane] =
            __popc(sampled_sources &
                   ((uint32_t{1} << static_cast<uint32_t>(source)) - 1));
      }
    }
  }
  for (int slot = num_unique_sources; slot < kSampleWindowSize; ++slot) {
    sources[slot][lane] = -1;
  }
}

}  // namespace internal
#endif

}  // namespace colmap::mvs

#undef COLMAP_SAMPLING_HD
