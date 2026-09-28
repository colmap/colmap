// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/optim/sampler.h"

#include <limits>
#include <vector>

namespace colmap {

// Majority-group sampler for RANSAC-based methods.
//
// With probability `structured_prob`, draws a structured sample of
// `majority_size` indices from one group plus the remaining indices from a
// different group; otherwise draws a uniform random sample. The majority
// group is chosen proportional to its size among groups with at least
// `majority_size` members, and the minority group uniformly among the
// other groups with enough members. This biases minimal samples toward
// structured configurations (e.g. 5+1 camera-pair samples for generalized
// relative pose) while retaining uniform samples for robustness when
// inliers are spread across groups.
//
// Groups are provided as one integer label per correspondence. Without
// groups, or whenever a structured sample is infeasible (fewer than two
// non-empty groups or no group with enough members), the sampler degrades
// to uniform random sampling.
//
// Note that structured sampling voids the uniformity assumption of the
// standard RANSAC stopping criterion: required trial counts are computed
// as if sampling were uniform, so the confidence guarantee is only
// approximate. Keep `structured_prob` well below 1 to retain a
// substantial fraction of uniform draws.
//
// Note that a separate sampler should be instantiated per thread.
// RANSAC-based methods copy the provided sampler instance per thread, so
// the group labels must be set before estimation.
class MajorityGroupSampler : public Sampler {
 public:
  // `majority_size` must be in [1, num_samples).
  explicit MajorityGroupSampler(size_t num_samples,
                                std::vector<int> group_ids,
                                double structured_prob,
                                size_t majority_size);

  void Initialize(size_t total_num_samples) override;

  size_t MaxNumSamples() override;

  void Sample(std::vector<size_t>* sampled_idxs) override;

 private:
  static constexpr size_t kInvalidMinorPos = std::numeric_limits<size_t>::max();
  const size_t num_samples_;
  const std::vector<int> group_ids_;
  const double structured_prob_;
  const size_t majority_size_;
  const size_t minority_size_;

  std::vector<size_t> sample_idxs_;
  std::vector<std::vector<size_t>> group_members_;
  std::vector<size_t> eligible_major_groups_;
  size_t eligible_major_total_size_ = 0;
  // Groups with at least minority_size_ members, plus for each group its
  // position in that list (kInvalidMinorPos if ineligible).
  std::vector<size_t> eligible_minor_groups_;
  std::vector<size_t> minor_pos_;
};

template <>
struct is_randomized_sampler<MajorityGroupSampler> : std::true_type {};

template <>
struct is_parallel_safe_sampler<MajorityGroupSampler> : std::true_type {};

}  // namespace colmap
