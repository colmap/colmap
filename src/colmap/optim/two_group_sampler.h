// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/optim/sampler.h"

#include <limits>
#include <vector>

namespace colmap {

// Two-group sampler for RANSAC-based methods.
//
// With probability `structured_prob`, draws a structured sample of
// `num_first_group_samples` indices from one group plus the remaining
// indices from a different group; otherwise draws a uniform random sample.
// This biases minimal samples toward structured configurations (e.g. 5+1
// camera-pair samples for generalized relative pose) while retaining uniform
// samples for robustness when inliers are spread across groups.
//
// The first group is chosen proportional to its size among groups with at
// least `num_first_group_samples` members, and the second group uniformly
// among the other groups with enough members. The uniform second-group choice
// is deliberate: it spreads the second part of the sample across groups
// instead of concentrating it in the largest ones, which performed on par or
// slightly better than a size-weighted choice for generalized relative pose.
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
class TwoGroupSampler : public Sampler {
 public:
  // `num_first_group_samples` must be in [1, num_samples).
  explicit TwoGroupSampler(size_t num_samples,
                           std::vector<int> group_ids,
                           double structured_prob,
                           size_t num_first_group_samples);

  void Initialize(size_t total_num_samples) override;

  size_t MaxNumSamples() override;

  void Sample(std::vector<size_t>* sampled_idxs) override;

 private:
  static constexpr size_t kInvalidGroupPos = std::numeric_limits<size_t>::max();
  const size_t num_samples_;
  const std::vector<int> group_ids_;
  const double structured_prob_;
  const size_t num_first_group_samples_;
  const size_t num_second_group_samples_;

  std::vector<size_t> sample_idxs_;
  std::vector<std::vector<size_t>> group_members_;
  // Groups with enough members for the first part of a structured sample and
  // their total number of members.
  std::vector<size_t> eligible_first_groups_;
  size_t eligible_first_total_size_ = 0;
  // Groups with enough members for the second part of a structured sample,
  // plus for each group its position in that list (kInvalidGroupPos if
  // ineligible).
  std::vector<size_t> eligible_second_groups_;
  std::vector<size_t> second_group_pos_;
};

template <>
struct is_randomized_sampler<TwoGroupSampler> : std::true_type {};

template <>
struct is_parallel_safe_sampler<TwoGroupSampler> : std::true_type {};

}  // namespace colmap
