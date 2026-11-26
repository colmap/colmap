// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/optim/sampler.h"

namespace colmap {

// Exhaustive sampler that generates combinations in lexicographic order.
//
// Note that a separate sampler should be instantiated per thread and it assumes
// that the input data is shuffled in advance.
class CombinationSampler : public Sampler {
 public:
  explicit CombinationSampler(size_t num_samples);

  void Initialize(size_t total_num_samples) override;

  size_t MaxNumSamples() override;

  void Sample(std::vector<size_t>* sampled_idxs) override;

 private:
  const size_t num_samples_;
  size_t total_num_samples_ = 0;
  std::vector<size_t> sample_idxs_;
};

template <>
struct is_randomized_sampler<CombinationSampler> : std::false_type {};

}  // namespace colmap
