// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/combination_sampler.h"

#include "colmap/math/math.h"

#include <numeric>

namespace colmap {

CombinationSampler::CombinationSampler(const size_t num_samples)
    : num_samples_(num_samples) {}

void CombinationSampler::Initialize(const size_t total_num_samples) {
  THROW_CHECK_LE(num_samples_, total_num_samples);
  total_sample_idxs_.resize(total_num_samples);
  // Note that the samples must be in increasing order for `NextCombination`.
  std::iota(total_sample_idxs_.begin(), total_sample_idxs_.end(), 0);
}

size_t CombinationSampler::MaxNumSamples() {
  return NChooseK(total_sample_idxs_.size(), num_samples_);
}

void CombinationSampler::Sample(std::vector<size_t>* sampled_idxs) {
  sampled_idxs->resize(num_samples_);
  for (size_t i = 0; i < num_samples_; ++i) {
    (*sampled_idxs)[i] = total_sample_idxs_[i];
  }

  if (!NextCombination(total_sample_idxs_.begin(),
                       total_sample_idxs_.begin() + num_samples_,
                       total_sample_idxs_.end())) {
    // Reached all possible combinations, so reset to original state.
    // Note that the samples must be in increasing order for `NextCombination`.
    std::iota(total_sample_idxs_.begin(), total_sample_idxs_.end(), 0);
  }
}

}  // namespace colmap
