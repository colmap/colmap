// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/combination_sampler.h"

#include "colmap/math/math.h"

#include <numeric>

namespace colmap {

CombinationSampler::CombinationSampler(const size_t num_samples)
    : num_samples_(num_samples) {}

void CombinationSampler::Initialize(const size_t total_num_samples) {
  THROW_CHECK_LE(num_samples_, total_num_samples);
  sample_idxs_.resize(num_samples_);
  std::iota(sample_idxs_.begin(), sample_idxs_.end(), size_t{0});
  total_num_samples_ = total_num_samples;
}

size_t CombinationSampler::MaxNumSamples() {
  return NChooseK(total_num_samples_, num_samples_);
}

void CombinationSampler::Sample(std::vector<size_t>* sampled_idxs) {
  *sampled_idxs = sample_idxs_;
  if (num_samples_ == total_num_samples_) {
    return;
  }

  for (size_t i = num_samples_; i > 0; --i) {
    const size_t pos = i - 1;
    if (sample_idxs_[pos] < total_num_samples_ - num_samples_ + pos) {
      ++sample_idxs_[pos];
      for (size_t j = pos + 1; j < num_samples_; ++j) {
        sample_idxs_[j] = sample_idxs_[j - 1] + 1;
      }
      return;
    }
  }

  // Reached all possible combinations, so reset to the first sample.
  std::iota(sample_idxs_.begin(), sample_idxs_.end(), size_t{0});
}

}  // namespace colmap
