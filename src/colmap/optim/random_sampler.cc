// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/random_sampler.h"

#include "colmap/math/random.h"

#include <numeric>

namespace colmap {

RandomSampler::RandomSampler(const size_t num_samples)
    : num_samples_(num_samples) {}

void RandomSampler::Initialize(const size_t total_num_samples) {
  THROW_CHECK_LE(num_samples_, total_num_samples);
  sample_idxs_.resize(total_num_samples);
  std::iota(sample_idxs_.begin(), sample_idxs_.end(), 0);
}

size_t RandomSampler::MaxNumSamples() {
  return std::numeric_limits<size_t>::max();
}

void RandomSampler::Sample(std::vector<size_t>* sampled_idxs) {
  Shuffle(static_cast<uint32_t>(num_samples_), &sample_idxs_);

  sampled_idxs->resize(num_samples_);
  for (size_t i = 0; i < num_samples_; ++i) {
    (*sampled_idxs)[i] = sample_idxs_[i];
  }
}

}  // namespace colmap
