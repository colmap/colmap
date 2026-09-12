// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/progressive_sampler.h"

#include "colmap/math/random.h"
#include "colmap/util/misc.h"

namespace colmap {

ProgressiveSampler::ProgressiveSampler(const size_t num_samples)
    : num_samples_(num_samples),
      total_num_samples_(0),
      t_(0),
      n_(0),
      T_n_(0),
      T_n_p_(0) {}

void ProgressiveSampler::Initialize(const size_t total_num_samples) {
  THROW_CHECK_LE(num_samples_, total_num_samples);
  total_num_samples_ = total_num_samples;

  t_ = 0;
  n_ = num_samples_;

  // Number of iterations before PROSAC behaves like RANSAC. Default value
  // is chosen according to the recommended value in the paper.
  const size_t kNumProgressiveIterations = 200000;

  // Compute T_n using recurrent relation in equation 3 (first part).
  T_n_ = kNumProgressiveIterations;
  T_n_p_ = 1.0;
  for (size_t i = 0; i < num_samples_; ++i) {
    T_n_ *= static_cast<double>(num_samples_ - i) / (total_num_samples_ - i);
  }
}

size_t ProgressiveSampler::MaxNumSamples() {
  return std::numeric_limits<size_t>::max();
}

void ProgressiveSampler::Sample(std::vector<size_t>* sampled_idxs) {
  t_ += 1;

  sampled_idxs->clear();
  sampled_idxs->reserve(num_samples_);

  // Compute T_n_p_ using recurrent relation in equation 3 (second part).
  if (t_ == T_n_p_ && n_ < total_num_samples_) {
    const double T_n_plus_1 = T_n_ * (n_ + 1.0) / (n_ + 1.0 - num_samples_);
    T_n_p_ += std::ceil(T_n_plus_1 - T_n_);
    T_n_ = T_n_plus_1;
    n_ += 1;
  }

  // Decide how many samples to draw from which part of the data as
  // specified in equation 5.
  size_t num_random_samples = num_samples_;
  size_t max_random_sample_idx = n_ - 1;
  if (T_n_p_ >= t_) {
    num_random_samples -= 1;
    max_random_sample_idx -= 1;
  }

  // Draw semi-random samples as described in algorithm 1.
  for (size_t i = 0; i < num_random_samples; ++i) {
    while (true) {
      const size_t random_idx =
          RandomUniformInteger<uint32_t>(0, max_random_sample_idx);
      if (!VectorContainsValue(*sampled_idxs, random_idx)) {
        sampled_idxs->push_back(random_idx);
        break;
      }
    }
  }

  // In progressive sampling mode, the last element is mandatory.
  if (T_n_p_ >= t_) {
    sampled_idxs->push_back(n_);
  }
}

}  // namespace colmap
