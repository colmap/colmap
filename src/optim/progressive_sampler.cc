// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "optim/progressive_sampler.h"

#include <numeric>

#include "util/misc.h"
#include "util/random.h"

namespace colmap {

ProgressiveSampler::ProgressiveSampler(const size_t num_samples)
    : num_samples_(num_samples),
      total_num_samples_(0),
      t_(0),
      n_(0),
      T_n_(0),
      T_n_p_(0) {}

void ProgressiveSampler::Initialize(const size_t total_num_samples) {
  CHECK_LE(num_samples_, total_num_samples);
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

std::vector<size_t> ProgressiveSampler::Sample() {
  t_ += 1;

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
  std::vector<size_t> sampled_idxs;
  sampled_idxs.reserve(num_samples_);
  for (size_t i = 0; i < num_random_samples; ++i) {
    while (true) {
      const size_t random_idx =
          RandomInteger<uint32_t>(0, max_random_sample_idx);
      if (!VectorContainsValue(sampled_idxs, random_idx)) {
        sampled_idxs.push_back(random_idx);
        break;
      }
    }
  }

  // In progressive sampling mode, the last element is mandatory.
  if (T_n_p_ >= t_) {
    sampled_idxs.push_back(n_);
  }

  return sampled_idxs;
}

}  // namespace colmap
