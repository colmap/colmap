// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/optim/random_sampler.h"

#include "colmap/util/random.h"

#include <numeric>

namespace colmap {

RandomSampler::RandomSampler(const size_t num_samples)
    : num_samples_(num_samples) {}

void RandomSampler::Initialize(const size_t total_num_samples) {
  CHECK_LE(num_samples_, total_num_samples);
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
