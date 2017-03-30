// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "optim/combination_sampler.h"

#include <numeric>

#include "util/math.h"
#include "util/random.h"

namespace colmap {

CombinationSampler::CombinationSampler(const size_t num_samples)
    : num_samples_(num_samples) {}

void CombinationSampler::Initialize(const size_t total_num_samples) {
  CHECK_LE(num_samples_, total_num_samples);
  total_sample_idxs_.resize(total_num_samples);
  // Note that the samples must be in increasing order for `NextCombination`.
  std::iota(total_sample_idxs_.begin(), total_sample_idxs_.end(), 0);
}

size_t CombinationSampler::MaxNumSamples() {
  return NChooseK(total_sample_idxs_.size(), num_samples_);
}

std::vector<size_t> CombinationSampler::Sample() {
  std::vector<size_t> sampled_idxs(num_samples_);
  for (size_t i = 0; i < num_samples_; ++i) {
    sampled_idxs[i] = total_sample_idxs_[i];
  }

  if (!NextCombination(total_sample_idxs_.begin(),
                       total_sample_idxs_.begin() + num_samples_,
                       total_sample_idxs_.end())) {
    // Reached all possible combinations, so reset to original state.
    // Note that the samples must be in increasing order for `NextCombination`.
    std::iota(total_sample_idxs_.begin(), total_sample_idxs_.end(), 0);
  }

  return sampled_idxs;
}

}  // namespace colmap
