// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/two_group_sampler.h"

#include "colmap/math/random.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"

#include <numeric>

namespace colmap {

TwoGroupSampler::TwoGroupSampler(size_t num_samples,
                                 std::vector<int> group_ids,
                                 double structured_prob,
                                 size_t num_first_group_samples)
    : num_samples_(num_samples),
      group_ids_(std::move(group_ids)),
      structured_prob_(structured_prob),
      num_first_group_samples_(num_first_group_samples),
      num_second_group_samples_(num_samples - num_first_group_samples) {
  THROW_CHECK_GE(structured_prob_, 0.0);
  THROW_CHECK_LE(structured_prob_, 1.0);
  THROW_CHECK_GE(num_first_group_samples_, 1);
  THROW_CHECK_LT(num_first_group_samples_, num_samples_);
}

void TwoGroupSampler::Initialize(const size_t total_num_samples) {
  THROW_CHECK_LE(num_samples_, total_num_samples);
  sample_idxs_.resize(total_num_samples);
  std::iota(sample_idxs_.begin(), sample_idxs_.end(), 0);

  group_members_.clear();
  eligible_first_groups_.clear();
  eligible_first_total_size_ = 0;
  eligible_second_groups_.clear();
  second_group_pos_.clear();
  if (!group_ids_.empty()) {
    THROW_CHECK_EQ(group_ids_.size(), total_num_samples);
    FlatHashMap<int, size_t> group_index;
    for (size_t i = 0; i < group_ids_.size(); ++i) {
      const auto [it, inserted] =
          group_index.try_emplace(group_ids_[i], group_members_.size());
      if (inserted) {
        group_members_.emplace_back();
      }
      group_members_[it->second].push_back(i);
    }
    second_group_pos_.assign(group_members_.size(), kInvalidGroupPos);
    for (size_t g = 0; g < group_members_.size(); ++g) {
      const size_t group_size = group_members_[g].size();
      if (group_size >= num_first_group_samples_) {
        eligible_first_groups_.push_back(g);
        eligible_first_total_size_ += group_size;
      }
      if (group_size >= num_second_group_samples_) {
        second_group_pos_[g] = eligible_second_groups_.size();
        eligible_second_groups_.push_back(g);
      }
    }
  }
}

size_t TwoGroupSampler::MaxNumSamples() {
  return std::numeric_limits<size_t>::max();
}

void TwoGroupSampler::Sample(std::vector<size_t>* sampled_idxs) {
  THROW_CHECK_NOTNULL(sampled_idxs);
  sampled_idxs->resize(num_samples_);

  // Structured sample: num_first_group_samples_ indices from a size-weighted
  // group plus the rest from a uniformly chosen different group.
  if (group_members_.size() >= 2 && !eligible_first_groups_.empty() &&
      RandomUniformReal(0.0, 1.0) < structured_prob_) {
    size_t first_group = eligible_first_groups_.back();
    size_t ballot =
        RandomUniformInteger<size_t>(0, eligible_first_total_size_ - 1);
    for (const size_t g : eligible_first_groups_) {
      if (ballot < group_members_[g].size()) {
        first_group = g;
        break;
      }
      ballot -= group_members_[g].size();
    }

    // Uniform second group among the precomputed eligible ones, skipping the
    // first group.
    size_t num_second_choices = eligible_second_groups_.size();
    const size_t first_group_pos = second_group_pos_[first_group];
    if (first_group_pos != kInvalidGroupPos) {
      num_second_choices -= 1;
    }
    if (num_second_choices > 0) {
      size_t pick = RandomUniformInteger<size_t>(0, num_second_choices - 1);
      if (first_group_pos != kInvalidGroupPos && pick >= first_group_pos) {
        ++pick;
      }
      const size_t second_group = eligible_second_groups_[pick];

      // Partially shuffle the persistent member lists in place, as for
      // sample_idxs_ below: the leading elements are a uniform random subset
      // regardless of the order left behind by previous draws.
      std::vector<size_t>& first_members = group_members_[first_group];
      Shuffle(static_cast<uint32_t>(num_first_group_samples_), &first_members);
      for (size_t i = 0; i < num_first_group_samples_; ++i) {
        (*sampled_idxs)[i] = first_members[i];
      }
      std::vector<size_t>& second_members = group_members_[second_group];
      Shuffle(static_cast<uint32_t>(num_second_group_samples_),
              &second_members);
      for (size_t i = 0; i < num_second_group_samples_; ++i) {
        (*sampled_idxs)[num_first_group_samples_ + i] = second_members[i];
      }
      return;
    }
  }

  // Uniform fallback.
  Shuffle(static_cast<uint32_t>(num_samples_), &sample_idxs_);
  for (size_t i = 0; i < num_samples_; ++i) {
    (*sampled_idxs)[i] = sample_idxs_[i];
  }
}

}  // namespace colmap
