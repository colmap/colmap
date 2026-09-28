// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/majority_group_sampler.h"

#include "colmap/math/random.h"
#include "colmap/util/logging.h"

#include <numeric>
#include <unordered_map>

namespace colmap {

MajorityGroupSampler::MajorityGroupSampler(size_t num_samples,
                                           std::vector<int> group_ids,
                                           double structured_prob,
                                           size_t majority_size)
    : num_samples_(num_samples),
      group_ids_(std::move(group_ids)),
      structured_prob_(structured_prob),
      majority_size_(majority_size),
      minority_size_(num_samples - majority_size) {
  THROW_CHECK_GE(structured_prob_, 0.0);
  THROW_CHECK_LE(structured_prob_, 1.0);
  THROW_CHECK_GE(majority_size_, 1);
  THROW_CHECK_LT(majority_size_, num_samples_);
}

void MajorityGroupSampler::Initialize(const size_t total_num_samples) {
  THROW_CHECK_LE(num_samples_, total_num_samples);
  sample_idxs_.resize(total_num_samples);
  std::iota(sample_idxs_.begin(), sample_idxs_.end(), 0);

  group_members_.clear();
  eligible_major_groups_.clear();
  eligible_major_total_size_ = 0;
  eligible_minor_groups_.clear();
  minor_pos_.clear();
  if (!group_ids_.empty()) {
    THROW_CHECK_EQ(group_ids_.size(), total_num_samples);
    std::unordered_map<int, size_t> group_index;
    for (size_t i = 0; i < group_ids_.size(); ++i) {
      const auto [it, inserted] =
          group_index.try_emplace(group_ids_[i], group_members_.size());
      if (inserted) {
        group_members_.emplace_back();
      }
      group_members_[it->second].push_back(i);
    }
    if (majority_size_ >= 1 && majority_size_ < num_samples_) {
      for (size_t g = 0; g < group_members_.size(); ++g) {
        if (group_members_[g].size() >= majority_size_) {
          eligible_major_groups_.push_back(g);
          eligible_major_total_size_ += group_members_[g].size();
        }
      }
    }
    minor_pos_.assign(group_members_.size(), kInvalidMinorPos);
    for (size_t g = 0; g < group_members_.size(); ++g) {
      if (group_members_[g].size() >= minority_size_) {
        minor_pos_[g] = eligible_minor_groups_.size();
        eligible_minor_groups_.push_back(g);
      }
    }
  }
}

size_t MajorityGroupSampler::MaxNumSamples() {
  return std::numeric_limits<size_t>::max();
}

void MajorityGroupSampler::Sample(std::vector<size_t>* sampled_idxs) {
  THROW_CHECK_NOTNULL(sampled_idxs);
  sampled_idxs->resize(num_samples_);

  // Structured sample: majority_size_ indices from a size-weighted group
  // plus the rest from a uniformly chosen different group.
  if (group_members_.size() >= 2 && !eligible_major_groups_.empty() &&
      RandomUniformReal(0.0, 1.0) < structured_prob_) {
    size_t major_group = eligible_major_groups_.back();
    size_t ballot =
        RandomUniformInteger<size_t>(0, eligible_major_total_size_ - 1);
    for (const size_t g : eligible_major_groups_) {
      if (ballot < group_members_[g].size()) {
        major_group = g;
        break;
      }
      ballot -= group_members_[g].size();
    }

    // Uniform minor group among the precomputed eligible ones, skipping the
    // majority group.
    size_t num_minor_choices = eligible_minor_groups_.size();
    const size_t major_minor_pos = minor_pos_[major_group];
    if (major_minor_pos != kInvalidMinorPos) {
      num_minor_choices -= 1;
    }
    if (num_minor_choices > 0) {
      size_t pick = RandomUniformInteger<size_t>(0, num_minor_choices - 1);
      if (major_minor_pos != kInvalidMinorPos && pick >= major_minor_pos) {
        ++pick;
      }
      const size_t minor_group = eligible_minor_groups_[pick];

      draw_buffer_ = group_members_[major_group];
      Shuffle(static_cast<uint32_t>(majority_size_), &draw_buffer_);
      for (size_t i = 0; i < majority_size_; ++i) {
        (*sampled_idxs)[i] = draw_buffer_[i];
      }
      draw_buffer_ = group_members_[minor_group];
      Shuffle(static_cast<uint32_t>(minority_size_), &draw_buffer_);
      for (size_t i = 0; i < minority_size_; ++i) {
        (*sampled_idxs)[majority_size_ + i] = draw_buffer_[i];
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
