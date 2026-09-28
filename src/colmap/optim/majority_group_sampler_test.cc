// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/majority_group_sampler.h"

#include "colmap/math/random.h"
#include "colmap/util/hash_containers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

bool IsStructuredSample(const std::vector<size_t>& samples,
                        const std::vector<int>& group_ids,
                        size_t majority_size) {
  if (samples.size() < 2) return false;
  FlatHashMap<int, size_t> counts;
  for (size_t idx : samples) counts[group_ids[idx]]++;
  if (counts.size() != 2) return false;
  for (const auto& [group, count] : counts) {
    if (count != majority_size && count != samples.size() - majority_size) {
      return false;
    }
  }
  return true;
}

TEST(MajorityGroupSampler, StructuredSampling) {
  SetPRNGSeed(0);
  // Two groups of 10; every draw must be 5+1 structured.
  std::vector<int> group_ids(20, 0);
  for (size_t i = 10; i < 20; ++i) group_ids[i] = 1;
  MajorityGroupSampler sampler(
      6, group_ids, /*structured_prob=*/1.0, /*majority_size=*/5);
  sampler.Initialize(20);
  for (int i = 0; i < 200; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(samples.size(), 6);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
    for (size_t idx : samples) EXPECT_LT(idx, 20);
    EXPECT_TRUE(IsStructuredSample(samples, group_ids, /*majority_size=*/5));
  }
}

TEST(MajorityGroupSampler, CustomMajoritySize) {
  SetPRNGSeed(0);
  // Two groups of 10; every draw must be 4+2 structured.
  std::vector<int> group_ids(20, 0);
  for (size_t i = 10; i < 20; ++i) group_ids[i] = 1;
  MajorityGroupSampler sampler(
      6, group_ids, /*structured_prob=*/1.0, /*majority_size=*/4);
  sampler.Initialize(20);
  for (int i = 0; i < 200; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
    EXPECT_TRUE(IsStructuredSample(samples, group_ids, /*majority_size=*/4));
  }
}

TEST(MajorityGroupSampler, MixedSampling) {
  SetPRNGSeed(0);
  std::vector<int> group_ids(20, 0);
  for (size_t i = 10; i < 20; ++i) group_ids[i] = 1;
  MajorityGroupSampler sampler(
      6, group_ids, /*structured_prob=*/0.5, /*majority_size=*/5);
  sampler.Initialize(20);
  int num_structured = 0;
  for (int i = 0; i < 500; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
    if (IsStructuredSample(samples, group_ids, /*majority_size=*/5)) {
      ++num_structured;
    }
  }
  EXPECT_GT(num_structured, 0);
  EXPECT_LT(num_structured, 500);
}

TEST(MajorityGroupSampler, UniformFallback) {
  SetPRNGSeed(0);
  // No groups: uniform sampling.
  MajorityGroupSampler sampler(
      6, /*group_ids=*/{}, /*structured_prob=*/0.5, /*majority_size=*/5);
  sampler.Initialize(20);
  for (int i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
  }

  // Single group: structured sampling infeasible, uniform fallback.
  MajorityGroupSampler single_group_sampler(6,
                                            std::vector<int>(20, 0),
                                            /*structured_prob=*/1.0,
                                            /*majority_size=*/5);
  single_group_sampler.Initialize(20);
  for (int i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    single_group_sampler.Sample(&samples);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
  }

  // Groups too small for a 5+1 sample: uniform fallback.
  std::vector<int> small_groups(20);
  for (size_t i = 0; i < 20; ++i) small_groups[i] = i / 2;
  MajorityGroupSampler small_group_sampler(
      6, small_groups, /*structured_prob=*/1.0, /*majority_size=*/5);
  small_group_sampler.Initialize(20);
  for (int i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    small_group_sampler.Sample(&samples);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
  }

  // Majority group exists but no group can supply 2 minority samples
  // for a 4+2 split: uniform fallback.
  std::vector<int> no_minority_groups(20, 0);
  for (size_t i = 10; i < 20; ++i) no_minority_groups[i] = i;
  MajorityGroupSampler no_minority_sampler(
      6, no_minority_groups, /*structured_prob=*/1.0, /*majority_size=*/4);
  no_minority_sampler.Initialize(20);
  for (int i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    no_minority_sampler.Sample(&samples);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
  }
}

}  // namespace
}  // namespace colmap
