// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/combination_sampler.h"

#include "colmap/util/hash_containers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CombinationSampler, LessSamples) {
  CombinationSampler sampler(2);
  sampler.Initialize(5);
  EXPECT_EQ(sampler.MaxNumSamples(), 10);
  std::vector<FlatHashSet<size_t>> sample_sets;
  for (size_t i = 0; i < 10; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(samples.size(), 2);
    sample_sets.emplace_back(samples.begin(), samples.end());
    EXPECT_EQ(sample_sets.back().size(), 2);
    for (size_t j = 0; j < i; ++j) {
      EXPECT_TRUE(sample_sets[j].count(samples[0]) == 0 ||
                  sample_sets[j].count(samples[1]) == 0);
    }
  }
  std::vector<size_t> samples;
  sampler.Sample(&samples);
  EXPECT_TRUE(sample_sets[0].count(samples[0]) == 1 &&
              sample_sets[0].count(samples[1]) == 1);
}

TEST(CombinationSampler, EqualSamples) {
  CombinationSampler sampler(5);
  sampler.Initialize(5);
  EXPECT_EQ(sampler.MaxNumSamples(), 1);
  for (size_t i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(samples.size(), 5);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 5);
  }
}

}  // namespace
}  // namespace colmap
