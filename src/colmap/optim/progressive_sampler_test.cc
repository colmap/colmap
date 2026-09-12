// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/progressive_sampler.h"

#include "colmap/util/hash_containers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ProgressiveSampler, LessSamples) {
  ProgressiveSampler sampler(2);
  sampler.Initialize(5);
  EXPECT_EQ(sampler.MaxNumSamples(), std::numeric_limits<size_t>::max());
  for (size_t i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(samples.size(), 2);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 2);
  }
}

TEST(ProgressiveSampler, EqualSamples) {
  ProgressiveSampler sampler(5);
  sampler.Initialize(5);
  EXPECT_EQ(sampler.MaxNumSamples(), std::numeric_limits<size_t>::max());
  for (size_t i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(samples.size(), 5);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 5);
  }
}

TEST(ProgressiveSampler, Progressive) {
  const size_t kNumSamples = 5;
  ProgressiveSampler sampler(kNumSamples);
  sampler.Initialize(50);
  size_t prev_last_sample = 5;
  for (size_t i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    for (size_t i = 0; i < samples.size() - 1; ++i) {
      EXPECT_LT(samples[i], samples.back());
      EXPECT_GE(samples.back(), prev_last_sample);
      prev_last_sample = samples.back();
    }
  }
}

}  // namespace
}  // namespace colmap
