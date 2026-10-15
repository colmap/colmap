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
    for (const size_t sample : samples) {
      EXPECT_LT(sample, 5);
    }
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
    for (const size_t sample : samples) {
      EXPECT_LT(sample, 5);
    }
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
    EXPECT_LT(samples.back(), 50);
    for (size_t i = 0; i < samples.size() - 1; ++i) {
      EXPECT_LT(samples[i], samples.back());
      EXPECT_GE(samples.back(), prev_last_sample);
      prev_last_sample = samples.back();
    }
  }
}

TEST(ProgressiveSampler, FullPopulationWithOneSample) {
  ProgressiveSampler sampler(1);
  for (size_t iteration = 0; iteration < 3; ++iteration) {
    sampler.Initialize(1);
    for (size_t draw = 0; draw < 2; ++draw) {
      std::vector<size_t> samples;
      sampler.Sample(&samples);
      ASSERT_EQ(samples.size(), 1);
      EXPECT_EQ(samples[0], 0);
    }
  }
}

}  // namespace
}  // namespace colmap
