// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/combination_sampler.h"

#include "colmap/math/math.h"
#include "colmap/util/hash_containers.h"

#include <exception>
#include <numeric>

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

TEST(CombinationSampler, MatchesGenericOrderAcrossCycles) {
  for (size_t total = 0; total <= 10; ++total) {
    for (size_t selected = 0; selected <= total; ++selected) {
      SCOPED_TRACE(testing::Message()
                   << "total=" << total << ", selected=" << selected);
      CombinationSampler sampler(selected);
      sampler.Initialize(total);
      const size_t count = NChooseK(total, selected);
      ASSERT_EQ(sampler.MaxNumSamples(), count);
      std::vector<size_t> population(total);
      std::iota(population.begin(), population.end(), 0);
      std::vector<size_t> actual;
      for (size_t trial = 0; trial < 2 * count + 1; ++trial) {
        const std::vector<size_t> expected(population.begin(),
                                           population.begin() + selected);
        sampler.Sample(&actual);
        ASSERT_EQ(actual, expected);
        if (!NextCombination(population.begin(),
                             population.begin() + selected,
                             population.end())) {
          std::iota(population.begin(), population.end(), 0);
        }
      }
    }
  }
}

TEST(CombinationSampler, CopyAndReinitializePreserveOrder) {
  CombinationSampler sampler(2);
  sampler.Initialize(5);
  std::vector<size_t> sample;
  for (size_t i = 0; i < 7; ++i) {
    sampler.Sample(&sample);
  }
  CombinationSampler copy(sampler);
  for (size_t i = 0; i < 12; ++i) {
    std::vector<size_t> copied_sample;
    sampler.Sample(&sample);
    copy.Sample(&copied_sample);
    ASSERT_EQ(sample, copied_sample);
  }

  sampler.Initialize(3);
  EXPECT_EQ(sampler.MaxNumSamples(), 3);
  const std::vector<std::vector<size_t>> expected = {
      {0, 1}, {0, 2}, {1, 2}, {0, 1}};
  for (const auto& next : expected) {
    sampler.Sample(&sample);
    ASSERT_EQ(sample, next);
  }
  sampler.Initialize(8);
  EXPECT_EQ(sampler.MaxNumSamples(), 28);
  sampler.Sample(&sample);
  EXPECT_EQ(sample, (std::vector<size_t>{0, 1}));
}

TEST(CombinationSampler, FailedInitializePreservesProgress) {
  CombinationSampler sampler(2);
  sampler.Initialize(5);
  std::vector<size_t> sample;
  sampler.Sample(&sample);
  CombinationSampler copy(sampler);
  EXPECT_THROW(sampler.Initialize(1), std::exception);
  EXPECT_EQ(sampler.MaxNumSamples(), 10);
  for (size_t i = 0; i < 12; ++i) {
    std::vector<size_t> copied_sample;
    sampler.Sample(&sample);
    copy.Sample(&copied_sample);
    ASSERT_EQ(sample, copied_sample);
  }
}

}  // namespace
}  // namespace colmap
