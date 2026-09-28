// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/two_group_sampler.h"

#include "colmap/math/random.h"
#include "colmap/optim/loransac.h"
#include "colmap/optim/ransac.h"
#include "colmap/util/hash_containers.h"

#include <atomic>
#include <numeric>

#include <gtest/gtest.h>

namespace colmap {
namespace {

bool IsStructuredSample(const std::vector<size_t>& samples,
                        const std::vector<int>& group_ids,
                        size_t num_first_group_samples) {
  if (samples.size() < 2) return false;
  FlatHashMap<int, size_t> counts;
  for (size_t idx : samples) counts[group_ids[idx]]++;
  if (counts.size() != 2) return false;
  for (const auto& [group, count] : counts) {
    if (count != num_first_group_samples &&
        count != samples.size() - num_first_group_samples) {
      return false;
    }
  }
  return true;
}

// Counts the minimal samples drawn by (LO-)RANSAC and how many of them are 5+1
// structured. The data points are their own group labels, so the structure can
// be checked directly on the sampled points. Returns no models, so RANSAC runs
// exactly max_num_trials trials.
class SampleStructureCounter {
 public:
  using X_t = int;
  using Y_t = int;
  using M_t = int;

  static const int kMinNumSamples = 6;

  SampleStructureCounter(std::atomic<int>* num_samples,
                         std::atomic<int>* num_structured_samples)
      : num_samples_(num_samples),
        num_structured_samples_(num_structured_samples) {}

  void Estimate(const std::vector<X_t>& group_ids,
                const std::vector<Y_t>& /*group_ids*/,
                std::vector<M_t>* /*models*/) const {
    std::vector<size_t> sample_idxs(group_ids.size());
    std::iota(sample_idxs.begin(), sample_idxs.end(), 0);
    ++*num_samples_;
    if (IsStructuredSample(
            sample_idxs, group_ids, /*num_first_group_samples=*/5)) {
      ++*num_structured_samples_;
    }
  }

  void Residuals(const std::vector<X_t>& group_ids,
                 const std::vector<Y_t>& /*group_ids*/,
                 const M_t& /*model*/,
                 std::vector<double>* residuals) const {
    residuals->assign(group_ids.size(), 0.0);
  }

 private:
  std::atomic<int>* num_samples_;
  std::atomic<int>* num_structured_samples_;
};

RANSACOptions CreateSampleStructureRANSACOptions(int num_threads) {
  RANSACOptions options;
  options.max_error = 1;
  options.max_num_trials = 200;
  options.random_seed = 0;
  options.num_threads = num_threads;
  return options;
}

TEST(TwoGroupSampler, StructuredSampling) {
  SetPRNGSeed(0);
  // Two groups of 10; every draw must be 5+1 structured.
  std::vector<int> group_ids(20, 0);
  for (size_t i = 10; i < 20; ++i) group_ids[i] = 1;
  TwoGroupSampler sampler(
      6, group_ids, /*structured_prob=*/1.0, /*num_first_group_samples=*/5);
  sampler.Initialize(20);
  for (int i = 0; i < 200; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(samples.size(), 6);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
    for (size_t idx : samples) EXPECT_LT(idx, 20);
    EXPECT_TRUE(
        IsStructuredSample(samples, group_ids, /*num_first_group_samples=*/5));
  }
}

TEST(TwoGroupSampler, CustomGroupSplit) {
  SetPRNGSeed(0);
  // Two groups of 10; every draw must be 4+2 structured.
  std::vector<int> group_ids(20, 0);
  for (size_t i = 10; i < 20; ++i) group_ids[i] = 1;
  TwoGroupSampler sampler(
      6, group_ids, /*structured_prob=*/1.0, /*num_first_group_samples=*/4);
  sampler.Initialize(20);
  for (int i = 0; i < 200; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
    EXPECT_TRUE(
        IsStructuredSample(samples, group_ids, /*num_first_group_samples=*/4));
  }
}

TEST(TwoGroupSampler, MixedSampling) {
  SetPRNGSeed(0);
  std::vector<int> group_ids(20, 0);
  for (size_t i = 10; i < 20; ++i) group_ids[i] = 1;
  TwoGroupSampler sampler(
      6, group_ids, /*structured_prob=*/0.5, /*num_first_group_samples=*/5);
  sampler.Initialize(20);
  int num_structured = 0;
  for (int i = 0; i < 500; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
    if (IsStructuredSample(samples, group_ids, /*num_first_group_samples=*/5)) {
      ++num_structured;
    }
  }
  EXPECT_GT(num_structured, 0);
  EXPECT_LT(num_structured, 500);
}

TEST(TwoGroupSampler, UniformSecondGroup) {
  SetPRNGSeed(0);
  // Only group 0 can supply the 5 first-group samples. The second group is
  // chosen uniformly regardless of size: group 2 (1 member) vs. group 1 (4
  // members), i.e., group 2 in ~50% of the draws (size-weighted: ~20%).
  std::vector<int> group_ids(20, 0);
  for (size_t i = 15; i < 19; ++i) group_ids[i] = 1;
  group_ids[19] = 2;
  TwoGroupSampler sampler(
      6, group_ids, /*structured_prob=*/1.0, /*num_first_group_samples=*/5);
  sampler.Initialize(20);
  constexpr int kNumDraws = 1000;
  int num_group2_draws = 0;
  for (int i = 0; i < kNumDraws; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_TRUE(
        IsStructuredSample(samples, group_ids, /*num_first_group_samples=*/5));
    for (size_t idx : samples) {
      if (group_ids[idx] == 2) ++num_group2_draws;
    }
  }
  EXPECT_NEAR(static_cast<double>(num_group2_draws) / kNumDraws, 0.5, 0.05);
}

TEST(TwoGroupSampler, UniformFallback) {
  SetPRNGSeed(0);
  // No groups: uniform sampling.
  TwoGroupSampler sampler(6,
                          /*group_ids=*/{},
                          /*structured_prob=*/0.5,
                          /*num_first_group_samples=*/5);
  sampler.Initialize(20);
  for (int i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    sampler.Sample(&samples);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
  }

  // Single group: structured sampling infeasible, uniform fallback.
  TwoGroupSampler single_group_sampler(6,
                                       std::vector<int>(20, 0),
                                       /*structured_prob=*/1.0,
                                       /*num_first_group_samples=*/5);
  single_group_sampler.Initialize(20);
  for (int i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    single_group_sampler.Sample(&samples);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
  }

  // Groups too small for a 5+1 sample: uniform fallback.
  std::vector<int> small_groups(20);
  for (size_t i = 0; i < 20; ++i) small_groups[i] = i / 2;
  TwoGroupSampler small_group_sampler(
      6, small_groups, /*structured_prob=*/1.0, /*num_first_group_samples=*/5);
  small_group_sampler.Initialize(20);
  for (int i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    small_group_sampler.Sample(&samples);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
  }

  // A group can supply the 4 first-group samples of a 4+2 split, but no other
  // group can supply the 2 second-group samples: uniform fallback.
  std::vector<int> no_second_groups(20, 0);
  for (size_t i = 10; i < 20; ++i) no_second_groups[i] = i;
  TwoGroupSampler no_second_group_sampler(6,
                                          no_second_groups,
                                          /*structured_prob=*/1.0,
                                          /*num_first_group_samples=*/4);
  no_second_group_sampler.Initialize(20);
  for (int i = 0; i < 100; ++i) {
    std::vector<size_t> samples;
    no_second_group_sampler.Sample(&samples);
    EXPECT_EQ(FlatHashSet<size_t>(samples.begin(), samples.end()).size(), 6);
  }
}

TEST(TwoGroupSampler, RANSACPropagatesGroupsToThreads) {
  // RANSAC copies the sampler per thread; every copy must keep the group labels
  // and draw only 5+1 structured samples.
  std::vector<int> group_ids(20, 0);
  for (size_t i = 10; i < 20; ++i) group_ids[i] = 1;
  for (const int num_threads : {1, 4}) {
    const RANSACOptions options =
        CreateSampleStructureRANSACOptions(num_threads);
    std::atomic<int> num_samples(0);
    std::atomic<int> num_structured_samples(0);
    RANSAC<SampleStructureCounter, InlierSupportMeasurer, TwoGroupSampler>
        ransac(options,
               SampleStructureCounter(&num_samples, &num_structured_samples),
               InlierSupportMeasurer(),
               TwoGroupSampler(SampleStructureCounter::kMinNumSamples,
                               group_ids,
                               /*structured_prob=*/1.0,
                               /*num_first_group_samples=*/5));
    EXPECT_FALSE(ransac.Estimate(group_ids, group_ids).success);
    EXPECT_EQ(num_samples, options.max_num_trials);
    EXPECT_EQ(num_structured_samples, options.max_num_trials);
  }
}

TEST(TwoGroupSampler, LORANSACPropagatesGroupsToThreads) {
  std::vector<int> group_ids(20, 0);
  for (size_t i = 10; i < 20; ++i) group_ids[i] = 1;
  for (const int num_threads : {1, 4}) {
    const RANSACOptions options =
        CreateSampleStructureRANSACOptions(num_threads);
    std::atomic<int> num_samples(0);
    std::atomic<int> num_structured_samples(0);
    const SampleStructureCounter counter(&num_samples, &num_structured_samples);
    LORANSAC<SampleStructureCounter,
             SampleStructureCounter,
             InlierSupportMeasurer,
             TwoGroupSampler>
        loransac(options,
                 counter,
                 counter,
                 InlierSupportMeasurer(),
                 TwoGroupSampler(SampleStructureCounter::kMinNumSamples,
                                 group_ids,
                                 /*structured_prob=*/1.0,
                                 /*num_first_group_samples=*/5));
    EXPECT_FALSE(loransac.Estimate(group_ids, group_ids).success);
    EXPECT_EQ(num_samples, options.max_num_trials);
    EXPECT_EQ(num_structured_samples, options.max_num_trials);
  }
}

}  // namespace
}  // namespace colmap
