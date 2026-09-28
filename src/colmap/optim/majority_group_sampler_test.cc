// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/majority_group_sampler.h"

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
    if (IsStructuredSample(sample_idxs, group_ids, /*majority_size=*/5)) {
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

TEST(MajorityGroupSampler, RANSACPropagatesGroupsToThreads) {
  // RANSAC copies the sampler per thread; every copy must keep the group labels
  // and draw only 5+1 structured samples.
  std::vector<int> group_ids(20, 0);
  for (size_t i = 10; i < 20; ++i) group_ids[i] = 1;
  for (const int num_threads : {1, 4}) {
    const RANSACOptions options =
        CreateSampleStructureRANSACOptions(num_threads);
    std::atomic<int> num_samples(0);
    std::atomic<int> num_structured_samples(0);
    RANSAC<SampleStructureCounter, InlierSupportMeasurer, MajorityGroupSampler>
        ransac(options,
               SampleStructureCounter(&num_samples, &num_structured_samples),
               InlierSupportMeasurer(),
               MajorityGroupSampler(SampleStructureCounter::kMinNumSamples,
                                    group_ids,
                                    /*structured_prob=*/1.0,
                                    /*majority_size=*/5));
    EXPECT_FALSE(ransac.Estimate(group_ids, group_ids).success);
    EXPECT_EQ(num_samples, options.max_num_trials);
    EXPECT_EQ(num_structured_samples, options.max_num_trials);
  }
}

TEST(MajorityGroupSampler, LORANSACPropagatesGroupsToThreads) {
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
             MajorityGroupSampler>
        loransac(options,
                 counter,
                 counter,
                 InlierSupportMeasurer(),
                 MajorityGroupSampler(SampleStructureCounter::kMinNumSamples,
                                      group_ids,
                                      /*structured_prob=*/1.0,
                                      /*majority_size=*/5));
    EXPECT_FALSE(loransac.Estimate(group_ids, group_ids).success);
    EXPECT_EQ(num_samples, options.max_num_trials);
    EXPECT_EQ(num_structured_samples, options.max_num_trials);
  }
}

}  // namespace
}  // namespace colmap
