// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/onnx_matchers.h"

#include "colmap/feature/matcher.h"
#include "colmap/feature/resources.h"
#include "colmap/feature/utils.h"
#include "colmap/util/onnx.h"

#include <cstring>
#include <future>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

class BruteForceONNXMatcherTest : public testing::Test {
 protected:
  static constexpr int kDescriptorDim = 128;

  static FeatureMatchingOptions CreateFeatureMatcherOptions(
      FeatureMatcherType type = FeatureMatcherType::ALIKED_BRUTEFORCE) {
    FeatureMatchingOptions options(type);
    options.use_gpu = false;
    return options;
  }

  static BruteForceONNXMatchingOptions CreateBruteForceONNXMatchingOptions() {
    BruteForceONNXMatchingOptions options;
    options.model_path = kDefaultBruteForceONNXMatcherUri;
    return options;
  }

  static std::unique_ptr<FeatureMatcher> CreateBruteForceMatcher(
      const FeatureMatchingOptions& options,
      const BruteForceONNXMatchingOptions& brute_force_options,
      std::vector<FeatureExtractorType> supported_feature_types =
          {FeatureExtractorType::ALIKED_N16ROT,
           FeatureExtractorType::ALIKED_N32},
      bool normalize_descriptors = false) {
    return CreateBruteForceONNXFeatureMatcher(
        options,
        brute_force_options,
        std::move(supported_feature_types),
        normalize_descriptors);
  }

  static std::shared_ptr<FeatureDescriptors> CreateRandomDescriptors(
      int num_descriptors,
      FeatureExtractorType type = FeatureExtractorType::ALIKED_N16ROT,
      bool normalize = true) {
    const auto descriptors = std::make_shared<FeatureDescriptors>();
    descriptors->type = type;
    descriptors->data.resize(num_descriptors, kDescriptorDim * sizeof(float));

    if (num_descriptors == 0) {
      return descriptors;
    }

    FeatureDescriptorsFloatData float_data =
        FeatureDescriptorsFloatData::Random(num_descriptors, kDescriptorDim);
    if (normalize) {
      L2NormalizeFeatureDescriptors(&float_data);
    }

    std::memcpy(descriptors->data.data(),
                float_data.data(),
                num_descriptors * kDescriptorDim * sizeof(float));
    return descriptors;
  }

  static std::shared_ptr<FeatureKeypoints> CreateDummyKeypoints(int count) {
    const auto keypoints = std::make_shared<FeatureKeypoints>(count);
    for (int i = 0; i < count; ++i) {
      (*keypoints)[i].x = static_cast<float>(i);
      (*keypoints)[i].y = static_cast<float>(i);
    }
    return keypoints;
  }
};

TEST_F(BruteForceONNXMatcherTest, SelfMatching) {
  constexpr int kNumDescriptors = 20;

  auto keypoints = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors = CreateRandomDescriptors(kNumDescriptors);

  BruteForceONNXMatchingOptions bf_options =
      CreateBruteForceONNXMatchingOptions();
  bf_options.max_ratio = 0;  // Disable ratio test.
  auto matcher =
      CreateBruteForceMatcher(CreateFeatureMatcherOptions(), bf_options);

  FeatureMatcher::Image image1{1, nullptr, keypoints, descriptors};
  FeatureMatcher::Image image2{2, nullptr, keypoints, descriptors};

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  // Self-matching should match every descriptor to itself.
  EXPECT_EQ(matches.size(), kNumDescriptors);
  for (const auto& match : matches) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
  }
}

TEST_F(BruteForceONNXMatcherTest, LomaB128Descriptors) {
  constexpr int kNumDescriptors = 20;

  auto keypoints = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors = CreateRandomDescriptors(kNumDescriptors,
                                             FeatureExtractorType::LOMA_B128,
                                             /*normalize=*/false);
  BruteForceONNXMatchingOptions bf_options =
      CreateBruteForceONNXMatchingOptions();
  auto matcher = CreateBruteForceMatcher(
      CreateFeatureMatcherOptions(FeatureMatcherType::LOMA_BRUTEFORCE),
      bf_options,
      {FeatureExtractorType::LOMA_B128},
      /*normalize_descriptors=*/true);

  FeatureMatcher::Image image1{1, nullptr, keypoints, descriptors};
  FeatureMatcher::Image image2{2, nullptr, keypoints, descriptors};
  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  EXPECT_GT(matches.size(), 0);
  for (const auto& match : matches) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
  }
}

TEST_F(BruteForceONNXMatcherTest, RatioTest) {
  constexpr int kNumDescriptors = 20;

  auto keypoints1 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors1 = CreateRandomDescriptors(kNumDescriptors);
  auto keypoints2 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors2 = CreateRandomDescriptors(kNumDescriptors);

  BruteForceONNXMatchingOptions bf_options_no_ratio =
      CreateBruteForceONNXMatchingOptions();
  bf_options_no_ratio.max_ratio = 0;  // Disable ratio test.
  bf_options_no_ratio.cross_check = false;
  bf_options_no_ratio.min_cossim = -1;  // Accept all.
  auto matcher_no_ratio = CreateBruteForceMatcher(CreateFeatureMatcherOptions(),
                                                  bf_options_no_ratio);

  BruteForceONNXMatchingOptions bf_options_ratio =
      CreateBruteForceONNXMatchingOptions();
  bf_options_ratio.max_ratio = 0.8;  // Enable ratio test.
  bf_options_ratio.cross_check = false;
  bf_options_ratio.min_cossim = -1;  // Accept all.
  auto matcher_ratio =
      CreateBruteForceMatcher(CreateFeatureMatcherOptions(), bf_options_ratio);

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches_no_ratio, matches_ratio;
  matcher_no_ratio->Match(image1, image2, &matches_no_ratio);
  matcher_ratio->Match(image1, image2, &matches_ratio);

  // Ratio test should filter some matches.
  EXPECT_GT(matches_no_ratio.size(), 0);
  EXPECT_LT(matches_ratio.size(), matches_no_ratio.size());
}

TEST_F(BruteForceONNXMatcherTest, CrossCheck) {
  constexpr int kNumDescriptors = 20;

  auto keypoints1 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors1 = CreateRandomDescriptors(kNumDescriptors);
  auto keypoints2 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors2 = CreateRandomDescriptors(kNumDescriptors);

  BruteForceONNXMatchingOptions bf_options_no_cross =
      CreateBruteForceONNXMatchingOptions();
  bf_options_no_cross.max_ratio = 0;
  bf_options_no_cross.cross_check = false;
  bf_options_no_cross.min_cossim = -1;
  auto matcher_no_cross = CreateBruteForceMatcher(CreateFeatureMatcherOptions(),
                                                  bf_options_no_cross);

  BruteForceONNXMatchingOptions bf_options_cross =
      CreateBruteForceONNXMatchingOptions();
  bf_options_cross.max_ratio = 0;
  bf_options_cross.cross_check = true;
  bf_options_cross.min_cossim = -1;
  auto matcher_cross =
      CreateBruteForceMatcher(CreateFeatureMatcherOptions(), bf_options_cross);

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches_no_cross, matches_cross;
  matcher_no_cross->Match(image1, image2, &matches_no_cross);
  matcher_cross->Match(image1, image2, &matches_cross);

  // Cross-check should filter some matches.
  EXPECT_GT(matches_no_cross.size(), 0);
  EXPECT_LT(matches_cross.size(), matches_no_cross.size());
}

TEST_F(BruteForceONNXMatcherTest, MinCossim) {
  constexpr int kNumDescriptors = 20;

  auto keypoints1 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors1 = CreateRandomDescriptors(kNumDescriptors);
  auto keypoints2 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors2 = CreateRandomDescriptors(kNumDescriptors);

  BruteForceONNXMatchingOptions bf_options_low =
      CreateBruteForceONNXMatchingOptions();
  bf_options_low.max_ratio = 0;
  bf_options_low.cross_check = false;
  bf_options_low.min_cossim = -1;  // Accept all.
  auto matcher_low =
      CreateBruteForceMatcher(CreateFeatureMatcherOptions(), bf_options_low);

  BruteForceONNXMatchingOptions bf_options_high =
      CreateBruteForceONNXMatchingOptions();
  bf_options_high.max_ratio = 0;
  bf_options_high.cross_check = false;
  bf_options_high.min_cossim = 0.9;  // Very strict.
  auto matcher_high =
      CreateBruteForceMatcher(CreateFeatureMatcherOptions(), bf_options_high);

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches_low, matches_high;
  matcher_low->Match(image1, image2, &matches_low);
  matcher_high->Match(image1, image2, &matches_high);

  // Higher min_cossim should filter more matches.
  EXPECT_GT(matches_low.size(), 0);
  EXPECT_LT(matches_high.size(), matches_low.size());
}

TEST_F(BruteForceONNXMatcherTest, TooFewDescriptors) {
  // Model requires at least 2 descriptors for ratio test.
  auto keypoints1 = CreateDummyKeypoints(1);
  auto descriptors1 = CreateRandomDescriptors(1);
  auto keypoints2 = CreateDummyKeypoints(10);
  auto descriptors2 = CreateRandomDescriptors(10);

  auto matcher = CreateBruteForceMatcher(CreateFeatureMatcherOptions(),
                                         CreateBruteForceONNXMatchingOptions());

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches12;
  matcher->Match(image1, image2, &matches12);
  EXPECT_EQ(matches12.size(), 0);

  FeatureMatches matches21;
  matcher->Match(image2, image1, &matches21);
  EXPECT_EQ(matches21.size(), 0);
}

TEST_F(BruteForceONNXMatcherTest, EmptyDescriptors) {
  auto keypoints1 = CreateDummyKeypoints(0);
  auto descriptors1 = CreateRandomDescriptors(0);
  auto keypoints2 = CreateDummyKeypoints(10);
  auto descriptors2 = CreateRandomDescriptors(10);

  auto matcher = CreateBruteForceMatcher(CreateFeatureMatcherOptions(),
                                         CreateBruteForceONNXMatchingOptions());

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  // Should return empty matches without crashing.
  EXPECT_EQ(matches.size(), 0);
}

#ifdef COLMAP_COREML_ENABLED
TEST_F(BruteForceONNXMatcherTest, CoreMLAutomaticallyUsesCPU) {
  constexpr int kNumDescriptors = 20;

  auto keypoints1 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors1 = CreateRandomDescriptors(kNumDescriptors);
  auto keypoints2 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors2 = CreateRandomDescriptors(kNumDescriptors);

  FeatureMatchingOptions options = CreateFeatureMatcherOptions();
  options.use_gpu = true;
  BruteForceONNXMatchingOptions bf_options =
      CreateBruteForceONNXMatchingOptions();
  bf_options.min_cossim = 1.0;
  bf_options.max_ratio = 0.0;
  bf_options.cross_check = false;
  auto matcher = CreateBruteForceMatcher(options, bf_options);

  FeatureMatcher::Image image1{1, nullptr, keypoints1, descriptors1};
  FeatureMatcher::Image image2{2, nullptr, keypoints2, descriptors2};
  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  EXPECT_TRUE(matches.empty());
}
#endif

TEST_F(BruteForceONNXMatcherTest, Caching) {
  constexpr int kNumDescriptors = 10;

  // Create three sets of descriptors for images A, B, C.
  auto keypointsA = CreateDummyKeypoints(kNumDescriptors);
  auto descriptorsA = CreateRandomDescriptors(kNumDescriptors);
  auto keypointsB = CreateDummyKeypoints(kNumDescriptors);
  auto descriptorsB = CreateRandomDescriptors(kNumDescriptors);
  auto keypointsC = CreateDummyKeypoints(kNumDescriptors);
  auto descriptorsC = CreateRandomDescriptors(kNumDescriptors);

  BruteForceONNXMatchingOptions bf_options =
      CreateBruteForceONNXMatchingOptions();
  bf_options.max_ratio = 0;
  bf_options.cross_check = false;
  bf_options.min_cossim = -1;
  auto matcher =
      CreateBruteForceMatcher(CreateFeatureMatcherOptions(), bf_options);

  FeatureMatcher::Image imageA{1, nullptr, keypointsA, descriptorsA};
  FeatureMatcher::Image imageB{2, nullptr, keypointsB, descriptorsB};
  FeatureMatcher::Image imageC{3, nullptr, keypointsC, descriptorsC};

  // Match (A, B).
  FeatureMatches matchesAB1;
  matcher->Match(imageA, imageB, &matchesAB1);
  EXPECT_GT(matchesAB1.size(), 0);

  // Match (A, B) again - should use cached features and produce same results.
  FeatureMatches matchesAB2;
  matcher->Match(imageA, imageB, &matchesAB2);
  ASSERT_EQ(matchesAB1.size(), matchesAB2.size());
  for (size_t i = 0; i < matchesAB1.size(); ++i) {
    EXPECT_EQ(matchesAB1[i].point2D_idx1, matchesAB2[i].point2D_idx1);
    EXPECT_EQ(matchesAB1[i].point2D_idx2, matchesAB2[i].point2D_idx2);
  }

  // Match (B, C) - B should be swapped from slot 2 to slot 1.
  FeatureMatches matchesBC;
  matcher->Match(imageB, imageC, &matchesBC);
  EXPECT_GT(matchesBC.size(), 0);

  // Match (A, B) again - verify correctness after swap.
  FeatureMatches matchesAB3;
  matcher->Match(imageA, imageB, &matchesAB3);
  ASSERT_EQ(matchesAB1.size(), matchesAB3.size());
  for (size_t i = 0; i < matchesAB1.size(); ++i) {
    EXPECT_EQ(matchesAB1[i].point2D_idx1, matchesAB3[i].point2D_idx1);
    EXPECT_EQ(matchesAB1[i].point2D_idx2, matchesAB3[i].point2D_idx2);
  }

  // Self-match A - verify self-matching still works.
  FeatureMatches matchesAA;
  matcher->Match(imageA, imageA, &matchesAA);
  EXPECT_EQ(matchesAA.size(), kNumDescriptors);
  for (const auto& match : matchesAA) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
  }

  // Concurrent Match() calls on cold image IDs deduplicate feature creation.
  const FeatureMatcher::Image coldA{10, nullptr, keypointsA, descriptorsA};
  const FeatureMatcher::Image coldB{11, nullptr, keypointsB, descriptorsB};
  std::vector<std::future<FeatureMatches>> futures;
  futures.reserve(4);
  for (int t = 0; t < 4; ++t) {
    futures.push_back(std::async(std::launch::async, [&]() {
      FeatureMatches out;
      matcher->Match(coldA, coldB, &out);
      return out;
    }));
  }
  for (auto& future : futures) {
    EXPECT_EQ(future.get(), matchesAB1);
  }
}

TEST(ONNXMatchSchedulerTest, SmallGpuStaysAtM1AndScalesDecodeWorkers) {
  constexpr size_t kGiB = 1ULL << 30;
  constexpr size_t kMiB = 1ULL << 20;
  size_t free_vram = 3300 * kMiB;
  const size_t total_vram = 4 * kGiB;

  ONNXMatchScheduler::Config config;
  config.adaptive_gpu = true;
  config.max_workers_cap = 8;
  config.max_num_keypoints = 2048;
  config.ram_per_worker_bytes = 128 * kMiB;
  config.min_slot_vram_bytes = 1024 * kMiB;
  config.avail_ram_fn = []() { return std::optional<size_t>(16 * kGiB); };
  config.avail_vram_fn = [&](size_t* free_b, size_t* total_b) {
    *free_b = free_vram;
    *total_b = total_vram;
    return true;
  };
  config.cache_vram_fn = []() {
    return ONNXMatchScheduler::CacheVramUsage{32 * kMiB, 480 * kMiB};
  };

  ONNXMatchScheduler scheduler(config);
  EXPECT_EQ(scheduler.MaxConcurrentInferences(), 1);
  EXPECT_EQ(scheduler.ActiveWorkersLimit(), 2);

  {
    const auto slot = scheduler.AcquireWorkerSlot();
    // Simulate M=1 activation arena allocating 1.2 GiB on the 4 GiB GPU.
    free_vram = 2100 * kMiB;
  }

  // Heavy CPU preprocessing (cold JPEG decode: 20 ms prep vs 10 ms inference)
  // keeps M=1 due to limited VRAM, while scaling N to 3 to keep the GPU fed.
  for (int i = 0; i < 8; ++i) {
    scheduler.RecordPair(/*prep_ns=*/20'000'000,
                         /*inf_ns=*/10'000'000,
                         /*num_runs=*/2,
                         /*pair_max_keypoints=*/2048);
  }
  EXPECT_EQ(scheduler.MaxConcurrentInferences(), 1);
  EXPECT_EQ(scheduler.ActiveWorkersLimit(), 3);

  // Once caches warm up (0.5 ms prep vs 10 ms inference), N relaxes to 2.
  for (int i = 0; i < 30; ++i) {
    scheduler.RecordPair(/*prep_ns=*/500'000,
                         /*inf_ns=*/10'000'000,
                         /*num_runs=*/1,
                         /*pair_max_keypoints=*/2048);
  }
  EXPECT_EQ(scheduler.MaxConcurrentInferences(), 1);
  EXPECT_EQ(scheduler.ActiveWorkersLimit(), 2);
}

TEST(ONNXMatchSchedulerTest, LargeGpuRampsMTo3AndAdaptsWorkers) {
  constexpr size_t kGiB = 1ULL << 30;
  size_t free_vram = 38 * kGiB;
  const size_t total_vram = 40 * kGiB;

  ONNXMatchScheduler::Config config;
  config.adaptive_gpu = true;
  config.max_workers_cap = 8;
  config.max_num_keypoints = 4096;
  config.ram_per_worker_bytes = 128ULL << 20;
  config.min_slot_vram_bytes = 1 * kGiB;
  config.avail_ram_fn = []() { return std::optional<size_t>(32 * kGiB); };
  config.avail_vram_fn = [&](size_t* free_b, size_t* total_b) {
    *free_b = free_vram;
    *total_b = total_vram;
    return true;
  };

  ONNXMatchScheduler scheduler(config);
  {
    const auto slot = scheduler.AcquireWorkerSlot();
    const auto p1 = scheduler.inference_limiter()->Acquire();
    free_vram = 37 * kGiB;
  }

  // Cold-decode phase (20 ms prep, 20 ms inf): M steps 1 -> 2 -> 3, N -> 6.
  for (int i = 0; i < 4; ++i) {
    scheduler.RecordPair(20'000'000, 20'000'000, 2, 4096);
  }
  EXPECT_EQ(scheduler.MaxConcurrentInferences(), 2);
  {
    const auto p1 = scheduler.inference_limiter()->Acquire();
    const auto p2 = scheduler.inference_limiter()->Acquire();
    free_vram = 36 * kGiB;
  }
  for (int i = 0; i < 4; ++i) {
    scheduler.RecordPair(20'000'000, 20'000'000, 2, 4096);
  }
  EXPECT_EQ(scheduler.MaxConcurrentInferences(), 3);
  EXPECT_EQ(scheduler.ActiveWorkersLimit(), 6);

  // Warm-cache phase (1 ms prep, 20 ms inf): N relaxes from 6 to 4 (= M + 1).
  for (int i = 0; i < 30; ++i) {
    scheduler.RecordPair(1'000'000, 20'000'000, 1, 4096);
  }
  EXPECT_EQ(scheduler.MaxConcurrentInferences(), 3);
  EXPECT_EQ(scheduler.ActiveWorkersLimit(), 4);
}

TEST(ONNXMatchSchedulerTest, CacheReservationRamBoundsAndOomBackoff) {
  constexpr size_t kMiB = 1ULL << 20;
  size_t avail_ram = 768 * kMiB;  // 512 MiB headroom + 256 MiB => 2 workers.
  size_t free_vram = 2200 * kMiB;
  const size_t total_vram =
      40960 * kMiB;  // Shared 40 GiB GPU with 2.2 GiB free.
  size_t unallocated_cache = 1200 * kMiB;

  ONNXMatchScheduler::Config config;
  config.adaptive_gpu = true;
  config.max_workers_cap = 8;
  config.max_num_keypoints = 2048;
  config.ram_per_worker_bytes = 128 * kMiB;
  config.min_slot_vram_bytes = 600 * kMiB;
  config.avail_ram_fn = [&]() { return std::optional<size_t>(avail_ram); };
  config.avail_vram_fn = [&](size_t* free_b, size_t* total_b) {
    *free_b = free_vram;
    *total_b = total_vram;
    return true;
  };
  config.cache_vram_fn = [&]() {
    return ONNXMatchScheduler::CacheVramUsage{0, unallocated_cache};
  };

  ONNXMatchScheduler scheduler(config);
  EXPECT_EQ(scheduler.ActiveWorkersLimit(), 2);
  {
    const auto p1 = scheduler.inference_limiter()->Acquire();
  }

  // Unallocated GPU cache reservation blocks M from stepping to 2, and low
  // host RAM clamps N at 2 despite high t_prep.
  for (int i = 0; i < 4; ++i) {
    scheduler.RecordPair(30'000'000, 10'000'000, 1, 2048);
  }
  EXPECT_EQ(scheduler.MaxConcurrentInferences(), 1);
  EXPECT_EQ(scheduler.ActiveWorkersLimit(), 2);

  // Freeing cache reservation + host RAM lets M step to 2 (holding at 2 until
  // 2 concurrent inferences actually materialize), then OOM locks M=1.
  unallocated_cache = 0;
  avail_ram = 8192 * kMiB;
  for (int i = 0; i < 12; ++i) {
    scheduler.RecordPair(30'000'000, 10'000'000, 1, 2048);
  }
  EXPECT_EQ(scheduler.MaxConcurrentInferences(), 2);
  EXPECT_GT(scheduler.ActiveWorkersLimit(), 2);

  scheduler.inference_limiter()->OnOutOfMemory(/*limit_at_acquire=*/2);
  for (int i = 0; i < 8; ++i) {
    scheduler.RecordPair(1'000'000, 10'000'000, 1, 2048);
  }
  EXPECT_EQ(scheduler.MaxConcurrentInferences(), 1);

  FeatureMatchingOptions cpu_opts(FeatureMatcherType::ALIKED_LIGHTGLUE);
  EXPECT_EQ(ONNXMatchScheduler::DefaultConfig(cpu_opts).max_num_keypoints,
            2048);
  cpu_opts.use_gpu = false;
  EXPECT_EQ(ComputeMaxONNXWorkersPerGpu(cpu_opts, /*num_gpus=*/1), 1);
}

}  // namespace
}  // namespace colmap
