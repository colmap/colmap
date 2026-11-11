// SPDX-License-Identifier: BSD-3-Clause

// Determinism of the cusift extractor and matcher: the same input must give
// bit-identical output on every call, from fresh and from reused instances.
//
// SiftGPU does not have this property: its detector can return different
// keypoint sets for the same image from run to run (colmap#1197). Cross-process
// repetition, the case colmap#1197 reports, is covered by
// src/pycolmap/feature/cusift_determinism_test.py.

#include "colmap/feature/cusift.h"
#include "colmap/feature/cusift_matcher.h"
#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/sift_test_utils.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/cuda.h"

#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

constexpr int kNumRepeats = 5;

FeatureMatchingOptions CreateCusiftMatchingOptions() {
  FeatureMatchingOptions options(FeatureMatcherType::CUSIFT_GEMM);
  options.use_gpu = true;
  return options;
}

struct Features {
  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
};

Features Extract(FeatureExtractor& extractor, const Bitmap& bitmap) {
  Features features;
  THROW_CHECK(
      extractor.Extract(bitmap, &features.keypoints, &features.descriptors));
  return features;
}

// Exact comparison, no tolerance: every float of every keypoint and every
// descriptor byte must match.
void ExpectIdentical(const Features& expected, const Features& actual) {
  ASSERT_EQ(expected.keypoints.size(), actual.keypoints.size());
  for (size_t i = 0; i < expected.keypoints.size(); ++i) {
    const FeatureKeypoint& a = expected.keypoints[i];
    const FeatureKeypoint& b = actual.keypoints[i];
    ASSERT_TRUE(a.x == b.x && a.y == b.y && a.a11 == b.a11 && a.a12 == b.a12 &&
                a.a21 == b.a21 && a.a22 == b.a22)
        << "keypoint " << i << " differs";
  }
  EXPECT_EQ(expected.descriptors.type, actual.descriptors.type);
  ASSERT_EQ(expected.descriptors.data.rows(), actual.descriptors.data.rows());
  ASSERT_EQ(expected.descriptors.data.cols(), actual.descriptors.data.cols());
  EXPECT_TRUE(expected.descriptors.data == actual.descriptors.data);
}

FeatureMatcher::Image MakeImage(const image_t image_id,
                                const Features& features) {
  return FeatureMatcher::Image{
      image_id,
      /*camera=*/nullptr,
      std::make_shared<FeatureKeypoints>(features.keypoints),
      std::make_shared<FeatureDescriptors>(features.descriptors)};
}

FeatureMatches Match(FeatureMatcher& matcher,
                     const FeatureMatcher::Image& image1,
                     const FeatureMatcher::Image& image2) {
  FeatureMatches matches;
  matcher.Match(image1, image2, &matches);
  return matches;
}

#if defined(COLMAP_CUSIFT_ENABLED)

TEST(CusiftDeterminism, ExtractionWithFreshExtractors) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  const Bitmap bitmap = CreateTexturedImage(640, 480);

  auto reference_extractor =
      CreateCusiftFeatureExtractor(CreateCusiftOptions());
  ASSERT_NE(reference_extractor, nullptr);
  const Features reference = Extract(*reference_extractor, bitmap);
  // Guard against a vacuous pass: identical output is only meaningful if
  // there is a substantial amount of it.
  ASSERT_GE(reference.keypoints.size(), 200);

  for (int i = 0; i < kNumRepeats; ++i) {
    SCOPED_TRACE(i);
    auto extractor = CreateCusiftFeatureExtractor(CreateCusiftOptions());
    ASSERT_NE(extractor, nullptr);
    ExpectIdentical(reference, Extract(*extractor, bitmap));
  }
}

TEST(CusiftDeterminism, ExtractionWithReusedExtractor) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // cusift keeps grow-only device and pinned host buffers across calls.
  // Interleaving a different, larger image checks that no state from one call
  // leaks into the next.
  const Bitmap bitmap = CreateTexturedImage(640, 480);
  const Bitmap other = CreateTexturedImage(800, 600, /*dx=*/13, /*dy=*/7);

  auto extractor = CreateCusiftFeatureExtractor(CreateCusiftOptions());
  ASSERT_NE(extractor, nullptr);
  const Features reference = Extract(*extractor, bitmap);
  ASSERT_GE(reference.keypoints.size(), 200);

  for (int i = 0; i < kNumRepeats; ++i) {
    SCOPED_TRACE(i);
    Extract(*extractor, other);
    ExpectIdentical(reference, Extract(*extractor, bitmap));
  }
}

// Releases its threads together, round after round (std::barrier is C++20).
class RoundBarrier {
 public:
  explicit RoundBarrier(const int num_threads) : num_threads_(num_threads) {}

  void ArriveAndWait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const int round = round_;
    if (++num_arrived_ == num_threads_) {
      num_arrived_ = 0;
      ++round_;
      released_.notify_all();
      return;
    }
    released_.wait(lock, [&]() { return round_ != round; });
  }

 private:
  const int num_threads_;
  int num_arrived_ = 0;
  int round_ = 0;
  std::mutex mutex_;
  std::condition_variable released_;
};

constexpr int kNumOverlappingRounds = 10;

// Extracts `bitmap` once per round, each round released together with the
// other threads at `barrier`. An error skips the remaining extractions but not
// the barrier, so no other thread is left waiting; it is rethrown at the end.
std::vector<Features> ExtractInRounds(const FeatureExtractionOptions& options,
                                      const Bitmap& bitmap,
                                      RoundBarrier* barrier) {
  std::vector<Features> results;
  std::exception_ptr error;
  std::unique_ptr<FeatureExtractor> extractor;
  try {
    extractor = THROW_CHECK_NOTNULL(CreateCusiftFeatureExtractor(options));
  } catch (...) {
    error = std::current_exception();
  }
  for (int round = 0; round < kNumOverlappingRounds; ++round) {
    barrier->ArriveAndWait();
    if (error) continue;
    try {
      results.push_back(Extract(*extractor, bitmap));
    } catch (...) {
      error = std::current_exception();
    }
  }
  if (error) std::rethrow_exception(error);
  return results;
}

TEST(CusiftDeterminism, ExtractionUnaffectedByOtherThreadsOnTheSameGpu) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // The blur weights live in one constant bank per GPU. Threads on the same
  // GPU that use different settings (here first_octave 0 vs -1, which give
  // different blur sigmas) must not overwrite the weights the other uses.
  // Threads A and B extract at the same time, round after round.
  const Bitmap bitmap = CreateTexturedImage(640, 480);
  FeatureExtractionOptions options_a = CreateCusiftOptions();
  options_a.sift->first_octave = 0;
  FeatureExtractionOptions options_b = CreateCusiftOptions();
  options_b.sift->first_octave = -1;
  const Features reference_a = Extract(
      *THROW_CHECK_NOTNULL(CreateCusiftFeatureExtractor(options_a)), bitmap);
  const Features reference_b = Extract(
      *THROW_CHECK_NOTNULL(CreateCusiftFeatureExtractor(options_b)), bitmap);
  ASSERT_GE(reference_a.keypoints.size(), 200);
  ASSERT_GE(reference_b.keypoints.size(), 200);

  RoundBarrier barrier(/*num_threads=*/2);
  std::future<std::vector<Features>> thread_a = std::async(std::launch::async,
                                                           ExtractInRounds,
                                                           std::cref(options_a),
                                                           std::cref(bitmap),
                                                           &barrier);
  std::future<std::vector<Features>> thread_b = std::async(std::launch::async,
                                                           ExtractInRounds,
                                                           std::cref(options_b),
                                                           std::cref(bitmap),
                                                           &barrier);
  const std::vector<Features> results_a = thread_a.get();
  const std::vector<Features> results_b = thread_b.get();
  ASSERT_EQ(results_a.size(), kNumOverlappingRounds);
  ASSERT_EQ(results_b.size(), kNumOverlappingRounds);

  for (int round = 0; round < kNumOverlappingRounds; ++round) {
    SCOPED_TRACE(round);
    ExpectIdentical(reference_a, results_a[round]);
    ExpectIdentical(reference_b, results_b[round]);
  }
}

class CusiftMatchingDeterminism : public ::testing::Test {
 protected:
  void SetUp() override {
    if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
    auto extractor = CreateCusiftFeatureExtractor(CreateCusiftOptions());
    ASSERT_NE(extractor, nullptr);
    features1_ = Extract(*extractor, CreateTexturedImage(640, 480));
    features2_ =
        Extract(*extractor, CreateTexturedImage(640, 480, /*dx=*/11, /*dy=*/5));
    unrelated_ = Extract(*extractor, CreateTexturedImage(480, 640, 200, 150));
  }

  Features features1_;
  Features features2_;
  Features unrelated_;
};

TEST_F(CusiftMatchingDeterminism, FreshMatchers) {
  const FeatureMatcher::Image image1 = MakeImage(1, features1_);
  const FeatureMatcher::Image image2 = MakeImage(2, features2_);

  auto reference_matcher =
      CreateCusiftFeatureMatcher(CreateCusiftMatchingOptions());
  ASSERT_NE(reference_matcher, nullptr);
  const FeatureMatches reference = Match(*reference_matcher, image1, image2);
  ASSERT_GE(reference.size(), 50);

  for (int i = 0; i < kNumRepeats; ++i) {
    SCOPED_TRACE(i);
    auto matcher = CreateCusiftFeatureMatcher(CreateCusiftMatchingOptions());
    ASSERT_NE(matcher, nullptr);
    // Same matches in the same order, not merely the same set.
    EXPECT_EQ(reference, Match(*matcher, image1, image2));
  }
}

TEST_F(CusiftMatchingDeterminism, ReusedMatcher) {
  // The GEMM matcher reuses grow-only buffers across pairs. Matching a pair
  // of a different size in between must not change the next result.
  const FeatureMatcher::Image image1 = MakeImage(1, features1_);
  const FeatureMatcher::Image image2 = MakeImage(2, features2_);
  const FeatureMatcher::Image image3 = MakeImage(3, unrelated_);

  auto matcher = CreateCusiftFeatureMatcher(CreateCusiftMatchingOptions());
  ASSERT_NE(matcher, nullptr);
  const FeatureMatches reference = Match(*matcher, image1, image2);
  ASSERT_GE(reference.size(), 50);

  for (int i = 0; i < kNumRepeats; ++i) {
    SCOPED_TRACE(i);
    Match(*matcher, image3, image1);
    EXPECT_EQ(reference, Match(*matcher, image1, image2));
  }
}

#endif  // COLMAP_CUSIFT_ENABLED

}  // namespace
}  // namespace colmap
