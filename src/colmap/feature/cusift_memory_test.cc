// SPDX-License-Identifier: BSD-3-Clause

// cusift's memory handling, in three parts:
// - Each extractor and matcher owns its GPU buffers, pinned host buffers and
//   cuBLAS handle, and COLMAP creates new ones, on new threads, for every run,
//   so what one allocated must be released when it is destroyed rather than
//   accumulate run after run. (Not thread_local: on Windows every CUDA call
//   made from a thread-exit destructor fails, which leaked these.)
// - It allocates from its own stream-ordered memory pool, so the device's
//   default pool, which other code in the process may use, keeps its settings.
// - Destroying an extractor returns the pool memory it no longer uses to the
//   driver, so GPU stages that run after extraction can have it.
// - An extractor's buffers, pool and constant bank are per device, so Extract
//   runs on the device it was created on, whatever the calling thread has
//   current; likewise Match for a matcher's cuBLAS handle and buffers.

#include "colmap/feature/cusift.h"
#include "colmap/feature/cusift_matcher.h"
#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/math/random.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/cuda.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "cudaImage.h"
#include "cusift_device.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

constexpr int kNumThreads = 5;
constexpr int64_t kMiB = 1024 * 1024;

void RunOnFreshThread(const std::function<void()>& fn) {
  std::thread thread(fn);
  thread.join();
}

// Bytes currently allocated from cusift's memory pool. Its extraction scratch
// is allocated there, so scratch that is never freed keeps this count up
// after its extractor is destroyed.
int64_t CusiftPoolUsedBytes() {
  return static_cast<int64_t>(cusift::GetMemoryPoolUsage().used);
}

// As CusiftPoolUsedBytes, once the stream-ordered frees queued so far have run.
int64_t SettledPoolUsedBytes() {
  THROW_CHECK_EQ(cudaStreamSynchronize(0), cudaSuccess);
  return CusiftPoolUsedBytes();
}

cudaMemPool_t DefaultPool() {
  int device = 0;
  THROW_CHECK_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaMemPool_t pool = nullptr;
  THROW_CHECK_EQ(cudaDeviceGetDefaultMemPool(&pool, device), cudaSuccess);
  return pool;
}

uint64_t GetPoolAttribute(cudaMemPool_t pool, cudaMemPoolAttr attribute) {
  uint64_t value = 0;
  THROW_CHECK_EQ(cudaMemPoolGetAttribute(pool, attribute, &value), cudaSuccess);
  return value;
}

int64_t FreeDeviceBytes() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  THROW_CHECK_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
  return static_cast<int64_t>(free_bytes);
}

int64_t Median(std::vector<int64_t> values) {
  THROW_CHECK(!values.empty());
  const auto middle = values.begin() + values.size() / 2;
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}

// Runs `fn` once to warm up (CUDA context, memory pool, cuBLAS library state),
// then kNumThreads more times, each on a fresh thread, and returns the median
// growth of `measure_used` per thread. A leak shows up as the size of one
// thread's buffers on every thread; noise from other processes on a shared
// GPU does not, so the median separates the two.
int64_t MedianGrowthPerThread(const std::function<void()>& fn,
                              const std::function<int64_t()>& measure_used) {
  RunOnFreshThread(fn);
  std::vector<int64_t> growth;
  for (int i = 0; i < kNumThreads; ++i) {
    const int64_t before = measure_used();
    RunOnFreshThread(fn);
    growth.push_back(measure_used() - before);
  }
  return Median(growth);
}

Bitmap CreateNoiseImage(const int width, const int height) {
  SetPRNGSeed(0);
  Bitmap bitmap(width, height, /*as_rgb=*/false);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      bitmap.SetPixel(
          x, y, BitmapColor<uint8_t>(RandomUniformInteger<int>(0, 255)));
    }
  }
  return bitmap;
}

FeatureExtractionOptions CreateLargeExtractionOptions() {
  FeatureExtractionOptions options(FeatureExtractorType::CUSIFT);
  options.use_gpu = true;
  options.sift->max_num_features = 32768;
  return options;
}

FeatureDescriptors CreateRandomDescriptors(const int num_features) {
  SetPRNGSeed(1);
  FeatureDescriptorsData data(num_features, 128);
  for (int i = 0; i < num_features; ++i) {
    for (int j = 0; j < 128; ++j) {
      data(i, j) = static_cast<uint8_t>(RandomUniformInteger<int>(0, 60));
    }
  }
  return FeatureDescriptors(FeatureExtractorType::SIFT, std::move(data));
}

// COLMAP's controllers create and extract on one thread, bound at creation to
// the extractor's GPU; direct and pycolmap callers may extract on another
// thread, with another device current. With one GPU this only shows that such
// a call works; with several it checks that Extract rebinds its own device.
TEST(CusiftDevice, ExtractRunsOnTheExtractorsDevice) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  const Bitmap bitmap = CreateNoiseImage(128, 128);
  FeatureExtractionOptions options(FeatureExtractorType::CUSIFT);
  options.use_gpu = true;

  std::unique_ptr<FeatureExtractor> extractor;
  int created_on = -1;
  RunOnFreshThread([&]() {
    extractor = THROW_CHECK_NOTNULL(CreateCusiftFeatureExtractor(options));
    THROW_CHECK_EQ(cudaGetDevice(&created_on), cudaSuccess);
  });

  const int other_device = (created_on + 1) % GetNumCudaDevices();
  int extracted_on = -1;
  RunOnFreshThread([&]() {
    THROW_CHECK_EQ(cudaSetDevice(other_device), cudaSuccess);
    FeatureKeypoints keypoints;
    THROW_CHECK(
        extractor->Extract(bitmap, &keypoints, /*descriptors=*/nullptr));
    THROW_CHECK_GT(keypoints.size(), 0);
    THROW_CHECK_EQ(cudaGetDevice(&extracted_on), cudaSuccess);
  });
  EXPECT_EQ(extracted_on, created_on);

  // The destructor trims the extractor's own pool, whatever device is current.
  int destroyed_with_current = -1;
  int64_t reserved_after = -1;
  RunOnFreshThread([&]() {
    THROW_CHECK_EQ(cudaSetDevice(other_device), cudaSuccess);
    extractor.reset();
    THROW_CHECK_EQ(cudaGetDevice(&destroyed_with_current), cudaSuccess);
    THROW_CHECK_EQ(cudaSetDevice(created_on), cudaSuccess);
    reserved_after =
        static_cast<int64_t>(cusift::GetMemoryPoolUsage().reserved);
  });
  EXPECT_EQ(destroyed_with_current, other_device);
  EXPECT_EQ(reserved_after, 0);
}

// As ExtractRunsOnTheExtractorsDevice, for the matcher's cuBLAS handle and
// buffers, which it frees on its own device: it may be destroyed on any thread.
TEST(CusiftDevice, MatchRunsOnTheMatchersDevice) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  // Rows with disjoint supports, each near COLMAP's descriptor norm of 512, so
  // every row self-matches and the call does real work.
  FeatureDescriptorsData data = FeatureDescriptorsData::Zero(12, 128);
  for (int i = 0; i < data.rows(); ++i) {
    data.row(i).segment(10 * i, 10).setConstant(162);
  }
  const auto descriptors = std::make_shared<FeatureDescriptors>(
      FeatureExtractorType::SIFT, std::move(data));
  const FeatureMatcher::Image image = {
      /*image_id=*/1, &camera, /*keypoints=*/nullptr, descriptors};
  FeatureMatchingOptions options(FeatureMatcherType::CUSIFT_GEMM);
  options.use_gpu = true;

  std::unique_ptr<FeatureMatcher> matcher;
  int created_on = -1;
  RunOnFreshThread([&]() {
    matcher = THROW_CHECK_NOTNULL(FeatureMatcher::Create(options));
    THROW_CHECK_EQ(cudaGetDevice(&created_on), cudaSuccess);
  });

  int matched_on = -1;
  RunOnFreshThread([&]() {
    const int other_device = (created_on + 1) % GetNumCudaDevices();
    THROW_CHECK_EQ(cudaSetDevice(other_device), cudaSuccess);
    FeatureMatches matches;
    matcher->Match(image, image, &matches);
    THROW_CHECK_GT(matches.size(), 0);
    THROW_CHECK_EQ(cudaGetDevice(&matched_on), cudaSuccess);
  });
  EXPECT_EQ(matched_on, created_on);
  matcher.reset();
}

TEST(CusiftMemory, ExtractorThreadReleasesScratchMemory) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // Measure on the device the work runs on (both pick it with gpu_index -1).
  BindCusiftDevice("-1");
  // A noise image yields thousands of keypoints, so the extractor's keypoint
  // and descriptor scratch is several MiB of pool memory.
  const Bitmap bitmap = CreateNoiseImage(2048, 2048);
  const FeatureExtractionOptions options = CreateLargeExtractionOptions();

  // Positive control: while the extractor lives, the scratch the test expects
  // back is in the pool.
  int64_t min_used_while_alive = INT64_MAX;
  const int64_t growth = MedianGrowthPerThread(
      [&]() {
        auto extractor = THROW_CHECK_NOTNULL(FeatureExtractor::Create(options));
        FeatureKeypoints keypoints;
        FeatureDescriptors descriptors;
        THROW_CHECK(extractor->Extract(bitmap, &keypoints, &descriptors));
        min_used_while_alive =
            std::min(min_used_while_alive, CusiftPoolUsedBytes());
      },
      CusiftPoolUsedBytes);
  EXPECT_GT(min_used_while_alive, kMiB);
  EXPECT_LT(growth, kMiB);
}

TEST(CusiftMemoryPool, LeavesTheDefaultPoolUntouched) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  const Bitmap bitmap = CreateNoiseImage(2048, 2048);
  RunOnFreshThread([&]() {
    // Binds this thread to the GPU cusift picks with gpu_index -1, without
    // cusift code, so the baseline is taken before any cusift work on it.
    SetBestCudaDevice(-1);
    const cudaMemPool_t pool = DefaultPool();
    const uint64_t threshold_before =
        GetPoolAttribute(pool, cudaMemPoolAttrReleaseThreshold);
    const uint64_t used_before =
        GetPoolAttribute(pool, cudaMemPoolAttrUsedMemCurrent);
    auto extractor = THROW_CHECK_NOTNULL(
        FeatureExtractor::Create(CreateLargeExtractionOptions()));
    FeatureKeypoints keypoints;
    FeatureDescriptors descriptors;
    THROW_CHECK(extractor->Extract(bitmap, &keypoints, &descriptors));
    // cusift neither raises the default pool's release threshold nor
    // allocates from it.
    EXPECT_EQ(GetPoolAttribute(pool, cudaMemPoolAttrReleaseThreshold),
              threshold_before);
    // Earlier tests in this process may already have created cusift's pool, so
    // check the baseline against the driver's default release threshold.
    EXPECT_EQ(threshold_before, 0u);
    EXPECT_EQ(GetPoolAttribute(pool, cudaMemPoolAttrUsedMemCurrent),
              used_before);
  });
}

TEST(CusiftMemoryPool, ExtractorReturnsUnusedMemoryWhenDestroyed) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // Measure on the device the work runs on (both pick it with gpu_index -1).
  BindCusiftDevice("-1");
  const Bitmap bitmap = CreateNoiseImage(2048, 2048);
  int64_t reserved_while_alive = 0;
  int64_t reserved_after = 0;
  RunOnFreshThread([&]() {
    auto extractor = THROW_CHECK_NOTNULL(
        FeatureExtractor::Create(CreateLargeExtractionOptions()));
    FeatureKeypoints keypoints;
    FeatureDescriptors descriptors;
    THROW_CHECK(extractor->Extract(bitmap, &keypoints, &descriptors));
    reserved_while_alive =
        static_cast<int64_t>(cusift::GetMemoryPoolUsage().reserved);
    extractor.reset();
    reserved_after =
        static_cast<int64_t>(cusift::GetMemoryPoolUsage().reserved);
  });
  // The pyramid of a 2048 x 2048 image at first_octave -1 is hundreds of MiB,
  // all kept for reuse while the extractor lives...
  EXPECT_GT(reserved_while_alive, 128 * kMiB);
  // ...and handed back to the driver once it is destroyed.
  EXPECT_LT(reserved_after, reserved_while_alive / 4);
}

// A CudaImage owns at most one buffer: allocating again frees the old one.
TEST(CusiftCudaImage, AllocateAgainFreesTheOwnedBuffer) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  BindCusiftDevice("-1");
  const int64_t base = SettledPoolUsedBytes();
  CudaImage image;
  image.Allocate(512, 512, 512);
  const int64_t image_bytes = SettledPoolUsedBytes() - base;
  if (image_bytes == 0) GTEST_SKIP() << "cusift fell back to the default pool";

  image.Allocate(512, 512, 512);

  EXPECT_EQ(SettledPoolUsedBytes() - base, image_bytes);
}

// An owning image re-bound as a view frees its buffer, and never the view's.
TEST(CusiftCudaImage, ViewOfAnotherBufferOwnsNothing) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  BindCusiftDevice("-1");
  const int64_t base = SettledPoolUsedBytes();
  CudaImage owner;
  owner.Allocate(512, 512, 512);
  const int64_t owner_bytes = SettledPoolUsedBytes() - base;
  if (owner_bytes == 0) GTEST_SKIP() << "cusift fell back to the default pool";

  {
    CudaImage image;
    image.Allocate(512, 512, 512);
    image.Allocate(512, 512, 512, owner.d_data);
    EXPECT_EQ(SettledPoolUsedBytes() - base, owner_bytes);
  }

  EXPECT_EQ(SettledPoolUsedBytes() - base, owner_bytes);
}

// Running out of memory fails the image, as SiftGPU does, instead of throwing
// out of the extraction worker; the extractor then works as before. The
// failure is injected at every pool allocation of an extraction in turn, so
// it also hits the scratch buffers when some are already allocated.
TEST(CusiftMemory, ExtractorSkipsImageWhenOutOfMemory) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  const Bitmap bitmap = CreateNoiseImage(256, 256);
  FeatureExtractionOptions options(FeatureExtractorType::CUSIFT);
  options.use_gpu = true;
  FeatureKeypoints expected;
  ASSERT_TRUE(THROW_CHECK_NOTNULL(FeatureExtractor::Create(options))
                  ->Extract(bitmap, &expected, /*descriptors=*/nullptr));

  int num_failed_extractions = 0;
  for (int failing_allocation = 1;; ++failing_allocation) {
    SCOPED_TRACE(failing_allocation);
    ASSERT_LT(failing_allocation, 1000) << "the injected failure never ran out";
    auto extractor = THROW_CHECK_NOTNULL(FeatureExtractor::Create(options));
    cusift::FailAllocationForTesting(failing_allocation);
    FeatureKeypoints keypoints;
    if (extractor->Extract(bitmap, &keypoints, /*descriptors=*/nullptr)) {
      // Past the last allocation: the injected failure did not fire.
      cusift::FailAllocationForTesting(0);
      break;
    }
    ++num_failed_extractions;
    ASSERT_TRUE(
        extractor->Extract(bitmap, &keypoints, /*descriptors=*/nullptr));
    ASSERT_EQ(keypoints.size(), expected.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
      ASSERT_EQ(keypoints[i].x, expected[i].x);
      ASSERT_EQ(keypoints[i].y, expected[i].y);
    }
  }
  // Every extraction allocates at least the input image and the scratch.
  EXPECT_GT(num_failed_extractions, 4);
}

// Any other error fails the image too, instead of ending the worker thread,
// which would leave the extraction controller waiting for it forever. An RGB
// bitmap is one: the extractor only takes grey ones.
TEST(CusiftMemory, ExtractorSkipsImageOnOtherErrors) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  FeatureExtractionOptions options(FeatureExtractorType::CUSIFT);
  options.use_gpu = true;
  const std::unique_ptr<FeatureExtractor> extractor =
      THROW_CHECK_NOTNULL(FeatureExtractor::Create(options));
  FeatureKeypoints keypoints;

  const bool extracted = extractor->Extract(
      Bitmap(256, 256, /*as_rgb=*/true), &keypoints, /*descriptors=*/nullptr);

  EXPECT_FALSE(extracted);
  EXPECT_TRUE(extractor->Extract(
      CreateNoiseImage(256, 256), &keypoints, /*descriptors=*/nullptr));
  EXPECT_GT(keypoints.size(), 0u);
}

// VLFeat keeps at most 4 orientations per keypoint, so an absurd
// max_num_orientations must give the same keypoints as 4. Uncapped, the host
// reservation fails, unless overcommit (mode 1) grants it.
TEST(CusiftMemory, ExtractorCapsOrientationReservation) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  const Bitmap bitmap = CreateNoiseImage(256, 256);
  FeatureExtractionOptions options(FeatureExtractorType::CUSIFT);
  options.use_gpu = true;
  options.sift->max_num_orientations = 4;
  FeatureKeypoints expected;
  ASSERT_TRUE(THROW_CHECK_NOTNULL(FeatureExtractor::Create(options))
                  ->Extract(bitmap, &expected, /*descriptors=*/nullptr));
  options.sift->max_num_orientations = 2'000'000'000;
  FeatureKeypoints keypoints;

  const bool extracted =
      THROW_CHECK_NOTNULL(FeatureExtractor::Create(options))
          ->Extract(bitmap, &keypoints, /*descriptors=*/nullptr);

  EXPECT_TRUE(extracted);
  EXPECT_GT(expected.size(), 0u);
  EXPECT_EQ(keypoints.size(), expected.size());
}

// Median device memory of kNumThreads CUSIFT_GEMM matchers made with `options`,
// robust to other processes briefly allocating on a shared GPU.
int64_t MatcherCreationBytes(const FeatureMatchingOptions& options) {
  std::vector<int64_t> used;
  for (int i = 0; i < kNumThreads; ++i) {
    RunOnFreshThread([&]() {
      BindCusiftDevice("-1");
      const int64_t before = FreeDeviceBytes();
      auto matcher = THROW_CHECK_NOTNULL(FeatureMatcher::Create(options));
      used.push_back(before - FreeDeviceBytes());
    });
  }
  return Median(used);
}

// A direct caller's matcher grows its buffers per pair; only COLMAP's matching
// workers, which clamp max_num_matches to the largest image, reserve upfront.
TEST(CusiftMemory, MatcherReservesUpfrontOnlyWhenAsked) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  FeatureMatchingOptions options(FeatureMatcherType::CUSIFT_GEMM);
  options.use_gpu = true;

  const int64_t direct = MatcherCreationBytes(options);
  options.cusift->reserve_max_num_matches = true;
  const int64_t reserved = MatcherCreationBytes(options);

  // 32768 descriptors per side come to about 100 MiB of buffers.
  EXPECT_LT(direct, 8 * kMiB);
  EXPECT_GT(reserved, 64 * kMiB);
}

TEST(CusiftMemory, MatcherThreadReleasesDeviceMemory) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  // Measure on the device the work runs on (both pick it with gpu_index -1).
  BindCusiftDevice("-1");
  // 8,192 descriptors per side: the similarity tile and the per-side buffers
  // come to well over 64 MiB of device memory per matcher.
  constexpr int kNumFeatures = 8192;
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(
          CreateRandomDescriptors(kNumFeatures))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(
          CreateRandomDescriptors(kNumFeatures))};
  FeatureMatchingOptions options(FeatureMatcherType::CUSIFT_GEMM);
  options.use_gpu = true;
  options.max_num_matches = kNumFeatures;
  constexpr int64_t kLeakBound = 32 * kMiB;

  // Positive control: every matcher finds matches and, while it lives, holds
  // more device memory than the leak bound.
  size_t min_num_matches = SIZE_MAX;
  int64_t min_used_while_alive = INT64_MAX;
  const int64_t growth = MedianGrowthPerThread(
      [&]() {
        SetBestCudaDevice(-1);
        const int64_t free_before = FreeDeviceBytes();
        auto matcher = THROW_CHECK_NOTNULL(FeatureMatcher::Create(options));
        FeatureMatches matches;
        matcher->Match(image1, image2, &matches);
        min_num_matches = std::min(min_num_matches, matches.size());
        min_used_while_alive =
            std::min(min_used_while_alive, free_before - FreeDeviceBytes());
      },
      []() { return -FreeDeviceBytes(); });
  EXPECT_GT(min_num_matches, 0u);
  EXPECT_GT(min_used_while_alive, kLeakBound);
  EXPECT_LT(growth, kLeakBound);
}

}  // namespace
}  // namespace colmap
