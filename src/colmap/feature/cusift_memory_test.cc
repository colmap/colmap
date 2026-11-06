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
  std::nth_element(
      growth.begin(), growth.begin() + kNumThreads / 2, growth.end());
  return growth[kNumThreads / 2];
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

void ExtractOnce(const FeatureExtractionOptions& options,
                 const Bitmap& bitmap) {
  auto extractor = THROW_CHECK_NOTNULL(FeatureExtractor::Create(options));
  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  THROW_CHECK(extractor->Extract(bitmap, &keypoints, &descriptors));
  THROW_CHECK_GT(keypoints.size(), 0);
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
  // Measure on the device the work runs on (both pick it with gpu_index -1).
  BindCusiftDevice("-1");
  const Bitmap bitmap = CreateNoiseImage(2048, 2048);
  RunOnFreshThread([&]() {
    // Binds this thread to the GPU, so DefaultPool() names the pool cusift
    // would have changed.
    ExtractOnce(CreateLargeExtractionOptions(), bitmap);
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
    EXPECT_NE(threshold_before, UINT64_MAX);
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

  const int64_t growth = MedianGrowthPerThread(
      [&]() {
        auto matcher = THROW_CHECK_NOTNULL(FeatureMatcher::Create(options));
        FeatureMatches matches;
        matcher->Match(image1, image2, &matches);
      },
      []() { return -FreeDeviceBytes(); });
  EXPECT_LT(growth, 32 * kMiB);
}

}  // namespace
}  // namespace colmap
