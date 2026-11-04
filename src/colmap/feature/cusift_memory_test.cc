// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// cusift's memory pool:
// - It allocates from its own stream-ordered memory pool, so the device's
//   default pool, which other code in the process may use, keeps its settings.
// - Destroying an extractor returns the pool memory it no longer uses to the
//   driver, so GPU stages that run after extraction can have it.

#include "colmap/feature/cusift.h"
#include "colmap/feature/extractor.h"
#include "colmap/feature/sift.h"
#include "colmap/math/random.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/cuda.h"
#include "colmap/util/logging.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <thread>

#include "cusift_device.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

constexpr int64_t kMiB = 1024 * 1024;

void RunOnFreshThread(const std::function<void()>& fn) {
  std::thread thread(fn);
  thread.join();
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

TEST(CusiftMemoryPool, LeavesTheDefaultPoolUntouched) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
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
  const Bitmap bitmap = CreateNoiseImage(2048, 2048);
  int64_t reserved_while_alive = 0;
  int64_t reserved_after = 0;
  int64_t used_after = 0;
  RunOnFreshThread([&]() {
    auto extractor = THROW_CHECK_NOTNULL(
        FeatureExtractor::Create(CreateLargeExtractionOptions()));
    FeatureKeypoints keypoints;
    FeatureDescriptors descriptors;
    THROW_CHECK(extractor->Extract(bitmap, &keypoints, &descriptors));
    reserved_while_alive =
        static_cast<int64_t>(cusift::GetMemoryPoolUsage().reserved);
    extractor.reset();
    const cusift::MemoryPoolUsage usage = cusift::GetMemoryPoolUsage();
    reserved_after = static_cast<int64_t>(usage.reserved);
    used_after = static_cast<int64_t>(usage.used);
  });
  LOG(INFO) << "cusift pool reserved (MiB): " << reserved_while_alive / kMiB
            << " while the extractor is alive, " << reserved_after / kMiB
            << " after (" << used_after / kMiB << " still in use)";
  // The pyramid of a 2048 x 2048 image at first_octave -1 is hundreds of MiB,
  // all kept for reuse while the extractor lives...
  EXPECT_GT(reserved_while_alive, 128 * kMiB);
  // ...and handed back to the driver once it is destroyed, apart from the
  // pool segment that still holds the thread's scratch (freed when the thread
  // exits). Measured on an RTX 5090: 1664 MiB reserved while alive, 64 MiB
  // after, 6 MiB of it in use.
  EXPECT_LT(reserved_after, reserved_while_alive / 4);
}

}  // namespace
}  // namespace colmap
