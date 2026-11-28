// SPDX-License-Identifier: BSD-3-Clause

#if defined(__CUDACC__) || defined(__HIPCC__)
#define BOOST_PP_VARIADICS 0
#endif

#include "colmap/mvs/gpu_mat.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/patch_match_cuda.h"
#include "colmap/mvs/patch_match_options.h"
#include "colmap/mvs/patch_match_sampling.cuh"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/cuda.h"
#include "colmap/util/cudacc.h"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {

TEST(PatchMatchSampling, PreservesSupportAndStrides) {
  EXPECT_FALSE(TransformPDFToCDF(nullptr, 0, 32));
  EXPECT_EQ(SampleSource(nullptr, 0, 32, 1.0f), -1);
  for (const int stride : {1, 8, 32}) {
    std::vector<float> probs(5 * stride, -7.0f);
    for (int i = 0; i < 5; ++i) probs[i * stride] = i % 2 ? -0.0f : 0.0f;
    const auto zeros = probs;
    EXPECT_FALSE(TransformPDFToCDF(probs.data(), 5, stride));
    EXPECT_EQ(
        std::memcmp(probs.data(), zeros.data(), probs.size() * sizeof(float)),
        0);
    probs[stride] = 1.0f;
    probs[3 * stride] = 3.0f;
    ASSERT_TRUE(TransformPDFToCDF(probs.data(), 5, stride));
    EXPECT_EQ(probs[0], 0.0f);
    EXPECT_EQ(probs[stride], 0.25f);
    EXPECT_EQ(probs[2 * stride], 0.25f);
    EXPECT_EQ(probs[3 * stride], 1.0f);
    EXPECT_EQ(probs[4 * stride], 1.0f);
    for (float u : {std::numeric_limits<float>::denorm_min(),
                    FLT_EPSILON / 2,
                    std::nextafter(FLT_EPSILON, 0.0f),
                    FLT_EPSILON,
                    std::nextafter(FLT_EPSILON, 1.0f)}) {
      EXPECT_EQ(SampleSource(probs.data(), 5, stride, u), 1);
    }
    EXPECT_EQ(SampleSource(probs.data(), 5, stride, .25f + FLT_EPSILON), 3);
    EXPECT_EQ(SampleSource(probs.data(), 5, stride, 1.0f), 3);
    for (int j = 0; j < 257; ++j) {
      const float u = float(j + 1) / 257.0f;
      const float old_threshold = u - FLT_EPSILON;
      int old_source = -1;
      for (int i = 0; i < 5; ++i) {
        if (probs[i * stride] > old_threshold) {
          old_source = i;
          break;
        }
      }
      EXPECT_EQ(SampleSource(probs.data(), 5, stride, u), old_source);
    }
    for (size_t i = 0; i < probs.size(); ++i) {
      if (i % stride != 0) EXPECT_EQ(probs[i], -7.0f);
    }
  }
  // Retain legacy classification and selection; these are not valid CDFs.
  float mixed[] = {-1.0f, 1.0f};
  ASSERT_TRUE(TransformPDFToCDF(mixed, 2, 1));
  EXPECT_TRUE(std::isinf(mixed[0]));
  EXPECT_TRUE(std::signbit(mixed[0]));
  EXPECT_TRUE(std::isnan(mixed[1]));
  EXPECT_EQ(SampleSource(mixed, 2, 1, .5f), -1);
  float reversed[] = {1.0f, -1.0f};
  ASSERT_TRUE(TransformPDFToCDF(reversed, 2, 1));
  EXPECT_TRUE(std::isinf(reversed[0]));
  EXPECT_FALSE(std::signbit(reversed[0]));
  EXPECT_TRUE(std::isnan(reversed[1]));
  EXPECT_EQ(SampleSource(reversed, 2, 1, .5f), 0);
  for (float first : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity()}) {
    float invalid[] = {first, 0.0f};
    ASSERT_TRUE(TransformPDFToCDF(invalid, 2, 1));
    EXPECT_TRUE(std::isnan(invalid[0]));
    EXPECT_TRUE(std::isnan(invalid[1]));
    EXPECT_EQ(SampleSource(invalid, 2, 1, .5f), -1);
  }
  float negative[] = {-1.0f, -3.0f};
  ASSERT_TRUE(TransformPDFToCDF(negative, 2, 1));
  EXPECT_EQ(negative[0], .25f);
  EXPECT_EQ(negative[1], 1.0f);
  EXPECT_EQ(SampleSource(negative, 2, 1, .5f), 1);
}

namespace {
class PatchMatchSamplingDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (GetNumCudaDevices() == 0) {
      GTEST_SKIP() << "No CUDA or HIP device available";
    }
  }
};

__global__ void CheckSourceSampling(int* errors) {
  int failures = 0;
  for (int i = 0; i < 3; ++i) {
    const int stride = i == 0 ? 1 : (i == 1 ? 8 : 32);
    float probs[96] = {};
    failures += TransformPDFToCDF(probs, 3, stride);
    failures += SampleSource(probs, 3, stride, FLT_EPSILON / 2) != -1;
    failures += SampleSource(probs, 3, stride, 1.0f) != -1;
    probs[stride] = 1.0f;
    probs[2 * stride] = 3.0f;
    failures += !TransformPDFToCDF(probs, 3, stride);
    failures +=
        probs[0] != 0.0f || probs[stride] != 0.25f || probs[2 * stride] != 1.0f;
    failures += SampleSource(probs, 3, stride, FLT_EPSILON / 2) != 1;
    failures += SampleSource(probs, 3, stride, 0.5f) != 2;
    failures += SampleSource(probs, 3, stride, 1.0f) != 2;
  }
  float invalid[2] = {-1.0f, 1.0f};
  failures += !TransformPDFToCDF(invalid, 2, 1);
  failures += SampleSource(invalid, 2, 1, 0.5f) != -1;
  invalid[0] = __int_as_float(0x7fc00000);
  invalid[1] = 0.0f;
  failures += !TransformPDFToCDF(invalid, 2, 1);
  failures += SampleSource(invalid, 2, 1, 0.5f) != -1;
  *errors = failures;
}
}  // namespace

TEST_F(PatchMatchSamplingDeviceTest, DeviceSupportAndLowerEndpoint) {
  GpuMat<int> errors(1, 1);
  CheckSourceSampling<<<1, 1>>>(errors.GetPtr());
  CUDA_SYNC_AND_CHECK();
  int actual = -1;
  errors.CopyToHost(&actual, sizeof(actual));
  EXPECT_EQ(actual, 0);
}

TEST_F(PatchMatchSamplingDeviceTest, EmptySupportRetainsInitializedParameters) {
  constexpr int width = 47;
  constexpr int height = 35;
  const float K[9] = {100.0f,
                      0.0f,
                      width / 2.0f,
                      0.0f,
                      100.0f,
                      height / 2.0f,
                      0.0f,
                      0.0f,
                      1.0f};
  const float R[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  std::vector<Image> images;
  for (int image_idx = 0; image_idx < 3; ++image_idx) {
    const float T[3] = {0.1f * image_idx, 0.0f, 0.0f};
    images.emplace_back("sampling-test", width, height, K, R, T);
    Bitmap bitmap(width, height, false);
    // Match the zero texture border so edge windows also have zero variance.
    bitmap.Fill(BitmapColor<uint8_t>(0));
    images.back().SetBitmap(std::move(bitmap));
  }
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = {1, 2};
  problem.images = &images;
  for (const int radius : {2, 5}) {
    SCOPED_TRACE(radius);
    PatchMatchOptions options;
    options.depth_min = 1.0;
    options.depth_max = 2.0;
    options.window_radius = radius;
    options.ncc_sigma = 1.0 / 16;
    options.num_iterations = 1;
    options.geom_consistency = false;
    options.filter = false;
    // Constant patches produce maximum NCC cost; exp(-512) underflows to
    // zero. There is no source evidence for replacing the initialized plane.
    PatchMatchCuda owner(options, problem);
    const auto depth = owner.GetDepthMap();
    const auto normal = owner.GetNormalMap();
    ASSERT_EQ(depth.GetData().size(), size_t{width * height});
    ASSERT_EQ(normal.GetData().size(), size_t{3 * width * height});
    ASSERT_GT(depth.GetData().front(), 0.0f);
    owner.Run();
    EXPECT_EQ(depth.GetData(), owner.GetDepthMap().GetData());
    EXPECT_EQ(normal.GetData(), owner.GetNormalMap().GetData());
  }
}

}  // namespace mvs
}  // namespace colmap
