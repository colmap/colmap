// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/utils.h"

#include "colmap/math/random_eigen.h"
#include "colmap/sensor/bitmap.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(FeatureKeypointsToPointsVector, Nominal) {
  FeatureKeypoints keypoints(2);
  keypoints[1].x = 0.1;
  keypoints[1].y = 0.2;
  const std::vector<Eigen::Vector2d> points =
      FeatureKeypointsToPointsVector(keypoints);
  EXPECT_EQ(points[0], Eigen::Vector2d(0, 0));
  EXPECT_EQ(points[1].cast<float>(), Eigen::Vector2f(0.1, 0.2));
}

TEST(L2NormalizeFeatureDescriptors, Nominal) {
  FeatureDescriptorsFloatData descriptors = RandomEigenMatrixXf(100, 128);
  descriptors.array() += 1.0f;
  L2NormalizeFeatureDescriptors(&descriptors);
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    EXPECT_NEAR(descriptors.row(r).norm(), 1, 1e-6);
  }
}

TEST(L1RootNormalizeFeatureDescriptors, Nominal) {
  FeatureDescriptorsFloatData descriptors = RandomEigenMatrixXf(100, 128);
  descriptors.array() += 1.0f;
  L1RootNormalizeFeatureDescriptors(&descriptors);
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    EXPECT_NEAR(descriptors.row(r).norm(), 1, 1e-6);
  }
}

TEST(FeatureDescriptorsToUnsignedByte, Nominal) {
  Eigen::MatrixXf descriptors = RandomEigenMatrixXf(100, 128);
  descriptors.array() += 1.0f;
  const FeatureDescriptorsData descriptors_uint8 =
      FeatureDescriptorsToUnsignedByte(descriptors);
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    for (Eigen::MatrixXf::Index c = 0; c < descriptors.cols(); ++c) {
      EXPECT_EQ(static_cast<uint8_t>(
                    std::min(255.0f, std::round(512.0f * descriptors(r, c)))),
                descriptors_uint8(r, c));
    }
  }
}

TEST(ExtractTopScaleFeatures, Nominal) {
  FeatureKeypoints keypoints(5);
  keypoints[0].Rescale(3);
  keypoints[1].Rescale(4);
  keypoints[2].Rescale(1);
  keypoints[3].Rescale(5);
  keypoints[4].Rescale(2);
  FeatureDescriptors descriptors;
  descriptors.data = FeatureDescriptorsData::Random(5, 128);

  auto top_keypoints2 = keypoints;
  auto top_descriptors2 = descriptors;
  ExtractTopScaleFeatures(&top_keypoints2, &top_descriptors2, 2);
  EXPECT_EQ(top_keypoints2.size(), 2);
  EXPECT_EQ(top_keypoints2[0].ComputeScale(), keypoints[3].ComputeScale());
  EXPECT_EQ(top_keypoints2[1].ComputeScale(), keypoints[1].ComputeScale());
  EXPECT_EQ(top_descriptors2.data.rows(), 2);
  EXPECT_EQ(top_descriptors2.data.row(0), descriptors.data.row(3));
  EXPECT_EQ(top_descriptors2.data.row(1), descriptors.data.row(1));

  auto top_keypoints5 = keypoints;
  auto top_descriptors5 = descriptors;
  ExtractTopScaleFeatures(&top_keypoints5, &top_descriptors5, 5);
  EXPECT_EQ(top_keypoints5.size(), 5);
  EXPECT_EQ(top_descriptors5.data.rows(), 5);
  EXPECT_EQ(top_descriptors5.data, descriptors.data);

  auto top_keypoints6 = keypoints;
  auto top_descriptors6 = descriptors;
  ExtractTopScaleFeatures(&top_keypoints6, &top_descriptors6, 6);
  EXPECT_EQ(top_keypoints5.size(), 5);
  EXPECT_EQ(top_descriptors6.data.rows(), 5);
  EXPECT_EQ(top_descriptors6.data, descriptors.data);
}

TEST(HWCToCHW, Nominal) {
  // 2x1 image, pitch with 2 padding bytes.
  const int width = 2;
  const int height = 1;
  const int pitch = 8;
  std::vector<uint8_t> data(pitch * height, 0);
  data[0] = 255;
  data[1] = 0;
  data[2] = 128;
  data[3] = 0;
  data[4] = 255;
  data[5] = 64;
  const std::vector<float> chw = HWCToCHW(data.data(), width, height, pitch);
  ASSERT_EQ(chw.size(), 6);
  EXPECT_FLOAT_EQ(chw[0], 1.0f);
  EXPECT_FLOAT_EQ(chw[1], 0.0f);
  EXPECT_FLOAT_EQ(chw[2], 0.0f);
  EXPECT_FLOAT_EQ(chw[3], 1.0f);
  EXPECT_FLOAT_EQ(chw[4], 128.0f / 255.0f);
  EXPECT_FLOAT_EQ(chw[5], 64.0f / 255.0f);
}

TEST(BitmapToCHW, Nominal) {
  Bitmap bitmap(2, 1, /*as_rgb=*/true);
  EXPECT_TRUE(bitmap.SetPixel(0, 0, BitmapColor<uint8_t>(255, 0, 128)));
  EXPECT_TRUE(bitmap.SetPixel(1, 0, BitmapColor<uint8_t>(0, 255, 64)));
  const std::vector<float> chw = BitmapToCHW(bitmap);
  ASSERT_EQ(chw.size(), 6);
  EXPECT_FLOAT_EQ(chw[0], 1.0f);
  EXPECT_FLOAT_EQ(chw[1], 0.0f);
  EXPECT_FLOAT_EQ(chw[2], 0.0f);
  EXPECT_FLOAT_EQ(chw[3], 1.0f);
  EXPECT_FLOAT_EQ(chw[4], 128.0f / 255.0f);
  EXPECT_FLOAT_EQ(chw[5], 64.0f / 255.0f);
}

TEST(BitmapToCHW, GreyThrows) {
  const Bitmap bitmap(2, 1, /*as_rgb=*/false);
  EXPECT_THROW(BitmapToCHW(bitmap), std::invalid_argument);
}

struct DummyCacheFeatures {
  int value = 0;
};

struct DummyCacheImage {
  image_t image_id = kInvalidImageId;
};

TEST(ImageFeatureCacheTest, CachingAndEviction) {
  ImageFeatureCache<DummyCacheFeatures> cache;
  int num_creates = 0;
  auto create = [&](const DummyCacheImage& image) {
    ++num_creates;
    DummyCacheFeatures features;
    features.value = static_cast<int>(image.image_id);
    return features;
  };

  const DummyCacheImage imageA{1};
  const DummyCacheImage imageB{2};
  const DummyCacheImage imageC{3};

  const DummyCacheFeatures& featuresA = cache.GetOrCreate(imageA, create);
  const DummyCacheFeatures& featuresB = cache.GetOrCreate(imageB, create);
  EXPECT_EQ(num_creates, 2);
  EXPECT_EQ(featuresA.value, 1);
  EXPECT_EQ(featuresB.value, 2);

  // Same images hit the cache.
  cache.GetOrCreate(imageA, create);
  cache.GetOrCreate(imageB, create);
  EXPECT_EQ(num_creates, 2);

  // (B, C): B hits, C evicts A (least recently used).
  cache.GetOrCreate(imageB, create);
  const DummyCacheFeatures& featuresC = cache.GetOrCreate(imageC, create);
  EXPECT_EQ(num_creates, 3);
  EXPECT_EQ(featuresC.value, 3);

  // B is retained while A was evicted.
  cache.GetOrCreate(imageB, create);
  EXPECT_EQ(num_creates, 3);
  cache.GetOrCreate(imageA, create);
  EXPECT_EQ(num_creates, 4);
}

TEST(ImageFeatureCacheTest, SameImageTwice) {
  ImageFeatureCache<DummyCacheFeatures> cache;
  int num_creates = 0;
  auto create = [&](const DummyCacheImage& image) {
    ++num_creates;
    DummyCacheFeatures features;
    features.value = static_cast<int>(image.image_id);
    return features;
  };

  const DummyCacheImage imageA{1};
  const DummyCacheFeatures& first = cache.GetOrCreate(imageA, create);
  const DummyCacheFeatures& second = cache.GetOrCreate(imageA, create);
  EXPECT_EQ(num_creates, 1);
  EXPECT_EQ(&first, &second);
  EXPECT_EQ(second.value, 1);
}

TEST(ImageFeatureCacheTest, InvalidIdAlwaysRecreates) {
  ImageFeatureCache<DummyCacheFeatures> cache;
  int num_creates = 0;
  auto create = [&](const DummyCacheImage& image) {
    ++num_creates;
    DummyCacheFeatures features;
    features.value = static_cast<int>(image.image_id);
    return features;
  };

  const DummyCacheImage invalid;
  const DummyCacheImage imageA{1};
  cache.GetOrCreate(imageA, create);
  cache.GetOrCreate(invalid, create);
  // Invalid entries are never reused.
  cache.GetOrCreate(imageA, create);
  EXPECT_EQ(num_creates, 2);
  cache.GetOrCreate(invalid, create);
  EXPECT_EQ(num_creates, 3);
}

}  // namespace
}  // namespace colmap
