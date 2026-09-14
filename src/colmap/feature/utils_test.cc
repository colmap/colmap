// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/utils.h"

#include "colmap/math/random_eigen.h"

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

}  // namespace
}  // namespace colmap
