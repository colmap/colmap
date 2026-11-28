// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/eigen_matchers.h"

#include <limits>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

struct TestClass {
  virtual ~TestClass() = default;
  virtual void TestMethod(const Eigen::MatrixXd&) const {}
};

struct MockTestClass : public TestClass {
  MOCK_METHOD(void, TestMethod, (const Eigen::MatrixXd&), (const, override));
};

TEST(EigenMatrix, Eq) {
  Eigen::MatrixXd x(2, 3);
  x << 1, 2, 3, 4, 5, 6;
  Eigen::MatrixXd y = x;
  EXPECT_THAT(x, EigenMatrixEq(y));
  y(0, 0) += 1;
  EXPECT_THAT(x, testing::Not(EigenMatrixEq(y)));
  y = x.block(0, 0, 2, 2);
  EXPECT_THAT(x, testing::Not(EigenMatrixEq(y)));

  testing::StrictMock<MockTestClass> mock;
  EXPECT_CALL(mock, TestMethod(EigenMatrixEq(x))).Times(1);
  EXPECT_CALL(mock, TestMethod(EigenMatrixEq(y))).Times(1);
  mock.TestMethod(x);
  mock.TestMethod(y);
}

TEST(EigenMatrix, Near) {
  Eigen::MatrixXd x(2, 3);
  x << 1, 2, 3, 4, 5, 6;
  Eigen::MatrixXd y = x;
  y(0, 0) += 1e-16;
  EXPECT_THAT(x, EigenMatrixNear(y, 1e-8));
  y(0, 0) += 1e-7;
  EXPECT_THAT(x, testing::Not(EigenMatrixNear(y, 1e-8)));
  y = x.block(0, 0, 2, 2);
  EXPECT_THAT(x, testing::Not(EigenMatrixNear(y)));

  testing::StrictMock<MockTestClass> mock;
  EXPECT_CALL(mock, TestMethod(EigenMatrixNear(x))).Times(1);
  EXPECT_CALL(mock, TestMethod(EigenMatrixNear(y))).Times(1);
  mock.TestMethod(x);
  mock.TestMethod(y);
}

TEST(EigenMatrix, NearZeroLeft) {
  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(2, 3);
  Eigen::MatrixXd y = x;
  EXPECT_THAT(x, EigenMatrixNear(y, 1e-8));
  x(0, 0) += 1e-16;
  EXPECT_THAT(x, EigenMatrixNear(y, 1e-8));
  x(0, 0) += 1e-7;
  EXPECT_THAT(x, testing::Not(EigenMatrixNear(y, 1e-8)));
}

TEST(EigenMatrix, NearZeroRight) {
  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(2, 3);
  Eigen::MatrixXd y = x;
  EXPECT_THAT(x, EigenMatrixNear(y, 1e-8));
  y(0, 0) += 1e-16;
  EXPECT_THAT(x, EigenMatrixNear(y, 1e-8));
  y(0, 0) += 1e-7;
  EXPECT_THAT(x, testing::Not(EigenMatrixNear(y, 1e-8)));
}

TEST(EigenSamplesFollowCov, FullRankAndRankDeficient) {
  std::mt19937 rng(0);
  std::normal_distribution<double> normal(0.0, 1.0);

  // Correlated 3x3 covariance with fixed-size Vector3d samples.
  Eigen::Matrix3d L;
  L << 2.0, 0.0, 0.0, 1.2, 1.5, 0.0, -0.8, 0.7, 0.9;
  const Eigen::Matrix3d cov = L * L.transpose();

  constexpr int kNumSamples = 500;
  std::vector<Eigen::Vector3d> samples;
  for (int i = 0; i < kNumSamples; ++i) {
    const Eigen::Vector3d z(normal(rng), normal(rng), normal(rng));
    samples.push_back(L * z);
  }
  EXPECT_THAT(samples, SamplesFollowCov(cov));
  EXPECT_THAT(samples, testing::Not(SamplesFollowCov(2.25 * cov)));
  EXPECT_THAT(samples,
              testing::Not(SamplesFollowCov(
                  Eigen::MatrixXd(cov.diagonal().asDiagonal()))));
  EXPECT_THAT(samples, testing::Not(SamplesFollowCov(cov, /*null_dim=*/1)));

  // Rank-deficient 3x3 covariance (null_dim = 1) with dynamic VectorXd samples.
  Eigen::Matrix<double, 3, 2> B;
  B << 1.0, 0.0, 0.5, 1.0, -0.5, 0.5;
  const Eigen::MatrixXd rank2_cov = B * B.transpose();
  std::vector<Eigen::VectorXd> rank2_samples;
  for (int i = 0; i < kNumSamples; ++i) {
    const Eigen::Vector2d z(normal(rng), normal(rng));
    rank2_samples.push_back(B * z);
  }
  EXPECT_THAT(rank2_samples, SamplesFollowCov(rank2_cov, /*null_dim=*/1));

  const Eigen::Vector3d null_dir = B.col(0).cross(B.col(1)).normalized();
  std::vector<Eigen::VectorXd> leaky_samples = rank2_samples;
  for (Eigen::VectorXd& sample : leaky_samples) {
    sample += 0.1 * normal(rng) * null_dir;
  }
  EXPECT_THAT(leaky_samples,
              testing::Not(SamplesFollowCov(rank2_cov, /*null_dim=*/1)));

  // Edge cases fail loudly rather than passing silently.
  std::vector<Eigen::Vector3d> empty_samples;
  EXPECT_THAT(empty_samples, testing::Not(SamplesFollowCov(cov)));
  EXPECT_THAT(samples, testing::Not(SamplesFollowCov(cov, /*null_dim=*/-1)));
  EXPECT_THAT(samples, testing::Not(SamplesFollowCov(cov, /*null_dim=*/3)));
  EXPECT_THAT(samples,
              testing::Not(SamplesFollowCov(Eigen::Matrix2d::Identity())));

  std::vector<Eigen::Vector3d> nan_samples = samples;
  nan_samples[0](0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THAT(nan_samples, testing::Not(SamplesFollowCov(cov)));
}

}  // namespace
}  // namespace colmap
