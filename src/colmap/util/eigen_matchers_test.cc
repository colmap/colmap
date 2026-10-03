// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/eigen_matchers.h"

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

}  // namespace
}  // namespace colmap
