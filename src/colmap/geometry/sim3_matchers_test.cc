// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/geometry/sim3_matchers.h"

#include "colmap/math/random_eigen.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

struct TestClass {
  virtual ~TestClass() = default;
  virtual void TestMethod(const Sim3d&) const {}
};

struct MockTestClass : public TestClass {
  MOCK_METHOD(void, TestMethod, (const Sim3d&), (const, override));
};

TEST(Sim3d, Eq) {
  const Sim3d x(2, RandomEigenQuaterniond(), RandomEigenVectord<3>());
  Sim3d y = x;
  EXPECT_THAT(x, Sim3dEq(y));
  y.scale() += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dEq(y)));
  y = x;
  y.rotation().w() += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dEq(y)));
  y = x;
  y.translation().x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dEq(y)));

  testing::StrictMock<MockTestClass> mock;
  EXPECT_CALL(mock, TestMethod(Sim3dEq(x))).Times(1);
  EXPECT_CALL(mock, TestMethod(Sim3dEq(y))).Times(1);
  mock.TestMethod(x);
  mock.TestMethod(y);
}

TEST(Sim3d, Near) {
  const Sim3d x(2, RandomEigenQuaterniond(), RandomEigenVectord<3>());
  Sim3d y = x;
  EXPECT_THAT(x, Sim3dNear(y, /*stol=*/1e-8, /*rtol=*/1e-8, /*ttol=*/1e-8));
  y.rotation().w() += 1e-7;
  EXPECT_THAT(
      x,
      testing::Not(Sim3dNear(y, /*stol=*/1e-8, /*rtol=*/1e-8, /*ttol=*/1e-8)));
  y = x;
  y.rotation().w() += 1e-7;
  EXPECT_THAT(
      x,
      testing::Not(Sim3dNear(y, /*stol=*/1e-8, /*rtol=*/1e-8, /*ttol=*/1e-8)));
  y = x;
  y.translation().x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dNear(y)));

  testing::StrictMock<MockTestClass> mock;
  EXPECT_CALL(mock, TestMethod(Sim3dNear(x))).Times(1);
  EXPECT_CALL(mock, TestMethod(Sim3dNear(y))).Times(1);
  mock.TestMethod(x);
  mock.TestMethod(y);
}

TEST(Sim3d, NearUsesIndependentScaleTolerance) {
  const Sim3d x(2, RandomEigenQuaterniond(), RandomEigenVectord<3>());
  Sim3d y = x;
  y.scale() += 1e-7;
  // The scale difference is above stol but below rtol/ttol, so only the
  // scale check must reject the match. This fails if the scale comparison
  // wrongly uses rtol instead of stol.
  EXPECT_THAT(
      x, testing::Not(Sim3dNear(y, /*stol=*/1e-8, /*rtol=*/1, /*ttol=*/1)));
  EXPECT_THAT(x, Sim3dNear(y, /*stol=*/1e-6, /*rtol=*/1e-8, /*ttol=*/1e-8));
}

TEST(Sim3d, LeftScaleNearIdentity) {
  Sim3d x(1, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  Sim3d y = x;
  EXPECT_THAT(x, Sim3dNear(y, 1e-8));
  x.scale() += 1e-16;
  EXPECT_THAT(x, Sim3dNear(y, 1e-8));
  x.scale() += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dNear(y, 1e-8)));
}

TEST(Sim3d, RightScaleNearIdentity) {
  Sim3d x(1, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  Sim3d y = x;
  EXPECT_THAT(x, Sim3dNear(y, 1e-8));
  y.scale() += 1e-16;
  EXPECT_THAT(x, Sim3dNear(y, 1e-8));
  y.scale() += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dNear(y, 1e-8)));
}

TEST(Sim3d, LeftRotationNearIdentity) {
  Sim3d x(1, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  Sim3d y = x;
  EXPECT_THAT(x, Sim3dNear(y, 1e-8));
  x.rotation().x() += 1e-16;
  EXPECT_THAT(x, Sim3dNear(y, 1e-8));
  x.rotation().x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dNear(y, 1e-8)));
}

TEST(Sim3d, RightRotationNearIdentity) {
  Sim3d x(1, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  Sim3d y = x;
  EXPECT_THAT(x, Sim3dNear(y, 1e-8));
  y.rotation().x() += 1e-16;
  EXPECT_THAT(x, Sim3dNear(y, 1e-8));
  y.rotation().x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dNear(y, 1e-8)));
}

TEST(Sim3d, LeftTranslationNearIdentity) {
  Sim3d x(1, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  Sim3d y = x;
  EXPECT_THAT(x, Sim3dNear(y, 1e-8));
  x.translation().x() += 1e-16;
  EXPECT_THAT(x, Sim3dNear(y, 1e-8));
  x.translation().x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dNear(y, 1e-8)));
}

TEST(Sim3d, RightTranslationNearIdentity) {
  Sim3d x(1, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  Sim3d y = x;
  EXPECT_THAT(x, Sim3dNear(y, 1e-8));
  y.translation().x() += 1e-16;
  EXPECT_THAT(x, Sim3dNear(y, 1e-8));
  y.translation().x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dNear(y, 1e-8)));
}

}  // namespace
}  // namespace colmap
