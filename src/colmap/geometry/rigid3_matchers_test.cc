// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/geometry/rigid3_matchers.h"

#include "colmap/math/random_eigen.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

struct TestClass {
  virtual ~TestClass() = default;
  virtual void TestMethod(const Rigid3d&) const {}
};

struct MockTestClass : public TestClass {
  MOCK_METHOD(void, TestMethod, (const Rigid3d&), (const, override));
};

TEST(Rigid3d, Eq) {
  const Rigid3d x(RandomEigenQuaterniond(), RandomEigenVectord<3>());
  Rigid3d y = x;
  EXPECT_THAT(x, Rigid3dEq(y));
  y.rotation().w() += 1e-7;
  EXPECT_THAT(x, testing::Not(Rigid3dEq(y)));
  y = x;
  y.translation().x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Rigid3dEq(y)));

  testing::StrictMock<MockTestClass> mock;
  EXPECT_CALL(mock, TestMethod(Rigid3dEq(x))).Times(1);
  EXPECT_CALL(mock, TestMethod(Rigid3dEq(y))).Times(1);
  mock.TestMethod(x);
  mock.TestMethod(y);
}

TEST(Rigid3d, Near) {
  const Rigid3d x(RandomEigenQuaterniond(), RandomEigenVectord<3>());
  Rigid3d y = x;
  EXPECT_THAT(x, Rigid3dNear(y, /*rtol=*/1e-8, /*ttol=*/1e-8));
  y.rotation().w() += 1e-7;
  EXPECT_THAT(x, testing::Not(Rigid3dNear(y, /*rtol=*/1e-8, /*ttol=*/1e-8)));
  y = x;
  y.translation().x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Rigid3dNear(y)));

  testing::StrictMock<MockTestClass> mock;
  EXPECT_CALL(mock, TestMethod(Rigid3dNear(x))).Times(1);
  EXPECT_CALL(mock, TestMethod(Rigid3dNear(y))).Times(1);
  mock.TestMethod(x);
  mock.TestMethod(y);
}

TEST(Rigid3d, LeftRotationNearIdentity) {
  Rigid3d x(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  Rigid3d y = x;
  EXPECT_THAT(x, Rigid3dNear(y, 1e-8));
  x.rotation().x() += 1e-16;
  EXPECT_THAT(x, Rigid3dNear(y, 1e-8));
  x.rotation().x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Rigid3dNear(y, 1e-8)));
}

TEST(Rigid3d, RightRotationNearIdentity) {
  Rigid3d x(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  Rigid3d y = x;
  EXPECT_THAT(x, Rigid3dNear(y, 1e-8));
  y.rotation().x() += 1e-16;
  EXPECT_THAT(x, Rigid3dNear(y, 1e-8));
  y.rotation().x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Rigid3dNear(y, 1e-8)));
}

TEST(Rigid3d, LeftTranslationNearIdentity) {
  Rigid3d x(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  Rigid3d y = x;
  EXPECT_THAT(x, Rigid3dNear(y, 1e-8));
  x.translation().x() += 1e-16;
  EXPECT_THAT(x, Rigid3dNear(y, 1e-8));
  x.translation().x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Rigid3dNear(y, 1e-8)));
}

TEST(Rigid3d, RightTranslationNearIdentity) {
  Rigid3d x(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  Rigid3d y = x;
  EXPECT_THAT(x, Rigid3dNear(y, 1e-8));
  y.translation().x() += 1e-16;
  EXPECT_THAT(x, Rigid3dNear(y, 1e-8));
  y.translation().x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Rigid3dNear(y, 1e-8)));
}

}  // namespace
}  // namespace colmap
