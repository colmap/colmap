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

#include "colmap/geometry/sim3_matchers.h"

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
  const Sim3d x(2, Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  Sim3d y = x;
  EXPECT_THAT(x, Sim3dEq(y));
  y.scale += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dEq(y)));
  y = x;
  y.rotation.w() += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dEq(y)));
  y = x;
  y.translation.x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dEq(y)));

  testing::StrictMock<MockTestClass> mock;
  EXPECT_CALL(mock, TestMethod(Sim3dEq(x))).Times(1);
  EXPECT_CALL(mock, TestMethod(Sim3dEq(y))).Times(1);
  mock.TestMethod(x);
  mock.TestMethod(y);
}

TEST(Sim3d, Near) {
  const Sim3d x(2, Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  Sim3d y = x;
  EXPECT_THAT(x, Sim3dNear(y, /*stol=*/1e-8, /*rtol=*/1e-8, /*ttol=*/1e-8));
  y.rotation.w() += 1e-7;
  EXPECT_THAT(
      x,
      testing::Not(Sim3dNear(y, /*stol=*/1e-8, /*rtol=*/1e-8, /*ttol=*/1e-8)));
  y = x;
  y.rotation.w() += 1e-7;
  EXPECT_THAT(
      x,
      testing::Not(Sim3dNear(y, /*stol=*/1e-8, /*rtol=*/1e-8, /*ttol=*/1e-8)));
  y = x;
  y.translation.x() += 1e-7;
  EXPECT_THAT(x, testing::Not(Sim3dNear(y)));

  testing::StrictMock<MockTestClass> mock;
  EXPECT_CALL(mock, TestMethod(Sim3dNear(x))).Times(1);
  EXPECT_CALL(mock, TestMethod(Sim3dNear(y))).Times(1);
  mock.TestMethod(x);
  mock.TestMethod(y);
}

}  // namespace
}  // namespace colmap
