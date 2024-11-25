// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

}  // namespace
}  // namespace colmap
