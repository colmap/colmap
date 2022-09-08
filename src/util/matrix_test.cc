// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "util/matrix"
#include "util/testing.h"

#include "util/matrix.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestIsNaN) {
  BOOST_CHECK(!IsNaN(Eigen::Vector3f::Zero()));
  BOOST_CHECK(!IsNaN(Eigen::Vector3d::Zero()));
  BOOST_CHECK(IsNaN(
      Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f)));
  BOOST_CHECK(IsNaN(
      Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0.0f, 0.0f)));
}

BOOST_AUTO_TEST_CASE(TestIsInf) {
  BOOST_CHECK(!IsInf(Eigen::Vector3f::Zero()));
  BOOST_CHECK(!IsInf(Eigen::Vector3d::Zero()));
  BOOST_CHECK(IsInf(
      Eigen::Vector3f(std::numeric_limits<float>::infinity(), 0.0f, 0.0f)));
  BOOST_CHECK(IsInf(
      Eigen::Vector3d(std::numeric_limits<double>::infinity(), 0.0f, 0.0f)));
}

BOOST_AUTO_TEST_CASE(TestDecomposeMatrixRQ) {
  for (int i = 0; i < 10; ++i) {
    const Eigen::Matrix4d A = Eigen::Matrix4d::Random();

    Eigen::Matrix4d R, Q;
    DecomposeMatrixRQ(A, &R, &Q);

    BOOST_CHECK(R.bottomRows(4).isUpperTriangular());
    BOOST_CHECK(Q.isUnitary());
    BOOST_CHECK_CLOSE(Q.determinant(), 1.0, 1e-6);
    BOOST_CHECK(A.isApprox(R * Q, 1e-6));
  }
}
