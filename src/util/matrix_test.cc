// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
