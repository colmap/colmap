// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/math/matrix.h"

#include "colmap/math/random_eigen.h"
#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(DecomposeMatrixRQ, Nominal) {
  for (int i = 0; i < 10; ++i) {
    const Eigen::Matrix4d A = RandomEigenMatrixd<4, 4>();

    Eigen::Matrix4d R, Q;
    DecomposeMatrixRQ(A, &R, &Q);

    EXPECT_TRUE(R.bottomRows(4).isUpperTriangular());
    EXPECT_TRUE(Q.isUnitary());
    EXPECT_NEAR(Q.determinant(), 1.0, 1e-6);
    EXPECT_THAT(A, EigenMatrixNear(Eigen::Matrix4d(R * Q), 1e-6));
  }
}

}  // namespace
}  // namespace colmap
