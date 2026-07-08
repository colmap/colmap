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

#include "colmap/estimators/cost_functions/tiny_manifold.h"

#include "colmap/util/eigen_matchers.h"

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Central finite-difference approximation of the (ambient x tangent) Jacobian
// of Manifold::Plus at delta = 0.
template <typename Manifold>
Eigen::MatrixXd NumericPlusJacobian(const Manifold& manifold, const double* x) {
  constexpr int kAmbient = Manifold::kAmbientSize;
  constexpr int kTangent = Manifold::kTangentSize;
  constexpr double kEps = 1e-6;
  Eigen::MatrixXd J(kAmbient, kTangent);
  for (int j = 0; j < kTangent; ++j) {
    std::vector<double> delta_plus(kTangent, 0.0);
    std::vector<double> delta_minus(kTangent, 0.0);
    delta_plus[j] = kEps;
    delta_minus[j] = -kEps;
    std::vector<double> x_plus(kAmbient);
    std::vector<double> x_minus(kAmbient);
    manifold.Plus(x, delta_plus.data(), x_plus.data());
    manifold.Plus(x, delta_minus.data(), x_minus.data());
    for (int r = 0; r < kAmbient; ++r) {
      J(r, j) = (x_plus[r] - x_minus[r]) / (2 * kEps);
    }
  }
  return J;
}

// Row-major analytic Jacobian returned by Manifold::PlusJacobian, as a matrix.
template <typename Manifold>
Eigen::MatrixXd AnalyticPlusJacobian(const Manifold& manifold,
                                     const double* x) {
  constexpr int kAmbient = Manifold::kAmbientSize;
  constexpr int kTangent = Manifold::kTangentSize;
  std::vector<double> data(kAmbient * kTangent);
  manifold.PlusJacobian(x, data.data());
  Eigen::MatrixXd J(kAmbient, kTangent);
  for (int r = 0; r < kAmbient; ++r) {
    for (int c = 0; c < kTangent; ++c) {
      J(r, c) = data[r * kTangent + c];
    }
  }
  return J;
}

TEST(EigenQuaternionManifold, PlusAtZeroIsIdentity) {
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()));
  const EigenQuaternionManifold manifold;
  const double delta[3] = {0, 0, 0};
  double x_plus[4];
  manifold.Plus(q.coeffs().data(), delta, x_plus);
  EXPECT_THAT(Eigen::Map<const Eigen::Vector4d>(x_plus),
              EigenMatrixNear(Eigen::Vector4d(q.coeffs()), 1e-12));
}

TEST(EigenQuaternionManifold, PlusComposesRotationOnTheRight) {
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()));
  const Eigen::Vector3d delta(0.05, -0.1, 0.08);
  const EigenQuaternionManifold manifold;
  double x_plus[4];
  manifold.Plus(q.coeffs().data(), delta.data(), x_plus);

  const Eigen::Map<const Eigen::Quaterniond> q_plus_map(x_plus);
  const Eigen::Quaterniond q_plus(q_plus_map);
  const Eigen::Matrix3d expected =
      q.toRotationMatrix() *
      Eigen::AngleAxisd(delta.norm(), delta.normalized()).toRotationMatrix();
  EXPECT_THAT(q_plus.toRotationMatrix(), EigenMatrixNear(expected, 1e-10));
}

TEST(EigenQuaternionManifold, PlusJacobianMatchesFiniteDiff) {
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()));
  const EigenQuaternionManifold manifold;
  EXPECT_THAT(
      AnalyticPlusJacobian(manifold, q.coeffs().data()),
      EigenMatrixNear(NumericPlusJacobian(manifold, q.coeffs().data()), 1e-6));
}

TEST(SphereManifold, PlusStaysOnUnitSphere) {
  const Eigen::Vector3d x = Eigen::Vector3d(0.5, -1.0, 2.0).normalized();
  const SphereManifold<3> manifold;
  const Eigen::Vector2d delta(0.15, -0.2);
  double x_plus[3];
  manifold.Plus(x.data(), delta.data(), x_plus);
  EXPECT_NEAR(Eigen::Map<const Eigen::Vector3d>(x_plus).norm(), 1.0, 1e-12);

  const double zero[2] = {0, 0};
  double x_plus_zero[3];
  manifold.Plus(x.data(), zero, x_plus_zero);
  EXPECT_THAT(Eigen::Map<const Eigen::Vector3d>(x_plus_zero),
              EigenMatrixNear(x, 1e-12));
}

TEST(SphereManifold, PlusJacobianMatchesFiniteDiff) {
  const Eigen::Vector3d x = Eigen::Vector3d(0.5, -1.0, 2.0).normalized();
  const SphereManifold<3> manifold;
  // The columns span the tangent plane, i.e. are orthogonal to x.
  const Eigen::MatrixXd J = AnalyticPlusJacobian(manifold, x.data());
  EXPECT_LT((x.transpose() * J).norm(), 1e-12);
  EXPECT_THAT(J,
              EigenMatrixNear(NumericPlusJacobian(manifold, x.data()), 1e-6));
}

TEST(ProductManifold, SizesAndBlockStructure) {
  using RelativePoseManifold =
      ProductManifold<EigenQuaternionManifold, SphereManifold<3>>;
  static_assert(RelativePoseManifold::kAmbientSize == 7);
  static_assert(RelativePoseManifold::kTangentSize == 5);

  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.9, Eigen::Vector3d(-1, 0.5, 2).normalized()));
  const Eigen::Vector3d t = Eigen::Vector3d(1.0, -2.0, 0.5).normalized();
  double x[7] = {q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};

  const RelativePoseManifold manifold;

  // Plus at zero recovers the point.
  const double zero[5] = {0, 0, 0, 0, 0};
  double x_plus[7];
  manifold.Plus(x, zero, x_plus);
  const Eigen::Matrix<double, 7, 1> x_plus_vec =
      Eigen::Map<const Eigen::Matrix<double, 7, 1>>(x_plus);
  const Eigen::Matrix<double, 7, 1> x_vec =
      Eigen::Map<const Eigen::Matrix<double, 7, 1>>(x);
  EXPECT_THAT(x_plus_vec, EigenMatrixNear(x_vec, 1e-12));

  // The analytic Jacobian is block-diagonal and matches finite differences.
  const Eigen::MatrixXd J = AnalyticPlusJacobian(manifold, x);
  EXPECT_THAT(J, EigenMatrixNear(NumericPlusJacobian(manifold, x), 1e-6));
  const double top_right_norm = J.topRightCorner(4, 2).norm();
  const double bottom_left_norm = J.bottomLeftCorner(3, 3).norm();
  EXPECT_LT(top_right_norm, 1e-12);
  EXPECT_LT(bottom_left_norm, 1e-12);
}

// A three-way product exercises the variadic recursion (depth 3). The two
// manifolds already in this file suffice; the blocks are laid out in argument
// order and the Plus Jacobian stays block-diagonal.
TEST(ProductManifold, ThreeWaySizesAndBlockStructure) {
  using ThreeWayManifold = ProductManifold<EigenQuaternionManifold,
                                           SphereManifold<3>,
                                           EigenQuaternionManifold>;
  static_assert(ThreeWayManifold::kAmbientSize == 4 + 3 + 4);
  static_assert(ThreeWayManifold::kTangentSize == 3 + 2 + 3);
  constexpr int kAmbient = ThreeWayManifold::kAmbientSize;
  constexpr int kTangent = ThreeWayManifold::kTangentSize;

  const Eigen::Quaterniond q0(
      Eigen::AngleAxisd(0.9, Eigen::Vector3d(-1, 0.5, 2).normalized()));
  const Eigen::Vector3d t = Eigen::Vector3d(1.0, -2.0, 0.5).normalized();
  const Eigen::Quaterniond q1(
      Eigen::AngleAxisd(0.3, Eigen::Vector3d(0.2, -1, 0.7).normalized()));
  double x[kAmbient] = {q0.x(),
                        q0.y(),
                        q0.z(),
                        q0.w(),
                        t.x(),
                        t.y(),
                        t.z(),
                        q1.x(),
                        q1.y(),
                        q1.z(),
                        q1.w()};

  const ThreeWayManifold manifold;

  // Plus at zero recovers the point.
  const double zero[kTangent] = {0, 0, 0, 0, 0, 0, 0, 0};
  double x_plus[kAmbient];
  manifold.Plus(x, zero, x_plus);
  using AmbientVec = Eigen::Matrix<double, kAmbient, 1>;
  const AmbientVec x_plus_vec = Eigen::Map<const AmbientVec>(x_plus);
  const AmbientVec x_vec = Eigen::Map<const AmbientVec>(x);
  EXPECT_THAT(x_plus_vec, EigenMatrixNear(x_vec, 1e-12));

  // The analytic Jacobian matches finite differences.
  Eigen::MatrixXd J = AnalyticPlusJacobian(manifold, x);
  EXPECT_THAT(J, EigenMatrixNear(NumericPlusJacobian(manifold, x), 1e-6));

  // Everything outside the three diagonal blocks (4x3, 3x2, 4x3) is zero.
  J.block(0, 0, 4, 3).setZero();
  J.block(4, 3, 3, 2).setZero();
  J.block(7, 5, 4, 3).setZero();
  EXPECT_LT(J.norm(), 1e-12);
}

}  // namespace
}  // namespace colmap
