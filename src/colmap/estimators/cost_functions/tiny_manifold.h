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

#pragma once

#include "colmap/estimators/cost_functions/quaternion_utils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// Compile-time manifold policies for colmap::TinySolver, mirroring the Ceres
// manifolds used elsewhere in COLMAP (see cost_functions/manifold.h). Each
// policy is a stateless, fixed-size struct satisfying the concept:
//
//   struct Manifold {
//     static constexpr int kAmbientSize;   // size of the parameter block
//     static constexpr int kTangentSize;   // degrees of freedom
//     static constexpr bool kIsEuclidean;  // true iff Plus is plain addition
//     // x_plus_delta = Plus(x, delta), sizes [ambient, tangent, ambient].
//     void Plus(const double* x, const double* delta, double* x_plus) const;
//     // Row-major (kAmbientSize x kTangentSize) Jacobian of Plus at delta = 0.
//     void PlusJacobian(const double* x, double* jacobian) const;
//   };
//
// The solver evaluates these on doubles only (it never differentiates through
// the manifold), so the methods are plain-double.

// Rotation manifold for an Eigen quaternion in (x, y, z, w) coefficient order.
// Retraction: q_plus = normalize(q * ExpMap(delta)) (right multiplication),
// matching how the previous tangent parameterization composed the increment.
struct EigenQuaternionManifold {
  static constexpr int kAmbientSize = 4;
  static constexpr int kTangentSize = 3;
  [[maybe_unused]] static constexpr bool kIsEuclidean = false;

  void Plus(const double* x, const double* delta, double* x_plus) const {
    double delta_q[4];
    EigenQuaternionFromAngleAxis(delta, delta_q);
    const EigenQuaternionMap<double> q(x);
    const EigenQuaternionMap<double> dq(delta_q);
    Eigen::Map<Eigen::Quaterniond> out(x_plus);
    out = (q * dq).normalized();
  }

  void PlusJacobian(const double* x, double* jacobian) const {
    // For the right-multiplication retraction q * ExpMap(delta), the Jacobian
    // at delta = 0 is 0.5 * L(q) restricted to the imaginary columns, where
    // L(q) is the quaternion left-multiplication matrix.
    const EigenQuaternionMap<double> q_map(x);
    const Eigen::Quaterniond q(q_map);
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> out(jacobian);
    out = 0.5 * QuaternionLeftMultMatrix(q).leftCols<3>();
  }
};

// Unit-sphere manifold in R^N. Retraction:
// x_plus = normalize(x + B * delta), where B is an orthonormal basis of the
// tangent plane at x / ||x||. Only the 2-sphere (N = 3) is implemented, which
// is all that is currently needed; the closed-form basis keeps the solver
// allocation-free.
template <int N>
struct SphereManifold {
  // TODO: To generalize to N > 3, replace TangentBasis()'s R^3-specific closed
  // form (unitOrthogonal + cross) with a (less efficient) QR/Householder
  // orthonormalization of the tangent space.
  static_assert(N == 3, "SphereManifold is only implemented for the 2-sphere.");
  static constexpr int kAmbientSize = N;
  static constexpr int kTangentSize = N - 1;
  [[maybe_unused]] static constexpr bool kIsEuclidean = false;

  void Plus(const double* x, const double* delta, double* x_plus) const {
    const Eigen::Map<const Eigen::Vector3d> xv(x);
    const Eigen::Map<const Eigen::Vector2d> d(delta);
    Eigen::Map<Eigen::Vector3d> out(x_plus);
    out = (xv + TangentBasis(xv) * d).normalized();
  }

  // Assumes x is (approximately) unit norm, as maintained by the solver's
  // sphere retraction; the tangent basis equals the Plus Jacobian only at unit
  // x.
  void PlusJacobian(const double* x, double* jacobian) const {
    const Eigen::Map<const Eigen::Vector3d> xv(x);
    Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> out(jacobian);
    out = TangentBasis(xv);
  }

 private:
  // Orthonormal basis (3 x 2) of the tangent plane at x / ||x||.
  static Eigen::Matrix<double, 3, 2> TangentBasis(const Eigen::Vector3d& x) {
    const Eigen::Vector3d x_hat = x.normalized();
    const Eigen::Vector3d b1 = x_hat.unitOrthogonal();
    Eigen::Matrix<double, 3, 2> B;
    B.col(0) = b1;
    B.col(1) = x_hat.cross(b1);
    return B;
  }
};

// Product of one or more manifolds acting on adjacent parameter blocks, e.g.
// ProductManifold<EigenQuaternionManifold, SphereManifold<3>,
// EuclideanManifold<1>> for a relative pose plus a shared focal length.
// Variadic, mirroring ceres::ProductManifold. The ambient/tangent blocks are
// laid out in argument order and the Plus Jacobian is block-diagonal.
template <typename... Manifolds>
struct ProductManifold;

// Base case: a single manifold. The product degenerates to a thin wrapper.
template <typename Manifold>
struct ProductManifold<Manifold> {
  static constexpr int kAmbientSize = Manifold::kAmbientSize;
  static constexpr int kTangentSize = Manifold::kTangentSize;
  [[maybe_unused]] static constexpr bool kIsEuclidean = Manifold::kIsEuclidean;

  Manifold manifold;

  void Plus(const double* x, const double* delta, double* x_plus) const {
    manifold.Plus(x, delta, x_plus);
  }

  void PlusJacobian(const double* x, double* jacobian) const {
    manifold.PlusJacobian(x, jacobian);
  }
};

// Recursive case: split into the head manifold and the product of the rest.
template <typename Head, typename... Tail>
struct ProductManifold<Head, Tail...> {
  using Rest = ProductManifold<Tail...>;

  static constexpr int kAmbientSize = Head::kAmbientSize + Rest::kAmbientSize;
  static constexpr int kTangentSize = Head::kTangentSize + Rest::kTangentSize;
  [[maybe_unused]] static constexpr bool kIsEuclidean =
      Head::kIsEuclidean && Rest::kIsEuclidean;

  Head head;
  Rest rest;

  void Plus(const double* x, const double* delta, double* x_plus) const {
    head.Plus(x, delta, x_plus);
    rest.Plus(x + Head::kAmbientSize,
              delta + Head::kTangentSize,
              x_plus + Head::kAmbientSize);
  }

  void PlusJacobian(const double* x, double* jacobian) const {
    // Row-major (kAmbientSize x kTangentSize), block-diagonal: the head block
    // sits in the top-left, the rest block in the bottom-right. A row-major
    // Eigen matrix with a single column is ill-formed, so single-tangent blocks
    // are stored column-major (identical layout for one column).
    constexpr int kOrder =
        kTangentSize == 1 ? Eigen::ColMajor : Eigen::RowMajor;
    Eigen::Map<Eigen::Matrix<double, kAmbientSize, kTangentSize, kOrder>> J(
        jacobian);
    J.setZero();

    constexpr int kHeadOrder =
        Head::kTangentSize == 1 ? Eigen::ColMajor : Eigen::RowMajor;
    Eigen::Matrix<double, Head::kAmbientSize, Head::kTangentSize, kHeadOrder>
        head_jacobian;
    head.PlusJacobian(x, head_jacobian.data());
    J.template topLeftCorner<Head::kAmbientSize, Head::kTangentSize>() =
        head_jacobian;

    constexpr int kRestOrder =
        Rest::kTangentSize == 1 ? Eigen::ColMajor : Eigen::RowMajor;
    Eigen::Matrix<double, Rest::kAmbientSize, Rest::kTangentSize, kRestOrder>
        rest_jacobian;
    rest.PlusJacobian(x + Head::kAmbientSize, rest_jacobian.data());
    J.template bottomRightCorner<Rest::kAmbientSize, Rest::kTangentSize>() =
        rest_jacobian;
  }
};

}  // namespace colmap
