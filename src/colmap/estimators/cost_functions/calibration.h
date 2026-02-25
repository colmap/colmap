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

#include <array>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

// Compute polynomial coefficients from cross-products of SVD-derived vectors
// for the Fetzer focal length estimation method. The coefficients encode the
// relationship between the two focal lengths derived from the fundamental
// matrix constraint.
// See: "Stable Intrinsic Auto-Calibration from Fundamental Matrices of Devices
// with Uncorrelated Camera Parameters", Fetzer et al., WACV 2020.
inline Eigen::Vector4d ComputeFetzerPolynomialCoefficients(
    const Eigen::Vector3d& ai,
    const Eigen::Vector3d& bi,
    const Eigen::Vector3d& aj,
    const Eigen::Vector3d& bj,
    const int u,
    const int v) {
  return {ai(u) * aj(v) - ai(v) * aj(u),
          ai(u) * bj(v) - ai(v) * bj(u),
          bi(u) * aj(v) - bi(v) * aj(u),
          bi(u) * bj(v) - bi(v) * bj(u)};
}

// Decompose the fundamental matrix (adjusted by principal points) via SVD and
// compute the polynomial coefficients for the Fetzer focal length method.
// Returns three coefficient vectors used to estimate the two focal lengths.
inline std::array<Eigen::Vector4d, 2> DecomposeFundamentalMatrixForFetzer(
    const Eigen::Matrix3d& i1_F_i0,
    const Eigen::Vector2d& principal_point0,
    const Eigen::Vector2d& principal_point1) {
  Eigen::Matrix3d K0 = Eigen::Matrix3d::Identity(3, 3);
  K0(0, 2) = principal_point0(0);
  K0(1, 2) = principal_point0(1);

  Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity(3, 3);
  K1(0, 2) = principal_point1(0);
  K1(1, 2) = principal_point1(1);

  // Factoring out the principal points before the SVD appears to be numerically
  // more stable than the method described in the paper.
  const Eigen::Matrix3d i1_G_i0 = K1.transpose() * i1_F_i0 * K0;

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      i1_G_i0, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d& s = svd.singularValues();

  const Eigen::Vector3d v0 = svd.matrixV().col(0);
  const Eigen::Vector3d v1 = svd.matrixV().col(1);

  const Eigen::Vector3d u0 = svd.matrixU().col(0);
  const Eigen::Vector3d u1 = svd.matrixU().col(1);

  // Equation 11. Notice there is a sign error in the paper.
  // Equation 8 shows the sign in aj(1) and bj(1) correctly.
  const Eigen::Vector3d ai(s(0) * s(0) * (v0(0) * v0(0) + v0(1) * v0(1)),
                           s(0) * s(1) * (v0(0) * v1(0) + v0(1) * v1(1)),
                           s(1) * s(1) * (v1(0) * v1(0) + v1(1) * v1(1)));

  const Eigen::Vector3d aj(u1(0) * u1(0) + u1(1) * u1(1),
                           -(u0(0) * u1(0) + u0(1) * u1(1)),
                           u0(0) * u0(0) + u0(1) * u0(1));

  const Eigen::Vector3d bi(s(0) * s(0) * v0(2) * v0(2),
                           s(0) * s(1) * v0(2) * v1(2),
                           s(1) * s(1) * v1(2) * v1(2));

  const Eigen::Vector3d bj(u1(2) * u1(2), -(u0(2) * u1(2)), u0(2) * u0(2));

  // Equation 12.
  // Experiments showed that the d02 term is not useful.
  // The d10, d21m d20 are redundant to d01, d12, d02.
  const Eigen::Vector4d d01 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 1, 0);
  const Eigen::Vector4d d12 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 2, 1);
  return {d01, d12};
}

template <typename T>
inline T ComputeFetzerResidual1(const Eigen::Vector<T, 4>& d,
                                const T& fi_sq,
                                const T& fj_sq) {
  // Equation 13.
  T denom = fj_sq * d(0) + d(1);
  denom = denom == T(0) ? T(1e-6) : denom;
  const T K1 = -(fj_sq * d(2) + d(3)) / denom;
  return (fi_sq - K1) / fi_sq;
}

template <typename T>
inline T ComputeFetzerResidual2(const Eigen::Vector<T, 4>& d,
                                const T& fi_sq,
                                const T& fj_sq) {
  // Equation 14.
  T denom = fi_sq * d(0) + d(2);
  denom = denom == T(0) ? T(1e-6) : denom;
  const T K2 = -(fi_sq * d(1) + d(3)) / denom;
  return (fj_sq - K2) / fj_sq;
}

// Cost functor for estimating focal lengths from the fundamental matrix using
// the Fetzer method. Used when two images have different cameras (different
// focal lengths). The residual measures the relative error between the
// estimated and expected focal lengths based on the fundamental matrix
// constraint.
class FetzerFocalLengthCostFunctor {
 public:
  FetzerFocalLengthCostFunctor(const Eigen::Matrix3d& j_F_i,
                               const Eigen::Vector2d& principal_point_i,
                               const Eigen::Vector2d& principal_point_j)
      : coeffs_(DecomposeFundamentalMatrixForFetzer(
            j_F_i, principal_point_i, principal_point_j)) {}

  static ceres::CostFunction* Create(const Eigen::Matrix3d& j_F_i,
                                     const Eigen::Vector2d& principal_point_i,
                                     const Eigen::Vector2d& principal_point_j) {
    return new ceres::
        AutoDiffCostFunction<FetzerFocalLengthCostFunctor, 2, 1, 1>(
            new FetzerFocalLengthCostFunctor(
                j_F_i, principal_point_i, principal_point_j));
  }

  template <typename T>
  bool operator()(const T* const focal_length_i,
                  const T* const focal_length_j,
                  T* residuals) const {
    const T fi_sq = focal_length_i[0] * focal_length_i[0];
    const T fj_sq = focal_length_j[0] * focal_length_j[0];
    const Eigen::Vector<T, 4> d01 = coeffs_[0].cast<T>();
    residuals[0] = ComputeFetzerResidual1(d01, fi_sq, fj_sq);
    const Eigen::Vector<T, 4> d12 = coeffs_[1].cast<T>();
    residuals[1] = ComputeFetzerResidual2(d12, fi_sq, fj_sq);
    return true;
  }

 private:
  const std::array<Eigen::Vector4d, 2> coeffs_;
};

// Cost functor for estimating focal length from the fundamental matrix using
// the Fetzer method. Used when two images share the same camera (same focal
// length). The residual measures the relative error between the estimated and
// expected focal length based on the fundamental matrix constraint.
class FetzerFocalLengthSameCameraCostFunctor {
 public:
  FetzerFocalLengthSameCameraCostFunctor(const Eigen::Matrix3d& j_F_i,
                                         const Eigen::Vector2d& principal_point)
      : coeffs_(DecomposeFundamentalMatrixForFetzer(
            j_F_i, principal_point, principal_point)) {}

  static ceres::CostFunction* Create(const Eigen::Matrix3d& j_F_i,
                                     const Eigen::Vector2d& principal_point) {
    return new ceres::
        AutoDiffCostFunction<FetzerFocalLengthSameCameraCostFunctor, 2, 1>(
            new FetzerFocalLengthSameCameraCostFunctor(j_F_i, principal_point));
  }

  template <typename T>
  bool operator()(const T* const focal_length, T* residuals) const {
    const T f_sq = focal_length[0] * focal_length[0];
    const Eigen::Vector<T, 4> d01 = coeffs_[0].cast<T>();
    residuals[0] = ComputeFetzerResidual1(d01, f_sq, f_sq);
    const Eigen::Vector<T, 4> d12 = coeffs_[1].cast<T>();
    residuals[1] = ComputeFetzerResidual2(d12, f_sq, f_sq);
    return true;
  }

 private:
  const std::array<Eigen::Vector4d, 2> coeffs_;
};

}  // namespace colmap
