// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <ceres/ceres.h>

namespace colmap {

// Cost functor for estimating the focal lengths of two pinhole cameras with
// known principal points from their fundamental matrix. The 2-dimensional
// residual has the squared norm
//
//    dist^2(A, E) / ||A||_F^2 = (s1 - s2)^2 / (2 (s1^2 + s2^2)),
//
// i.e., the normalized squared Frobenius distance of the calibrated fundamental
// matrix A = K2^T F K1 with singular values s1 >= s2 to the essential manifold
// E. Instead of an SVD, the residual is evaluated via an isometric reduction of
// A onto the orthogonal complements of its left and right null spaces, where
// the essential constraint becomes membership in the scaled 2x2 orthogonal
// matrices. In contrast to the equivalent scalar residual sqrt(dist^2/||A||^2),
// both components are smooth in the focal lengths and give a full-rank
// Gauss-Newton approximation at the solution.
//
// The parameters are log focal lengths, which enforces positivity.
//
// See: "Smooth Essential-Manifold Residuals for View-Graph Calibration".
class EssentialManifoldFocalLengthCostFunctor {
 public:
  // The initial focal lengths only determine on which of the two components of
  // the scaled orthogonal matrices the residual is measured. Since this choice
  // cannot change for positive focal lengths, any positive values are valid.
  EssentialManifoldFocalLengthCostFunctor(
      const Eigen::Matrix3d& cam2_F_cam1,
      const Eigen::Vector2d& principal_point1,
      const Eigen::Vector2d& principal_point2,
      double initial_focal_length1,
      double initial_focal_length2)
      : principal_point1_(principal_point1),
        principal_point2_(principal_point2) {
    // Project to the closest rank-2 matrix, so that the null vectors are exact
    // and the reduction is an isometry.
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        cam2_F_cam1, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_values = svd.singularValues();
    singular_values(2) = 0;
    F_ = svd.matrixU() * singular_values.asDiagonal() *
         svd.matrixV().transpose();
    left_null_ = svd.matrixU().col(2);
    right_null_ = svd.matrixV().col(2);
    left_axis_ = CompletionAxis(left_null_, principal_point2_);
    right_axis_ = CompletionAxis(right_null_, principal_point1_);
    F_right_axis_ = F_ * right_axis_;
    Ft_left_axis_ = F_.transpose() * left_axis_;
    left_axis_F_right_axis_ = left_axis_.dot(F_right_axis_);
    const Eigen::Matrix2d C = ReducedCalibratedFundamental(
        initial_focal_length1, initial_focal_length2);
    sign_ = C.determinant() >= 0 ? 1 : -1;
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d& cam2_F_cam1,
                                     const Eigen::Vector2d& principal_point1,
                                     const Eigen::Vector2d& principal_point2,
                                     double initial_focal_length1,
                                     double initial_focal_length2) {
    return new ceres::
        AutoDiffCostFunction<EssentialManifoldFocalLengthCostFunctor, 2, 1, 1>(
            new EssentialManifoldFocalLengthCostFunctor(cam2_F_cam1,
                                                        principal_point1,
                                                        principal_point2,
                                                        initial_focal_length1,
                                                        initial_focal_length2));
  }

  template <typename T>
  bool operator()(const T* const log_focal_length1,
                  const T* const log_focal_length2,
                  T* residuals) const {
    const Eigen::Matrix<T, 2, 2> C = ReducedCalibratedFundamental(
        ceres::exp(log_focal_length1[0]), ceres::exp(log_focal_length2[0]));
    // For rank-2 A, ||C||_F = ||A||_F.
    const T denom = T(std::sqrt(2.0)) * C.norm();
    residuals[0] = (C(0, 0) - T(sign_) * C(1, 1)) / denom;
    residuals[1] = (C(0, 1) + T(sign_) * C(1, 0)) / denom;
    return true;
  }

  // Evaluates the residuals at the given (not log) focal lengths.
  Eigen::Vector2d Evaluate(double focal_length1, double focal_length2) const {
    const double log_focal_length1 = std::log(focal_length1);
    const double log_focal_length2 = std::log(focal_length2);
    Eigen::Vector2d residuals;
    (*this)(&log_focal_length1, &log_focal_length2, residuals.data());
    return residuals;
  }

 private:
  // Unit vector orthogonal to K^-1 * n for any focal length f > 0. With the
  // principal point (u, v) fixed, K^-1 * n ~ (w0, w1, f * n2) with
  // w = (n0 - u * n2, n1 - v * n2, 0), which always lies in span(w, e3), so
  // w x e3 is orthogonal to it. If w = 0, K^-1 * n ~ e3 and any unit vector in
  // the xy-plane is orthogonal to it.
  static Eigen::Vector3d CompletionAxis(
      const Eigen::Vector3d& n, const Eigen::Vector2d& principal_point) {
    const Eigen::Vector3d axis(
        n(1) - principal_point(1) * n(2), principal_point(0) * n(2) - n(0), 0);
    const double norm = axis.norm();
    return norm > 1e-12 ? Eigen::Vector3d(axis / norm)
                        : Eigen::Vector3d::UnitX();
  }

  // Computes C = Pa * A * Pb^T, where the rows of Pa and Pb are orthonormal
  // bases of the complements of the left and right null spaces of A.
  template <typename T>
  Eigen::Matrix<T, 2, 2> ReducedCalibratedFundamental(
      const T& focal_length1, const T& focal_length2) const {
    using Vector3T = Eigen::Matrix<T, 3, 1>;

    const auto calibrate = [](const Vector3T& x,
                              const T& focal_length,
                              const Eigen::Vector2d& principal_point) {
      return Vector3T(focal_length * x(0) + T(principal_point(0)) * x(2),
                      focal_length * x(1) + T(principal_point(1)) * x(2),
                      x(2));
    };
    const auto normalized_uncalibrate =
        [](const Eigen::Vector3d& x,
           const T& focal_length,
           const Eigen::Vector2d& principal_point) {
          Vector3T y((T(x(0)) - T(principal_point(0) * x(2))) / focal_length,
                     (T(x(1)) - T(principal_point(1) * x(2))) / focal_length,
                     T(x(2)));
          y.normalize();
          return y;
        };

    // The null vectors of A = K2^T F K1 are K2^-1 * left_null and
    // K1^-1 * right_null, and their orthonormal complements follow from the
    // fixed completion axes without Gram-Schmidt.
    const Vector3T left_axis2 =
        normalized_uncalibrate(left_null_, focal_length2, principal_point2_)
            .cross(left_axis_.cast<T>());
    const Vector3T right_axis2 =
        normalized_uncalibrate(right_null_, focal_length1, principal_point1_)
            .cross(right_axis_.cast<T>());
    const Vector3T K2_left_axis2 =
        calibrate(left_axis2, focal_length2, principal_point2_);
    const Vector3T K1_right_axis2 =
        calibrate(right_axis2, focal_length1, principal_point1_);

    // C_ij = (K2 * left_axis_i)^T * F * (K1 * right_axis_j), where
    // K * axis = f * axis for the completion axes in the xy-plane.
    Eigen::Matrix<T, 2, 2> C;
    C(0, 0) = focal_length1 * focal_length2 * T(left_axis_F_right_axis_);
    C(0, 1) = focal_length2 * Ft_left_axis_.cast<T>().dot(K1_right_axis2);
    C(1, 0) = focal_length1 * K2_left_axis2.dot(F_right_axis_.cast<T>());
    C(1, 1) = K2_left_axis2.dot(F_.cast<T>() * K1_right_axis2);
    return C;
  }

  Eigen::Vector2d principal_point1_;
  Eigen::Vector2d principal_point2_;
  Eigen::Matrix3d F_;
  Eigen::Vector3d left_null_;
  Eigen::Vector3d right_null_;
  Eigen::Vector3d left_axis_;
  Eigen::Vector3d right_axis_;
  // Precomputed terms of C that only depend on the completion axes.
  Eigen::Vector3d F_right_axis_;
  Eigen::Vector3d Ft_left_axis_;
  double left_axis_F_right_axis_ = 0;
  double sign_ = 1;
};

// Same as EssentialManifoldFocalLengthCostFunctor for two images that share
// the same camera and thus the same focal length.
class EssentialManifoldFocalLengthSameCameraCostFunctor {
 public:
  EssentialManifoldFocalLengthSameCameraCostFunctor(
      const Eigen::Matrix3d& cam2_F_cam1,
      const Eigen::Vector2d& principal_point,
      double initial_focal_length)
      : functor_(cam2_F_cam1,
                 principal_point,
                 principal_point,
                 initial_focal_length,
                 initial_focal_length) {}

  static ceres::CostFunction* Create(const Eigen::Matrix3d& cam2_F_cam1,
                                     const Eigen::Vector2d& principal_point,
                                     double initial_focal_length) {
    return new ceres::AutoDiffCostFunction<
        EssentialManifoldFocalLengthSameCameraCostFunctor,
        2,
        1>(new EssentialManifoldFocalLengthSameCameraCostFunctor(
        cam2_F_cam1, principal_point, initial_focal_length));
  }

  template <typename T>
  bool operator()(const T* const log_focal_length, T* residuals) const {
    return functor_(log_focal_length, log_focal_length, residuals);
  }

 private:
  const EssentialManifoldFocalLengthCostFunctor functor_;
};

}  // namespace colmap
