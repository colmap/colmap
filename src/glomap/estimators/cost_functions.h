
#pragma once

#include "colmap/estimators/cost_function_utils.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace glomap {

// Computes the error between a translation direction and the direction formed
// from two positions such that: t_ij - scale * (p_j - p_i) is minimized.
// The positions can either be two camera centers or one camera center and one
// 3D point.
struct BATAPairwiseDirectionCostFunctor {
  explicit BATAPairwiseDirectionCostFunctor(
      const Eigen::Vector3d& pos2_from_pos1_dir)
      : pos2_from_pos1_dir_(pos2_from_pos1_dir) {}

  template <typename T>
  bool operator()(const T* pos1,
                  const T* pos2,
                  const T* scale,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec = pos2_from_pos1_dir_.cast<T>() -
                    scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(pos2) -
                                Eigen::Map<const Eigen::Matrix<T, 3, 1>>(pos1));
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& pos2_from_pos1_dir) {
    return (
        new ceres::
            AutoDiffCostFunction<BATAPairwiseDirectionCostFunctor, 3, 3, 3, 1>(
                new BATAPairwiseDirectionCostFunctor(pos2_from_pos1_dir)));
  }

  const Eigen::Vector3d pos2_from_pos1_dir_;
};

// Computes the error between a translation direction and the direction formed
// from a camera (c) and 3D point (p), such that:
// t_ij - scale * (p - c + rig_scale * t_rig) is minimized.
struct RigBATAPairwiseDirectionCostFunctor {
  RigBATAPairwiseDirectionCostFunctor(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Vector3d& cam_from_rig_translation)
      : cam_from_point3D_dir_(cam_from_point3D_dir),
        cam_from_rig_translation_(cam_from_rig_translation) {}

  template <typename T>
  bool operator()(const T* point3D,
                  const T* rig_in_world,
                  const T* scale,
                  const T* rig_scale,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        cam_from_point3D_dir_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(point3D) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(rig_in_world) +
                    rig_scale[0] * cam_from_rig_translation_.cast<T>());
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Vector3d& cam_from_rig_translation) {
    return (new ceres::AutoDiffCostFunction<RigBATAPairwiseDirectionCostFunctor,
                                            3,
                                            3,
                                            3,
                                            1,
                                            1>(
        new RigBATAPairwiseDirectionCostFunctor(cam_from_point3D_dir,
                                                cam_from_rig_translation)));
  }

  const Eigen::Vector3d cam_from_point3D_dir_;
  const Eigen::Vector3d cam_from_rig_translation_;
};

// Computes the error between a translation direction and the direction formed
// from a camera (c) and 3D point (p) with unknown rig translation, such that:
// t_ij - scale * (p - c + t_rig) is minimized.
struct RigUnknownBATAPairwiseDirectionCostFunctor {
  RigUnknownBATAPairwiseDirectionCostFunctor(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Quaterniond& rig_from_world_rot)
      : cam_from_point3D_dir_(cam_from_point3D_dir),
        world_from_rig_rot_(rig_from_world_rot.inverse()) {}

  // The error is given by the position error described above.
  template <typename T>
  bool operator()(const T* point3D,
                  const T* rig_in_world,
                  const T* cam_in_rig,
                  const T* scale,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> cam_from_rig_translation =
        world_from_rig_rot_.cast<T>() *
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(cam_in_rig);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        cam_from_point3D_dir_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(point3D) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(rig_in_world) -
                    cam_from_rig_translation);
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Quaterniond& rig_from_world_rot) {
    return (new ceres::AutoDiffCostFunction<
            RigUnknownBATAPairwiseDirectionCostFunctor,
            3,
            3,
            3,
            3,
            1>(new RigUnknownBATAPairwiseDirectionCostFunctor(
        cam_from_point3D_dir, rig_from_world_rot)));
  }

  const Eigen::Vector3d cam_from_point3D_dir_;
  const Eigen::Quaterniond world_from_rig_rot_;
};

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
  // Equation 12.
  return {ai(u) * aj(v) - ai(v) * aj(u),
          ai(u) * bj(v) - ai(v) * bj(u),
          bi(u) * aj(v) - bi(v) * aj(u),
          bi(u) * bj(v) - bi(v) * bj(u)};
}

// Decompose the fundamental matrix (adjusted by principal points) via SVD and
// compute the polynomial coefficients for the Fetzer focal length method.
// Returns three coefficient vectors used to estimate the two focal lengths.
inline std::array<Eigen::Vector4d, 6> DecomposeFundamentalMatrixForFetzer(
    const Eigen::Matrix3d& j_F_i,
    const Eigen::Vector2d& principal_point_i,
    const Eigen::Vector2d& principal_point_j) {
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      j_F_i, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d& s = svd.singularValues();

  const Eigen::Vector3d& v1 = svd.matrixV().col(0);
  const Eigen::Vector3d& v2 = svd.matrixV().col(1);

  const Eigen::Vector3d& u1 = svd.matrixU().col(0);
  const Eigen::Vector3d& u2 = svd.matrixU().col(1);

  const double s1 = s(0);
  const double s2 = s(1);

  const double v11 = v1(0);
  const double v12 = v1(1);
  const double v21 = v2(0);
  const double v22 = v2(1);

  const double u11 = u1(0);
  const double u12 = u1(1);
  const double u21 = u2(0);
  const double u22 = u2(1);

  const double cpi_v1 = principal_point_i.homogeneous().dot(v1);
  const double cpi_v2 = principal_point_i.homogeneous().dot(v2);
  const double cpj_u1 = principal_point_j.homogeneous().dot(v1);
  const double cpj_u2 = principal_point_j.homogeneous().dot(v2);

  // Equation 11.
  const Eigen::Vector3d ai(s1 * s1 * (v11 * v11 + v12 * v12),
                           s1 * s2 * (v11 * v21 + v12 * v22),
                           s2 * s2 * (v21 * v21 + v22 * v22));

  const Eigen::Vector3d aj(
      u21 * u21 + u22 * u22, u11 * u21 + u12 * u22, u11 * u11 + u12 * u12);

  const Eigen::Vector3d bi(s1 * s1 * cpi_v1 * cpi_v1,
                           s1 * s2 * cpi_v1 * cpi_v2,
                           s2 * s2 * cpi_v2 * cpi_v2);

  const Eigen::Vector3d bj(cpj_u2 * cpj_u2, cpj_u1 * cpj_u2, cpj_u1 * cpj_u1);

  // Equation 12.
  const Eigen::Vector4d d01 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 0, 1);
  const Eigen::Vector4d d10 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 1, 0);
  const Eigen::Vector4d d02 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 0, 2);
  const Eigen::Vector4d d20 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 2, 0);
  const Eigen::Vector4d d12 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 1, 2);
  const Eigen::Vector4d d21 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 2, 1);
  return {d01, d10, d02, d20, d12, d21};
}

template <typename T>
inline T ComputeFetzerResidual1(const Eigen::Vector<T, 4>& d,
                                const T& fi_sq,
                                const T& fj_sq) {
  // Equation 13.
  T denom = (fj_sq * d(0) + d(1));
  denom = denom == T(0) ? T(1e-6) : denom;
  const T K1 = -(fj_sq * d(2) + d(3)) / denom;
  return (fi_sq - K1) / fi_sq;
}

template <typename T>
inline T ComputeFetzerResidual2(const Eigen::Vector<T, 4>& d,
                                const T& fi_sq,
                                const T& fj_sq) {
  // Equation 14.
  T denom = (fi_sq * d(0) + d(2));
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
        AutoDiffCostFunction<FetzerFocalLengthCostFunctor, 12, 1, 1>(
            new FetzerFocalLengthCostFunctor(
                j_F_i, principal_point_i, principal_point_j));
  }

  template <typename T>
  bool operator()(const T* const focal_length_i,
                  const T* const focal_length_j,
                  T* residuals) const {
    const T fi_sq = focal_length_i[0] * focal_length_i[0];
    const T fj_sq = focal_length_j[0] * focal_length_j[0];

    // The total of 12 reisudals only contribute 2 independent constraints.
    // We still compute all of them to obtain a "quasi-symmetric" and smooth
    // energy landscape according to the paper.
    for (int i = 0; i < 6; ++i) {
      const Eigen::Vector<T, 4> di = coeffs_[i].cast<T>();
      residuals[i] = ComputeFetzerResidual1(di, fi_sq, fj_sq);
      residuals[2 * i] = ComputeFetzerResidual2(di, fi_sq, fj_sq);
    }

    return true;
  }

 private:
  const std::array<Eigen::Vector4d, 6> coeffs_;
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

    // The total of 12 reisudals only contribute 2 independent constraints.
    // We still compute all of them to obtain a "quasi-symmetric" and smooth
    // energy landscape according to the paper.
    for (int i = 0; i < 6; ++i) {
      const Eigen::Vector<T, 4> di = coeffs_[i].cast<T>();
      residuals[i] = ComputeFetzerResidual1(di, f_sq, f_sq);
      residuals[2 * i] = ComputeFetzerResidual2(di, f_sq, f_sq);
    }

    return true;
  }

 private:
  const std::array<Eigen::Vector4d, 6> coeffs_;
};

// Computes residual between estimated gravity and measured gravity prior.
// This is a type alias to the generic NormalPriorCostFunctor for 3D vectors.
using GravityCostFunctor = colmap::NormalPriorCostFunctor<3>;

}  // namespace glomap
