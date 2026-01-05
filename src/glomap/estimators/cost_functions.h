
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
inline std::array<Eigen::Vector4d, 3> DecomposeFundamentalMatrixForFetzer(
    const Eigen::Matrix3d& j_F_i,
    const Eigen::Vector2d& principal_point_i,
    const Eigen::Vector2d& principal_point_j) {
  Eigen::Matrix3d K0 = Eigen::Matrix3d::Identity(3, 3);
  K0(0, 2) = principal_point_i(0);
  K0(1, 2) = principal_point_i(1);

  Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity(3, 3);
  K1(0, 2) = principal_point_j(0);
  K1(1, 2) = principal_point_j(1);

  // Factoring out the principal points before the SVD appears to be numerically
  // more stable than the method described in the paper.
  const Eigen::Matrix3d i1_G_i0 = K1.transpose() * j_F_i * K0;

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      i1_G_i0, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d& s = svd.singularValues();

  const Eigen::Vector3d& v1 = svd.matrixV().col(0);
  const Eigen::Vector3d& v2 = svd.matrixV().col(1);

  const Eigen::Vector3d& u1 = svd.matrixU().col(0);
  const Eigen::Vector3d& u2 = svd.matrixU().col(1);

  const double s1 = s(0);
  const double s2 = s(1);

  const double v11 = v1(0);
  const double v12 = v1(1);
  const double v13 = v1(2);
  const double v21 = v2(0);
  const double v22 = v2(1);
  const double v23 = v2(2);

  const double u11 = u1(0);
  const double u12 = u1(1);
  const double u13 = u1(2);
  const double u21 = u2(0);
  const double u22 = u2(1);
  const double u23 = u2(2);

  // Equation 11. Notice there is a sign error in the paper.
  // Equation 8 shows the sign in aj(1) and bj(1) correctly.
  const Eigen::Vector3d ai(s1 * s1 * (v11 * v11 + v12 * v12),
                           s1 * s2 * (v11 * v21 + v12 * v22),
                           s2 * s2 * (v21 * v21 + v22 * v22));

  const Eigen::Vector3d aj(
      u21 * u21 + u22 * u22, -(u11 * u21 + u12 * u22), u11 * u11 + u12 * u12);

  const Eigen::Vector3d bi(
      s1 * s1 * v13 * v13, s1 * s2 * v13 * v23, s2 * s2 * v23 * v23);

  const Eigen::Vector3d bj(u23 * u23, -(u13 * u23), u13 * u13);

  // Equation 12. d12, d20, d21 are redundant.
  const Eigen::Vector4d d01 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 0, 1);
  const Eigen::Vector4d d02 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 0, 2);
  const Eigen::Vector4d d12 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 1, 2);
  return {d01, d02, d12};
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
        AutoDiffCostFunction<FetzerFocalLengthCostFunctor, 6, 1, 1>(
            new FetzerFocalLengthCostFunctor(
                j_F_i, principal_point_i, principal_point_j));
  }

  template <typename T>
  bool operator()(const T* const focal_length_i,
                  const T* const focal_length_j,
                  T* residuals) const {
    const T fi_sq = focal_length_i[0] * focal_length_i[0];
    const T fj_sq = focal_length_j[0] * focal_length_j[0];

    // The total of 6 residuals only contribute 2 independent constraints.
    // We still compute all of them to obtain a "quasi-symmetric" and smooth
    // energy landscape according to the paper.
    for (int i = 0; i < 3; ++i) {
      const Eigen::Vector<T, 4> di = coeffs_[i].cast<T>();
      residuals[i] = ComputeFetzerResidual1(di, fi_sq, fj_sq);
      residuals[2 * i] = ComputeFetzerResidual2(di, fi_sq, fj_sq);
    }

    // residuals[0] = ComputeFetzerResidual1(
    //     Eigen::Vector<T, 4>(coeffs_[0].cast<T>()), fi_sq, fj_sq);
    // residuals[1] = ComputeFetzerResidual2(
    //     Eigen::Vector<T, 4>(coeffs_[2].cast<T>()), fi_sq, fj_sq);
    // for (int i = 2; i < 6; ++i) {
    //   residuals[i] = T(0.0);
    // }

    return true;
  }

 private:
  const std::array<Eigen::Vector4d, 3> coeffs_;
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
        AutoDiffCostFunction<FetzerFocalLengthSameCameraCostFunctor, 6, 1>(
            new FetzerFocalLengthSameCameraCostFunctor(j_F_i, principal_point));
  }

  template <typename T>
  bool operator()(const T* const focal_length, T* residuals) const {
    const T f_sq = focal_length[0] * focal_length[0];

    // The total of 6 residuals only contribute 2 independent constraints.
    // We still compute all of them to obtain a "quasi-symmetric" and smooth
    // energy landscape according to the paper.
    for (int i = 0; i < 3; ++i) {
      const Eigen::Vector<T, 4> di = coeffs_[i].cast<T>();
      residuals[i] = ComputeFetzerResidual1(di, f_sq, f_sq);
      residuals[2 * i] = ComputeFetzerResidual2(di, f_sq, f_sq);
    }

    return true;
  }

 private:
  const std::array<Eigen::Vector4d, 3> coeffs_;
};

namespace old {

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
inline std::array<Eigen::Vector4d, 3> DecomposeFundamentalMatrixForFetzer(
    const Eigen::Matrix3d& i1_F_i0,
    const Eigen::Vector2d& principal_point0,
    const Eigen::Vector2d& principal_point1) {
  Eigen::Matrix3d K0 = Eigen::Matrix3d::Identity(3, 3);
  K0(0, 2) = principal_point0(0);
  K0(1, 2) = principal_point0(1);

  Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity(3, 3);
  K1(0, 2) = principal_point1(0);
  K1(1, 2) = principal_point1(1);

  const Eigen::Matrix3d i1_G_i0 = K1.transpose() * i1_F_i0 * K0;

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      i1_G_i0, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d& s = svd.singularValues();

  const Eigen::Vector3d v0 = svd.matrixV().col(0);
  const Eigen::Vector3d v1 = svd.matrixV().col(1);

  const Eigen::Vector3d u0 = svd.matrixU().col(0);
  const Eigen::Vector3d u1 = svd.matrixU().col(1);

  // LOG(INFO) << "s: " << s.transpose();
  // LOG(INFO) << "v1: " << v0.transpose();
  // LOG(INFO) << "v2: " << v1.transpose();
  // LOG(INFO) << "u1: " << u0.transpose();
  // LOG(INFO) << "u2: " << u1.transpose();

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

  // LOG(INFO) << "ai: " << ai.transpose();
  // LOG(INFO) << "aj: " << aj.transpose();
  // LOG(INFO) << "bi: " << bi.transpose();
  // LOG(INFO) << "bj: " << bj.transpose();

  const Eigen::Vector4d d01 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 1, 0);
  const Eigen::Vector4d d02 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 0, 2);
  const Eigen::Vector4d d12 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 2, 1);
  return {d01, d02, d12};
}

// Cost functor for estimating focal lengths from the fundamental matrix using
// the Fetzer method. Used when two images have different cameras (different
// focal lengths). The residual measures the relative error between the
// estimated and expected focal lengths based on the fundamental matrix
// constraint.
class FetzerFocalLengthCostFunctor {
 public:
  FetzerFocalLengthCostFunctor(const Eigen::Matrix3d& i1_F_i0,
                               const Eigen::Vector2d& principal_point0,
                               const Eigen::Vector2d& principal_point1)
      : coeffs_(DecomposeFundamentalMatrixForFetzer(
            i1_F_i0, principal_point0, principal_point1)) {}

  static ceres::CostFunction* Create(const Eigen::Matrix3d& i1_F_i0,
                                     const Eigen::Vector2d& principal_point0,
                                     const Eigen::Vector2d& principal_point1) {
    return new ceres::
        AutoDiffCostFunction<FetzerFocalLengthCostFunctor, 2, 1, 1>(
            new FetzerFocalLengthCostFunctor(
                i1_F_i0, principal_point0, principal_point1));
  }

  template <typename T>
  bool operator()(const T* const fi_, const T* const fj_, T* residuals) const {
    const Eigen::Vector<T, 4> d01_ = coeffs_[0].cast<T>();
    const Eigen::Vector<T, 4> d12_ = coeffs_[2].cast<T>();

    const T fi = fi_[0];
    const T fj = fj_[0];

    T di = (fj * fj * d01_(0) + d01_(1));
    T dj = (fi * fi * d12_(0) + d12_(2));
    di = di == T(0) ? T(1e-6) : di;
    dj = dj == T(0) ? T(1e-6) : dj;

    const T K0_01 = -(fj * fj * d01_(2) + d01_(3)) / di;
    const T K1_12 = -(fi * fi * d12_(1) + d12_(3)) / dj;

    residuals[0] = (fi * fi - K0_01) / (fi * fi);
    residuals[1] = (fj * fj - K1_12) / (fj * fj);

    return true;
  }

 private:
  const std::array<Eigen::Vector4d, 3> coeffs_;
};

}  // namespace old

// Computes residual between estimated gravity and measured gravity prior.
// This is a type alias to the generic NormalPriorCostFunctor for 3D vectors.
using GravityCostFunctor = colmap::NormalPriorCostFunctor<3>;

}  // namespace glomap
