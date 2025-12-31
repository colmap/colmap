
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
// Reference: Zhuang et al., "Baseline Desensitizing In Translation Averaging",
// CVPR 2018.
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
    return (new ceres::AutoDiffCostFunction<BATAPairwiseDirectionCostFunctor,
                                            3,
                                            3,
                                            3,
                                            1>(
        new BATAPairwiseDirectionCostFunctor(pos2_from_pos1_dir)));
  }

  const Eigen::Vector3d pos2_from_pos1_dir_;
};

// Computes the error between a translation direction and the direction formed
// from a camera (c) and 3D point (p) with constant rig extrinsics, such that:
// t_ij - scale * (p - c + rig_scale * t_rig) is minimized.
struct RigBATAPairwiseDirectionConstantRigCostFunctor {
  RigBATAPairwiseDirectionConstantRigCostFunctor(
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
    return (new ceres::AutoDiffCostFunction<
            RigBATAPairwiseDirectionConstantRigCostFunctor,
            3,
            3,
            3,
            1,
            1>(new RigBATAPairwiseDirectionConstantRigCostFunctor(
        cam_from_point3D_dir, cam_from_rig_translation)));
  }

  const Eigen::Vector3d cam_from_point3D_dir_;
  const Eigen::Vector3d cam_from_rig_translation_;
};

// Computes the error between a translation direction and the direction formed
// from a camera (c) and 3D point (p) with variable rig extrinsics, such that:
// t_ij - scale * (p - c + t_rig) is minimized.
struct RigBATAPairwiseDirectionCostFunctor {
  RigBATAPairwiseDirectionCostFunctor(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Quaterniond& rig_from_world_rot)
      : cam_from_point3D_dir_(cam_from_point3D_dir),
        world_from_rig_rot_(rig_from_world_rot.inverse()) {}

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
    return (new ceres::AutoDiffCostFunction<RigBATAPairwiseDirectionCostFunctor,
                                            3,
                                            3,
                                            3,
                                            3,
                                            1>(
        new RigBATAPairwiseDirectionCostFunctor(cam_from_point3D_dir,
                                                rig_from_world_rot)));
  }

  const Eigen::Vector3d cam_from_point3D_dir_;
  const Eigen::Quaterniond world_from_rig_rot_;
};

// Compute polynomial coefficients from cross-products of SVD-derived vectors
// for the Fetzer focal length estimation method. The coefficients encode the
// relationship between the two focal lengths derived from the fundamental
// matrix constraint.
// Reference: Sweeney et al., "Optimizing the Viewing Graph for Structure-from-
// Motion", ICCV 2015.
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
    const Eigen::Matrix3d& i1_G_i0) {
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      i1_G_i0, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d& s = svd.singularValues();

  const Eigen::Vector3d v_0 = svd.matrixV().col(0);
  const Eigen::Vector3d v_1 = svd.matrixV().col(1);

  const Eigen::Vector3d u_0 = svd.matrixU().col(0);
  const Eigen::Vector3d u_1 = svd.matrixU().col(1);

  const Eigen::Vector3d ai(s(0) * s(0) * (v_0(0) * v_0(0) + v_0(1) * v_0(1)),
                           s(0) * s(1) * (v_0(0) * v_1(0) + v_0(1) * v_1(1)),
                           s(1) * s(1) * (v_1(0) * v_1(0) + v_1(1) * v_1(1)));

  const Eigen::Vector3d aj(u_1(0) * u_1(0) + u_1(1) * u_1(1),
                           -(u_0(0) * u_1(0) + u_0(1) * u_1(1)),
                           u_0(0) * u_0(0) + u_0(1) * u_0(1));

  const Eigen::Vector3d bi(s(0) * s(0) * v_0(2) * v_0(2),
                           s(0) * s(1) * v_0(2) * v_1(2),
                           s(1) * s(1) * v_1(2) * v_1(2));

  const Eigen::Vector3d bj(
      u_1(2) * u_1(2), -(u_0(2) * u_1(2)), u_0(2) * u_0(2));

  const Eigen::Vector4d d_01 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 1, 0);
  const Eigen::Vector4d d_02 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 0, 2);
  const Eigen::Vector4d d_12 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 2, 1);
  return {d_01, d_02, d_12};
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
                               const Eigen::Vector2d& principal_point1) {
    Eigen::Matrix3d K0 = Eigen::Matrix3d::Identity(3, 3);
    K0(0, 2) = principal_point0(0);
    K0(1, 2) = principal_point0(1);

    Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity(3, 3);
    K1(0, 2) = principal_point1(0);
    K1(1, 2) = principal_point1(1);

    const Eigen::Matrix3d i1_G_i0 = K1.transpose() * i1_F_i0 * K0;

    const std::array<Eigen::Vector4d, 3> coeffs =
        DecomposeFundamentalMatrixForFetzer(i1_G_i0);

    d_01_ = coeffs[0];
    d_02_ = coeffs[1];
    d_12_ = coeffs[2];
  }

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
    const Eigen::Vector<T, 4> coeffs_01 = d_01_.cast<T>();
    const Eigen::Vector<T, 4> coeffs_12 = d_12_.cast<T>();

    const T fi = fi_[0];
    const T fj = fj_[0];

    T di = (fj * fj * coeffs_01(0) + coeffs_01(1));
    T dj = (fi * fi * coeffs_12(0) + coeffs_12(2));
    di = di == T(0) ? T(1e-6) : di;
    dj = dj == T(0) ? T(1e-6) : dj;

    const T K0_01 = -(fj * fj * coeffs_01(2) + coeffs_01(3)) / di;
    const T K1_12 = -(fi * fi * coeffs_12(1) + coeffs_12(3)) / dj;

    residuals[0] = (fi * fi - K0_01) / (fi * fi);
    residuals[1] = (fj * fj - K1_12) / (fj * fj);

    return true;
  }

 private:
  Eigen::Vector4d d_01_;
  Eigen::Vector4d d_02_;
  Eigen::Vector4d d_12_;
};

// Cost functor for estimating focal length from the fundamental matrix using
// the Fetzer method. Used when two images share the same camera (same focal
// length). The residual measures the relative error between the estimated and
// expected focal length based on the fundamental matrix constraint.
class FetzerFocalLengthSameCameraCostFunctor {
 public:
  FetzerFocalLengthSameCameraCostFunctor(
      const Eigen::Matrix3d& i1_F_i0, const Eigen::Vector2d& principal_point) {
    Eigen::Matrix3d K0 = Eigen::Matrix3d::Identity(3, 3);
    K0(0, 2) = principal_point(0);
    K0(1, 2) = principal_point(1);

    Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity(3, 3);
    K1(0, 2) = principal_point(0);
    K1(1, 2) = principal_point(1);

    const Eigen::Matrix3d i1_G_i0 = K1.transpose() * i1_F_i0 * K0;

    const std::array<Eigen::Vector4d, 3> coeffs =
        DecomposeFundamentalMatrixForFetzer(i1_G_i0);

    d_01_ = coeffs[0];
    d_02_ = coeffs[1];
    d_12_ = coeffs[2];
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d& i1_F_i0,
                                     const Eigen::Vector2d& principal_point) {
    return new ceres::
        AutoDiffCostFunction<FetzerFocalLengthSameCameraCostFunctor, 2, 1>(
            new FetzerFocalLengthSameCameraCostFunctor(i1_F_i0,
                                                       principal_point));
  }

  template <typename T>
  bool operator()(const T* const fi_, T* residuals) const {
    const Eigen::Vector<T, 4> coeffs_01 = d_01_.cast<T>();
    const Eigen::Vector<T, 4> coeffs_12 = d_12_.cast<T>();

    const T fi = fi_[0];
    const T fj = fi_[0];

    T di = (fj * fj * coeffs_01(0) + coeffs_01(1));
    T dj = (fi * fi * coeffs_12(0) + coeffs_12(2));
    di = di == T(0) ? T(1e-6) : di;
    dj = dj == T(0) ? T(1e-6) : dj;

    const T K0_01 = -(fj * fj * coeffs_01(2) + coeffs_01(3)) / di;
    const T K1_12 = -(fi * fi * coeffs_12(1) + coeffs_12(3)) / dj;

    residuals[0] = (fi * fi - K0_01) / (fi * fi);
    residuals[1] = (fj * fj - K1_12) / (fj * fj);

    return true;
  }

 private:
  Eigen::Vector4d d_01_;
  Eigen::Vector4d d_02_;
  Eigen::Vector4d d_12_;
};

// Computes residual between estimated gravity and measured gravity prior.
// This is a type alias to the generic NormalPriorCostFunctor for 3D vectors.
using GravityCostFunctor = colmap::NormalPriorCostFunctor<3>;

}  // namespace glomap
