
#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace glomap {

// ----------------------------------------
// BATAPairwiseDirectionError
// ----------------------------------------
// Computes the error between a translation direction and the direction formed
// from two positions such that t_ij - scale * (c_j - c_i) is minimized.
struct BATAPairwiseDirectionError {
  BATAPairwiseDirectionError(const Eigen::Vector3d& translation_obs)
      : translation_obs_(translation_obs) {}

  // The error is given by the position error described above.
  template <typename T>
  bool operator()(const T* position1,
                  const T* position2,
                  const T* scale,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        translation_obs_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1));
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs) {
    return (
        new ceres::AutoDiffCostFunction<BATAPairwiseDirectionError, 3, 3, 3, 1>(
            new BATAPairwiseDirectionError(translation_obs)));
  }

  // TODO: add covariance
  const Eigen::Vector3d translation_obs_;
};

// ----------------------------------------
// RigBATAPairwiseDirectionError
// ----------------------------------------
// Computes the error between a translation direction and the direction formed
// from two positions such that t_ij - scale * (c_j - c_i + scale_rig * t_rig)
// is minimized.
struct RigBATAPairwiseDirectionError {
  RigBATAPairwiseDirectionError(const Eigen::Vector3d& translation_obs,
                                const Eigen::Vector3d& translation_rig)
      : translation_obs_(translation_obs), translation_rig_(translation_rig) {}

  // The error is given by the position error described above.
  template <typename T>
  bool operator()(const T* position1,
                  const T* position2,
                  const T* scale,
                  const T* scale_rig,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        translation_obs_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1) +
                    scale_rig[0] * translation_rig_.cast<T>());
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs,
                                     const Eigen::Vector3d& translation_rig) {
    return (
        new ceres::
            AutoDiffCostFunction<RigBATAPairwiseDirectionError, 3, 3, 3, 1, 1>(
                new RigBATAPairwiseDirectionError(translation_obs,
                                                  translation_rig)));
  }

  // TODO: add covariance
  const Eigen::Vector3d translation_obs_;
  const Eigen::Vector3d translation_rig_;  // = c_R_w^T * c_t_r
};

// ----------------------------------------
// RigUnknownBATAPairwiseDirectionError
// ----------------------------------------
// Computes the error between a translation direction and the direction formed
// from three positions such that v - scale * ((X - r_c_w) - r_R_w^T * c_c_r) is
// minimized.
struct RigUnknownBATAPairwiseDirectionError {
  RigUnknownBATAPairwiseDirectionError(
      const Eigen::Vector3d& translation_obs,
      const Eigen::Quaterniond& rig_from_world_rot)
      : translation_obs_(translation_obs),
        rig_from_world_rot_(rig_from_world_rot) {}

  // The error is given by the position error described above.
  template <typename T>
  bool operator()(const T* point3d,
                  const T* rig_from_world_center,
                  const T* cam_from_rig_center,
                  const T* scale,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);

    Eigen::Matrix<T, 3, 1> translation_rig =
        rig_from_world_rot_.toRotationMatrix().transpose() *
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(cam_from_rig_center);

    residuals_vec =
        translation_obs_.cast<T>() -
        scale[0] *
            (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(point3d) -
             Eigen::Map<const Eigen::Matrix<T, 3, 1>>(rig_from_world_center) -
             translation_rig);
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& translation_obs,
      const Eigen::Quaterniond& rig_from_world_rot) {
    return (
        new ceres::AutoDiffCostFunction<RigUnknownBATAPairwiseDirectionError,
                                        3,
                                        3,
                                        3,
                                        3,
                                        1>(
            new RigUnknownBATAPairwiseDirectionError(translation_obs,
                                                     rig_from_world_rot)));
  }

  // TODO: add covariance
  const Eigen::Vector3d translation_obs_;
  const Eigen::Quaterniond rig_from_world_rot_;  // = c_R_w^T * c_t_r
};

// ----------------------------------------
// FetzerFocalLengthCost
// ----------------------------------------
// Below are assets for DMAP by Philipp Lindenberger
inline Eigen::Vector4d fetzer_d(const Eigen::Vector3d& ai,
                                const Eigen::Vector3d& bi,
                                const Eigen::Vector3d& aj,
                                const Eigen::Vector3d& bj,
                                const int u,
                                const int v) {
  Eigen::Vector4d d;
  d.setZero();
  d(0) = ai(u) * aj(v) - ai(v) * aj(u);
  d(1) = ai(u) * bj(v) - ai(v) * bj(u);
  d(2) = bi(u) * aj(v) - bi(v) * aj(u);
  d(3) = bi(u) * bj(v) - bi(v) * bj(u);
  return d;
}

inline std::array<Eigen::Vector4d, 3> fetzer_ds(
    const Eigen::Matrix3d& i1_G_i0) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      i1_G_i0, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d s = svd.singularValues();

  Eigen::Vector3d v_0 = svd.matrixV().col(0);
  Eigen::Vector3d v_1 = svd.matrixV().col(1);

  Eigen::Vector3d u_0 = svd.matrixU().col(0);
  Eigen::Vector3d u_1 = svd.matrixU().col(1);

  Eigen::Vector3d ai =
      Eigen::Vector3d(s(0) * s(0) * (v_0(0) * v_0(0) + v_0(1) * v_0(1)),
                      s(0) * s(1) * (v_0(0) * v_1(0) + v_0(1) * v_1(1)),
                      s(1) * s(1) * (v_1(0) * v_1(0) + v_1(1) * v_1(1)));

  Eigen::Vector3d aj = Eigen::Vector3d(u_1(0) * u_1(0) + u_1(1) * u_1(1),
                                       -(u_0(0) * u_1(0) + u_0(1) * u_1(1)),
                                       u_0(0) * u_0(0) + u_0(1) * u_0(1));

  Eigen::Vector3d bi = Eigen::Vector3d(s(0) * s(0) * v_0(2) * v_0(2),
                                       s(0) * s(1) * v_0(2) * v_1(2),
                                       s(1) * s(1) * v_1(2) * v_1(2));

  Eigen::Vector3d bj =
      Eigen::Vector3d(u_1(2) * u_1(2), -(u_0(2) * u_1(2)), u_0(2) * u_0(2));

  Eigen::Vector4d d_01 = fetzer_d(ai, bi, aj, bj, 1, 0);
  Eigen::Vector4d d_02 = fetzer_d(ai, bi, aj, bj, 0, 2);
  Eigen::Vector4d d_12 = fetzer_d(ai, bi, aj, bj, 2, 1);

  std::array<Eigen::Vector4d, 3> ds;
  ds[0] = d_01;
  ds[1] = d_02;
  ds[2] = d_12;

  return ds;
}

class FetzerFocalLengthCost {
 public:
  FetzerFocalLengthCost(const Eigen::Matrix3d& i1_F_i0,
                        const Eigen::Vector2d& principal_point0,
                        const Eigen::Vector2d& principal_point1) {
    Eigen::Matrix3d K0 = Eigen::Matrix3d::Identity(3, 3);
    K0(0, 2) = principal_point0(0);
    K0(1, 2) = principal_point0(1);

    Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity(3, 3);
    K1(0, 2) = principal_point1(0);
    K1(1, 2) = principal_point1(1);

    const Eigen::Matrix3d i1_G_i0 = K1.transpose() * i1_F_i0 * K0;

    const std::array<Eigen::Vector4d, 3> ds = fetzer_ds(i1_G_i0);

    d_01 = ds[0];
    d_02 = ds[1];
    d_12 = ds[2];
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d i1_F_i0,
                                     const Eigen::Vector2d& principal_point0,
                                     const Eigen::Vector2d& principal_point1) {
    return (new ceres::AutoDiffCostFunction<FetzerFocalLengthCost, 2, 1, 1>(
        new FetzerFocalLengthCost(
            i1_F_i0, principal_point0, principal_point1)));
  }

  template <typename T>
  bool operator()(const T* const fi_, const T* const fj_, T* residuals) const {
    const Eigen::Vector<T, 4> d_01_ = d_01.cast<T>();
    const Eigen::Vector<T, 4> d_12_ = d_12.cast<T>();

    const T fi = fi_[0];
    const T fj = fj_[0];

    T di = (fj * fj * d_01_(0) + d_01_(1));
    T dj = (fi * fi * d_12_(0) + d_12_(2));
    di = di == T(0) ? T(1e-6) : di;
    dj = dj == T(0) ? T(1e-6) : dj;

    const T K0_01 = -(fj * fj * d_01_(2) + d_01_(3)) / di;
    const T K1_12 = -(fi * fi * d_12_(1) + d_12_(3)) / dj;

    residuals[0] = (fi * fi - K0_01) / (fi * fi);
    residuals[1] = (fj * fj - K1_12) / (fj * fj);

    return true;
  }

 private:
  Eigen::Vector4d d_01;
  Eigen::Vector4d d_02;
  Eigen::Vector4d d_12;
};

// Calibration error for the image pairs sharing the camera
class FetzerFocalLengthSameCameraCost {
 public:
  FetzerFocalLengthSameCameraCost(const Eigen::Matrix3d& i1_F_i0,
                                  const Eigen::Vector2d& principal_point) {
    Eigen::Matrix3d K0 = Eigen::Matrix3d::Identity(3, 3);
    K0(0, 2) = principal_point(0);
    K0(1, 2) = principal_point(1);

    Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity(3, 3);
    K1(0, 2) = principal_point(0);
    K1(1, 2) = principal_point(1);

    const Eigen::Matrix3d i1_G_i0 = K1.transpose() * i1_F_i0 * K0;

    const std::array<Eigen::Vector4d, 3> ds = fetzer_ds(i1_G_i0);

    d_01 = ds[0];
    d_02 = ds[1];
    d_12 = ds[2];
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d i1_F_i0,
                                     const Eigen::Vector2d& principal_point) {
    return (
        new ceres::AutoDiffCostFunction<FetzerFocalLengthSameCameraCost, 2, 1>(
            new FetzerFocalLengthSameCameraCost(i1_F_i0, principal_point)));
  }

  template <typename T>
  bool operator()(const T* const fi_, T* residuals) const {
    const Eigen::Vector<T, 4> d_01_ = d_01.cast<T>();
    const Eigen::Vector<T, 4> d_12_ = d_12.cast<T>();

    const T fi = fi_[0];
    const T fj = fi_[0];

    T di = (fj * fj * d_01_(0) + d_01_(1));
    T dj = (fi * fi * d_12_(0) + d_12_(2));
    di = di == T(0) ? T(1e-6) : di;
    dj = dj == T(0) ? T(1e-6) : dj;

    const T K0_01 = -(fj * fj * d_01_(2) + d_01_(3)) / di;
    const T K1_12 = -(fi * fi * d_12_(1) + d_12_(3)) / dj;

    residuals[0] = (fi * fi - K0_01) / (fi * fi);
    residuals[1] = (fj * fj - K1_12) / (fj * fj);

    return true;
  }

 private:
  Eigen::Vector4d d_01;
  Eigen::Vector4d d_02;
  Eigen::Vector4d d_12;
};

// ----------------------------------------
// GravError
// ----------------------------------------
struct GravError {
  GravError(const Eigen::Vector3d& grav_obs) : grav_obs_(grav_obs) {}

  template <typename T>
  bool operator()(const T* const gvec, T* residuals) const {
    Eigen::Matrix<T, 3, 1> grav_est;
    grav_est << gvec[0], gvec[1], gvec[2];

    for (int i = 0; i < 3; i++) {
      residuals[i] = grav_est[i] - grav_obs_[i];
    }

    return true;
  }

  // Factory function
  static ceres::CostFunction* CreateCost(const Eigen::Vector3d& grav_obs) {
    return (new ceres::AutoDiffCostFunction<GravError, 3, 3>(
        new GravError(grav_obs)));
  }

 private:
  const Eigen::Vector3d& grav_obs_;
};

}  // namespace glomap
