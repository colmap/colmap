
#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace glomap {

// Computes the error between a translation direction and the direction formed
// from two positions such that t_ij - scale * (p_j - p_i) is minimized.
// The positions can either be camera centers or camera centers and 3D points.
struct BATAPairwiseDirectionError {
  explicit BATAPairwiseDirectionError(const Eigen::Vector3d& pos2_from_pos1_dir)
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
        new ceres::AutoDiffCostFunction<BATAPairwiseDirectionError, 3, 3, 3, 1>(
            new BATAPairwiseDirectionError(pos2_from_pos1_dir)));
  }

  const Eigen::Vector3d pos2_from_pos1_dir_;
};

// Computes the error between a translation direction and the direction formed
// from two positions such that t_ij - scale * (c_j - c_i + rig_scale * t_rig)
// is minimized.
struct RigBATAPairwiseDirectionError {
  RigBATAPairwiseDirectionError(const Eigen::Vector3d& cam_from_point3D_dir,
                                const Eigen::Vector3d& cam_from_rig_dir)
      : cam_from_point3D_dir_(cam_from_point3D_dir),
        cam_from_rig_dir_(cam_from_rig_dir) {}

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
                    rig_scale[0] * cam_from_rig_dir_.cast<T>());
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Vector3d& cam_from_rig_dir) {
    return (
        new ceres::
            AutoDiffCostFunction<RigBATAPairwiseDirectionError, 3, 3, 3, 1, 1>(
                new RigBATAPairwiseDirectionError(cam_from_point3D_dir,
                                                  cam_from_rig_dir)));
  }

  const Eigen::Vector3d cam_from_point3D_dir_;
  const Eigen::Vector3d cam_from_rig_dir_;
};

// Computes the error between a translation direction and the direction formed
// from three positions such that v - scale * ((X - r_c_w) - r_R_w^T * c_c_r) is
// minimized.
struct RigUnknownBATAPairwiseDirectionError {
  RigUnknownBATAPairwiseDirectionError(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Quaterniond& rig_from_world_rot)
      : cam_from_point3D_dir_(cam_from_point3D_dir),
        rig_from_world_rot_(rig_from_world_rot) {}

  // The error is given by the position error described above.
  template <typename T>
  bool operator()(const T* point3D,
                  const T* rig_in_world,
                  const T* cam_in_rig,
                  const T* scale,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> cam_from_rig_dir =
        rig_from_world_rot_.cast<T>().inverse() *
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(cam_in_rig);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        cam_from_point3D_dir_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(point3D) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(rig_in_world) -
                    cam_from_rig_dir);
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Quaterniond& rig_from_world_rot) {
    return (
        new ceres::AutoDiffCostFunction<RigUnknownBATAPairwiseDirectionError,
                                        3,
                                        3,
                                        3,
                                        3,
                                        1>(
            new RigUnknownBATAPairwiseDirectionError(cam_from_point3D_dir,
                                                     rig_from_world_rot)));
  }

  const Eigen::Vector3d cam_from_point3D_dir_;
  const Eigen::Quaterniond rig_from_world_rot_;
};

inline Eigen::Vector4d FetzerFocalLengthCostHelper(const Eigen::Vector3d& ai,
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

inline std::array<Eigen::Vector4d, 3> FetzerFocalLengthCostHelper(
    const Eigen::Matrix3d& i1_G_i0) {
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      i1_G_i0, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d s = svd.singularValues();

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
      FetzerFocalLengthCostHelper(ai, bi, aj, bj, 1, 0);
  const Eigen::Vector4d d_02 =
      FetzerFocalLengthCostHelper(ai, bi, aj, bj, 0, 2);
  const Eigen::Vector4d d_12 =
      FetzerFocalLengthCostHelper(ai, bi, aj, bj, 2, 1);
  return {d_01, d_02, d_12};
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

    const std::array<Eigen::Vector4d, 3> ds =
        FetzerFocalLengthCostHelper(i1_G_i0);

    d_01 = ds[0];
    d_02 = ds[1];
    d_12 = ds[2];
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d& i1_F_i0,
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

    const std::array<Eigen::Vector4d, 3> ds =
        FetzerFocalLengthCostHelper(i1_G_i0);

    d_01 = ds[0];
    d_02 = ds[1];
    d_12 = ds[2];
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d& i1_F_i0,
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

struct GravityError {
  explicit GravityError(const Eigen::Vector3d& measured_gravity)
      : measured_gravity_(measured_gravity) {}

  template <typename T>
  bool operator()(const T* const gravity, T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(gravity) -
                    measured_gravity_.cast<T>();

    return true;
  }

  static ceres::CostFunction* CreateCost(
      const Eigen::Vector3d& measured_gravity) {
    return (new ceres::AutoDiffCostFunction<GravityError, 3, 3>(
        new GravityError(measured_gravity)));
  }

 private:
  const Eigen::Vector3d measured_gravity_;
};

}  // namespace glomap
