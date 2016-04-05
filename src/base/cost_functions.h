// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_BASE_COST_FUNCTIONS_H_
#define COLMAP_SRC_BASE_COST_FUNCTIONS_H_

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

// Standard bundle adjustment cost function for variable
// camera pose and calibration and point parameters.
template <typename CameraModel>
class BundleAdjustmentCostFunction {
 public:
  BundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
      : point2D_(point2D) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            BundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 3,
            CameraModel::num_params>(
        new BundleAdjustmentCostFunction(point2D)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    // Rotate and translate.
    T point3D_local[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, point3D_local);
    point3D_local[0] += tvec[0];
    point3D_local[1] += tvec[1];
    point3D_local[2] += tvec[2];

    // Normalize to image plane
    point3D_local[0] /= point3D_local[2];
    point3D_local[1] /= point3D_local[2];

    // Distort and transform to pixel space.
    T x, y;
    CameraModel::WorldToImage(camera_params, point3D_local[0], point3D_local[1],
                              &x, &y);

    // Re-projection error.
    residuals[0] = x - T(point2D_(0));
    residuals[1] = y - T(point2D_(1));

    return true;
  }

 private:
  const Eigen::Vector2d point2D_;
};

// Bundle adjustment cost function for variable
// camera calibration and point parameters, and fixed camera pose.
template <typename CameraModel>
class BundleAdjustmentConstantPoseCostFunction {
 public:
  BundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d& qvec,
                                           const Eigen::Vector3d& tvec,
                                           const Eigen::Vector2d& point2D)
      : qvec_(qvec), tvec_(tvec), point2D_(point2D) {}

  static ceres::CostFunction* Create(const Eigen::Vector4d& qvec,
                                     const Eigen::Vector3d& tvec,
                                     const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            BundleAdjustmentConstantPoseCostFunction<CameraModel>, 2, 3,
            CameraModel::num_params>(
        new BundleAdjustmentConstantPoseCostFunction(qvec, tvec, point2D)));
  }

  template <typename T>
  bool operator()(const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    T qvec[4] = {T(qvec_(0)), T(qvec_(1)), T(qvec_(2)), T(qvec_(3))};

    // Rotate and translate.
    T point3D_local[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, point3D_local);
    point3D_local[0] += T(tvec_(0));
    point3D_local[1] += T(tvec_(1));
    point3D_local[2] += T(tvec_(2));

    // Normalize to image plane.
    point3D_local[0] /= point3D_local[2];
    point3D_local[1] /= point3D_local[2];

    // Distort and transform to pixel space.
    T x, y;
    CameraModel::WorldToImage(camera_params, point3D_local[0], point3D_local[1],
                              &x, &y);

    // Re-projection error.
    residuals[0] = x - T(point2D_(0));
    residuals[1] = y - T(point2D_(1));

    return true;
  }

 private:
  const Eigen::Vector4d qvec_;
  const Eigen::Vector3d tvec_;
  const Eigen::Vector2d point2D_;
};

// Cost function for refining two-view geometry based on the Sampson-Error.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be on the unit sphere around the first pose, i.e. the
// pose of the second camera is parameterized by a 3D rotation and a
// 3D translation with unit norm. `tvec` is therefore over-parameterized as is
// and should be down-projected using `UnitTranslationPlus`.
class RelativePoseCostFunction {
 public:
  RelativePoseCostFunction(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
      : x1_(x1), x2_(x2) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& x1,
                                     const Eigen::Vector2d& x2) {
    return (new ceres::AutoDiffCostFunction<RelativePoseCostFunction, 1, 4, 3>(
        new RelativePoseCostFunction(x1, x2)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  T* residuals) const {
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
    ceres::QuaternionToRotation(qvec, R.data());

    // Matrix representation of the cross product t x R.
    Eigen::Matrix<T, 3, 3> t_x;
    t_x << T(0), -tvec[2], tvec[1], tvec[2], T(0), -tvec[0], -tvec[1], tvec[0],
        T(0);

    // Essential matrix.
    const Eigen::Matrix<T, 3, 3> E = t_x * R;

    // Homogeneous image coordinates.
    const Eigen::Matrix<T, 3, 1> x1_h(T(x1_(0)), T(x1_(1)), T(1));
    const Eigen::Matrix<T, 3, 1> x2_h(T(x2_(0)), T(x2_(1)), T(1));

    // Squared sampson error.
    const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
    const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
    const T x2tEx1 = x2_h.transpose() * Ex1;
    residuals[0] = x2tEx1 * x2tEx1 / (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) +
                                      Etx2(0) * Etx2(0) + Etx2(1) * Etx2(1));

    return true;
  }

 private:
  const Eigen::Vector2d x1_;
  const Eigen::Vector2d x2_;
};

// Plus operation of 2D local parameterization of unit translation in 3D.
struct UnitTranslationPlus {
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    x_plus_delta[0] = x[0] + delta[0];
    x_plus_delta[1] = x[1] + delta[1];
    x_plus_delta[2] = x[2] + delta[2];

    const T squared_norm = x_plus_delta[0] * x_plus_delta[0] +
                           x_plus_delta[1] * x_plus_delta[1] +
                           x_plus_delta[2] * x_plus_delta[2];

    if (squared_norm > T(0)) {
      const T norm = T(1.0) / ceres::sqrt(squared_norm);
      x_plus_delta[0] *= norm;
      x_plus_delta[1] *= norm;
      x_plus_delta[2] *= norm;
    }

    return true;
  }
};

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_COST_FUNCTIONS_H_
