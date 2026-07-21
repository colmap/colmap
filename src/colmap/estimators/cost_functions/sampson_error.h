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
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/pose.h"
#include "colmap/util/logging.h"

#include <cmath>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

// Builds the essential matrix E = [t]_x R from a relative pose given in the
// Rigid3d parameter layout [qx, qy, qz, qw, tx, ty, tz]. Templated on the
// scalar so it works under Ceres autodiff.
template <typename T>
Eigen::Matrix<T, 3, 3> EssentialMatrixFromPoseParams(
    const T* const cam2_from_cam1) {
  const Eigen::Matrix<T, 3, 3> R =
      EigenQuaternionMap<T>(cam2_from_cam1).toRotationMatrix();
  // Matrix representation of the cross product t x R.
  Eigen::Matrix<T, 3, 3> t_x;
  t_x << T(0), -cam2_from_cam1[6], cam2_from_cam1[5], cam2_from_cam1[6], T(0),
      -cam2_from_cam1[4], -cam2_from_cam1[5], cam2_from_cam1[4], T(0);
  return t_x * R;
}

// Signed Sampson error under an essential/fundamental matrix. point1/point2 are
// points on the image plane (x, y, 1), not unit bearings. For rays use
// TangentSampsonError. Returns 0 when the denominator vanishes.
template <typename T>
T SampsonError(const Eigen::Matrix<T, 3, 3>& E,
               const Eigen::Matrix<T, 3, 1>& point1,
               const Eigen::Matrix<T, 3, 1>& point2) {
  const Eigen::Matrix<T, 3, 1> epipolar_line1 = E * point1;
  const T num = point2.dot(epipolar_line1);
  const Eigen::Matrix<T, 4, 1> denom(point2.dot(E.col(0)),
                                     point2.dot(E.col(1)),
                                     epipolar_line1.x(),
                                     epipolar_line1.y());
  const T denom_norm = denom.norm();
  if (denom_norm == static_cast<T>(0)) {
    return static_cast<T>(0);
  }
  return num / denom_norm;
}

// Signed tangent Sampson error of one correspondence under E, in pixels, using
// the unprojection Jacobians J_ray1 = d(ray1)/d(pixel1) and J_ray2. Returns 0
// when the denominator vanishes. See ComputeSquaredTangentSampsonError.
template <typename T>
T TangentSampsonError(const Eigen::Matrix<T, 3, 3>& E,
                      const Eigen::Matrix<T, 3, 1>& cam_ray1,
                      const Eigen::Matrix<T, 3, 2>& J_ray1,
                      const Eigen::Matrix<T, 3, 1>& cam_ray2,
                      const Eigen::Matrix<T, 3, 2>& J_ray2) {
  const Eigen::Matrix<T, 3, 1> Eray1 = E * cam_ray1;
  const Eigen::Matrix<T, 3, 1> Etray2 = E.transpose() * cam_ray2;
  const T num = cam_ray2.dot(Eray1);
  Eigen::Matrix<T, 4, 1> denom;
  denom << J_ray1.transpose() * Etray2, J_ray2.transpose() * Eray1;
  const T denom_norm = denom.norm();
  if (denom_norm == static_cast<T>(0)) {
    return static_cast<T>(0);
  }
  return num / denom_norm;
}

// Refines a relative pose by the Sampson error of image-plane point (x, y, 1)
// correspondences. See SampsonError. The pose is [qx, qy, qz, qw, tx, ty, tz]
// with the translation on the unit sphere, so it needs a SphereManifold on
// tvec. For calibrated rays with unprojection Jacobians use the
// TangentSampsonErrorCostFunctor, which is pixel-accurate for any central
// model.
class SampsonErrorCostFunctor
    : public AutoDiffCostFunctor<SampsonErrorCostFunctor, 1, 7> {
 public:
  SampsonErrorCostFunctor(const Eigen::Vector3d& point1,
                          const Eigen::Vector3d& point2)
      : point1_(point1), point2_(point2) {
    // Enforce the (x, y, 1) contract. Unit bearings (z != 1) would be silently
    // rescaled, since the Sampson error is not invariant to the point's scale.
    THROW_CHECK_LT(std::abs(point1.z() - 1.0), 1e-6);
    THROW_CHECK_LT(std::abs(point2.z() - 1.0), 1e-6);
  }

  template <typename T>
  bool operator()(const T* const cam2_from_cam1, T* residuals) const {
    const Eigen::Matrix<T, 3, 3> E =
        EssentialMatrixFromPoseParams(cam2_from_cam1);
    residuals[0] = SampsonError<T>(E, point1_.cast<T>(), point2_.cast<T>());
    return true;
  }

 private:
  const Eigen::Vector3d point1_;
  const Eigen::Vector3d point2_;
};

// Refines a relative pose by the pixel-unit tangent Sampson error of calibrated
// ray correspondences with unprojection Jacobians. See TangentSampsonError.
// Pose layout matches SampsonErrorCostFunctor. Pixel-accurate for any central
// model.
class TangentSampsonErrorCostFunctor
    : public AutoDiffCostFunctor<TangentSampsonErrorCostFunctor, 1, 7> {
 public:
  TangentSampsonErrorCostFunctor(const CamRayWithJac& cam_ray1_with_jac,
                                 const CamRayWithJac& cam_ray2_with_jac)
      : cam_ray1_with_jac_(cam_ray1_with_jac),
        cam_ray2_with_jac_(cam_ray2_with_jac) {}

  template <typename T>
  bool operator()(const T* const cam2_from_cam1, T* residuals) const {
    const Eigen::Matrix<T, 3, 3> E =
        EssentialMatrixFromPoseParams(cam2_from_cam1);
    residuals[0] =
        TangentSampsonError<T>(E,
                               cam_ray1_with_jac_.ray.cast<T>(),
                               cam_ray1_with_jac_.jacobian.cast<T>(),
                               cam_ray2_with_jac_.ray.cast<T>(),
                               cam_ray2_with_jac_.jacobian.cast<T>());
    return true;
  }

 private:
  const CamRayWithJac cam_ray1_with_jac_;
  const CamRayWithJac cam_ray2_with_jac_;
};

}  // namespace colmap
