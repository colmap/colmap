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

#include "colmap/estimators/cost_functions/utils.h"

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

// Cost function for refining two-view geometry based on the Sampson-Error.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be on the unit sphere around the first pose, i.e. the
// pose of the second camera is parameterized by a 3D rotation and a
// 3D translation with unit norm. `tvec` is therefore over-parameterized as is
// and should be down-projected using `SphereManifold`.
class SampsonErrorCostFunctor
    : public AutoDiffCostFunctor<SampsonErrorCostFunctor, 1, 7> {
 public:
  SampsonErrorCostFunctor(const Eigen::Vector3d& cam_ray1,
                          const Eigen::Vector3d& cam_ray2)
      : cam_ray1_(cam_ray1), cam_ray2_(cam_ray2) {}

  template <typename T>
  bool operator()(const T* const cam2_from_cam1, T* residuals) const {
    const Eigen::Matrix<T, 3, 3> R =
        EigenQuaternionMap<T>(cam2_from_cam1).toRotationMatrix();

    // Matrix representation of the cross product t x R.
    Eigen::Matrix<T, 3, 3> t_x;
    t_x << T(0), -cam2_from_cam1[6], cam2_from_cam1[5], cam2_from_cam1[6], T(0),
        -cam2_from_cam1[4], -cam2_from_cam1[5], cam2_from_cam1[4], T(0);

    // Essential matrix.
    const Eigen::Matrix<T, 3, 3> E = t_x * R;

    // Squared sampson error.
    const Eigen::Matrix<T, 3, 1> epipolar_line1 = E * cam_ray1_.cast<T>();
    const Eigen::Matrix<T, 3, 1> cam_ray2 = cam_ray2_.cast<T>();
    const T num = cam_ray2.dot(epipolar_line1);
    const Eigen::Matrix<T, 4, 1> denom(cam_ray2.dot(E.col(0)),
                                       cam_ray2.dot(E.col(1)),
                                       epipolar_line1.x(),
                                       epipolar_line1.y());
    const T denom_norm = denom.norm();
    if (denom_norm == static_cast<T>(0)) {
      residuals[0] = static_cast<T>(0);
    } else {
      residuals[0] = num / denom_norm;
    }

    return true;
  }

 private:
  const Eigen::Vector3d cam_ray1_;
  const Eigen::Vector3d cam_ray2_;
};

}  // namespace colmap
