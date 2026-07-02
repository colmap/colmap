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

#include "colmap/geometry/rigid3.h"

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/rotation.h>

namespace colmap {

// The 5-DoF minimal-tangent parameterization of a relative pose about a base
// pose: rotation via the SO(3) boxplus and translation on the unit sphere,
// matching ProductManifold(EigenQuaternion, Sphere<3>). Shared by the cost
// functor (to evaluate residuals) and the refinement harness (to recover the
// refined pose from the solved tangent), so the boxplus lives in one place.
class TangentRelativePose {
 public:
  explicit TangentRelativePose(const Rigid3d& cam2_from_cam1)
      : base_rotation_(
            cam2_from_cam1.rotation().normalized().toRotationMatrix()),
        base_translation_(cam2_from_cam1.translation().normalized()),
        // Any orthonormal basis {b1, b2} of the plane orthogonal to the base
        // translation.
        b1_(base_translation_.unitOrthogonal()),
        b2_(base_translation_.cross(b1_)) {}

  // Boxplus (pose = base pose ⊞ params): applies the tangent increment
  // `params` = [omega(3), delta_sphere(2)] to the base pose, yielding the
  // rotation matrix and unit translation of the resulting pose.
  template <typename T>
  void BoxPlus(const T* const params,
               Eigen::Matrix<T, 3, 3>* R,
               Eigen::Matrix<T, 3, 1>* t) const {
    T dR[9];
    ceres::AngleAxisToRotationMatrix(params, dR);
    *R =
        base_rotation_.cast<T>() * Eigen::Map<const Eigen::Matrix<T, 3, 3>>(dR);
    *t = (base_translation_.cast<T>() + params[3] * b1_.cast<T>() +
          params[4] * b2_.cast<T>())
             .normalized();
  }

 private:
  const Eigen::Matrix3d base_rotation_;
  const Eigen::Vector3d base_translation_;
  const Eigen::Vector3d b1_;
  const Eigen::Vector3d b2_;
};

// Sampson-error cost functor for fixed-size (ceres::TinySolver) refinement of a
// two-view relative pose.
//
// Like SampsonErrorCostFunctor it minimizes the Sampson error of E = [t]_x R,
// but instead of parameterizing the full pose and relying on an external
// manifold, it is parameterized on the 5-DoF minimal tangent (see
// TangentRelativePose), with the boxplus applied inside operator(). This
// makes it usable by TinySolver, which has no manifold, and it evaluates all
// correspondences in a single (dynamically sized) residual.
class TangentSampsonErrorCostFunctor {
 public:
  TangentSampsonErrorCostFunctor(const TangentRelativePose& tangent,
                                 const std::vector<Eigen::Vector3d>& cam_rays1,
                                 const std::vector<Eigen::Vector3d>& cam_rays2)
      : tangent_(tangent), cam_rays1_(cam_rays1), cam_rays2_(cam_rays2) {}

  int NumResiduals() const { return static_cast<int>(cam_rays1_.size()); }

  template <typename T>
  bool operator()(const T* const params, T* residuals) const {
    Eigen::Matrix<T, 3, 3> R;
    Eigen::Matrix<T, 3, 1> t;
    tangent_.BoxPlus(params, &R, &t);

    // Essential matrix E = [t]_x R.
    Eigen::Matrix<T, 3, 3> t_x;
    t_x << T(0), -t(2), t(1), t(2), T(0), -t(0), -t(1), t(0), T(0);
    const Eigen::Matrix<T, 3, 3> E = t_x * R;

    for (size_t i = 0; i < cam_rays1_.size(); ++i) {
      const Eigen::Matrix<T, 3, 1> cam_ray1 = cam_rays1_[i].cast<T>();
      const Eigen::Matrix<T, 3, 1> cam_ray2 = cam_rays2_[i].cast<T>();
      const Eigen::Matrix<T, 3, 1> epipolar_line1 = E * cam_ray1;
      const T num = cam_ray2.dot(epipolar_line1);
      const Eigen::Matrix<T, 4, 1> denom(cam_ray2.dot(E.col(0)),
                                         cam_ray2.dot(E.col(1)),
                                         epipolar_line1.x(),
                                         epipolar_line1.y());
      residuals[i] = num / denom.norm();
    }
    return true;
  }

 private:
  TangentRelativePose tangent_;
  const std::vector<Eigen::Vector3d>& cam_rays1_;
  const std::vector<Eigen::Vector3d>& cam_rays2_;
};

}  // namespace colmap
