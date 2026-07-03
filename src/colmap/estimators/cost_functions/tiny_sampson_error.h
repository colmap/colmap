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

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// Sampson-error cost functor for fixed-size (colmap::TinySolver) refinement of
// a two-view relative pose.
//
// Like SampsonErrorCostFunctor it minimizes the Sampson error of E = [t]_x R,
// parameterized by the full 7-parameter relative pose
// [qx, qy, qz, qw, tx, ty, tz] (Rigid3d::params layout). Unlike the per-point
// SampsonErrorCostFunctor it evaluates all correspondences in a single
// (dynamically sized) residual and builds E only once. The rotation-plus-sphere
// manifold is applied by the solver (see tiny_manifold.h), not baked into this
// functor, so the ambient pose is parameterized directly.
class TinySampsonErrorCostFunctor {
 public:
  using Scalar = double;
  enum {
    NUM_RESIDUALS = Eigen::Dynamic,
    NUM_PARAMETERS = 7,
  };

  TinySampsonErrorCostFunctor(const std::vector<Eigen::Vector3d>& cam_rays1,
                              const std::vector<Eigen::Vector3d>& cam_rays2)
      : cam_rays1_(cam_rays1), cam_rays2_(cam_rays2) {}

  int NumResiduals() const { return static_cast<int>(cam_rays1_.size()); }

  template <typename T>
  bool operator()(const T* const cam2_from_cam1, T* residuals) const {
    const Eigen::Matrix<T, 3, 3> R =
        EigenQuaternionMap<T>(cam2_from_cam1).toRotationMatrix();

    // Matrix representation of the cross product t x R.
    Eigen::Matrix<T, 3, 3> t_x;
    t_x << T(0), -cam2_from_cam1[6], cam2_from_cam1[5], cam2_from_cam1[6], T(0),
        -cam2_from_cam1[4], -cam2_from_cam1[5], cam2_from_cam1[4], T(0);

    // Essential matrix E = [t]_x R.
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
  const std::vector<Eigen::Vector3d>& cam_rays1_;
  const std::vector<Eigen::Vector3d>& cam_rays2_;
};

}  // namespace colmap
