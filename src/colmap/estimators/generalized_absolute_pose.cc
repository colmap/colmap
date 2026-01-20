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

#include "colmap/estimators/generalized_absolute_pose.h"

#include "colmap/estimators/poselib_utils.h"
#include "colmap/util/logging.h"

#include <PoseLib/solvers/gp3p.h>
#include <PoseLib/solvers/p3p.h>

namespace colmap {

GP3PEstimator::GP3PEstimator(ResidualType residual_type)
    : residual_type_(residual_type) {}

void GP3PEstimator::Estimate(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             std::vector<M_t>* rigs_from_world) {
  THROW_CHECK_EQ(points2D.size(), 3);
  THROW_CHECK_EQ(points3D.size(), 3);
  THROW_CHECK_NOTNULL(rigs_from_world);

  rigs_from_world->clear();

  std::vector<Eigen::Vector3d> rays_in_rig(3);
  std::vector<Eigen::Vector3d> origins_in_rig(3);
  for (int i = 0; i < 3; ++i) {
    // Compute inverse of cam_from_rig: [R|t]^-1 = [R^T | -R^T*t]
    const Eigen::Matrix3d R = points2D[i].cam_from_rig.leftCols<3>();
    const Eigen::Vector3d t = points2D[i].cam_from_rig.col(3);
    const Eigen::Matrix3d R_inv = R.transpose();
    const Eigen::Vector3d t_inv = -R_inv * t;
    rays_in_rig[i] = (R_inv * points2D[i].ray_in_cam).normalized();
    origins_in_rig[i] = t_inv;
  }

  std::vector<poselib::CameraPose> poses;
  if (origins_in_rig[0].isApprox(origins_in_rig[1], 1e-6) &&
      origins_in_rig[0].isApprox(origins_in_rig[2], 1e-6)) {
    // In case of a panoramic camera/rig, fall back to P3P.
    poselib::p3p(rays_in_rig, points3D, &poses);
    for (poselib::CameraPose& pose : poses) {
      pose.t += origins_in_rig[0];
    }
  } else {
    poselib::gp3p(origins_in_rig, rays_in_rig, points3D, &poses);
  }

  rigs_from_world->reserve(poses.size());
  for (const poselib::CameraPose& pose : poses) {
    rigs_from_world->emplace_back(ConvertPoseLibPoseToRigid3d(pose));
  }
}

void GP3PEstimator::Residuals(const std::vector<X_t>& points2D,
                              const std::vector<Y_t>& points3D,
                              const M_t& rig_from_world,
                              std::vector<double>* residuals) const {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  residuals->resize(points2D.size(), 0);

  // Convert rig_from_world to matrix and extract elements once
  const Eigen::Matrix3x4d R = rig_from_world.ToMatrix();
  const double R_00 = R(0, 0), R_01 = R(0, 1), R_02 = R(0, 2), R_03 = R(0, 3);
  const double R_10 = R(1, 0), R_11 = R(1, 1), R_12 = R(1, 2), R_13 = R(1, 3);
  const double R_20 = R(2, 0), R_21 = R(2, 1), R_22 = R(2, 2), R_23 = R(2, 3);

  for (size_t i = 0; i < points2D.size(); ++i) {
    const Eigen::Matrix3x4d& C = points2D[i].cam_from_rig;
    const double X_0 = points3D[i](0);
    const double X_1 = points3D[i](1);
    const double X_2 = points3D[i](2);

    // Project 3D point from world to rig frame
    const double prx_0 = R_00 * X_0 + R_01 * X_1 + R_02 * X_2 + R_03;
    const double prx_1 = R_10 * X_0 + R_11 * X_1 + R_12 * X_2 + R_13;
    const double prx_2 = R_20 * X_0 + R_21 * X_1 + R_22 * X_2 + R_23;

    // Project from rig frame to camera frame
    const double pcx_0 =
        C(0, 0) * prx_0 + C(0, 1) * prx_1 + C(0, 2) * prx_2 + C(0, 3);
    const double pcx_1 =
        C(1, 0) * prx_0 + C(1, 1) * prx_1 + C(1, 2) * prx_2 + C(1, 3);
    const double pcx_2 =
        C(2, 0) * prx_0 + C(2, 1) * prx_1 + C(2, 2) * prx_2 + C(2, 3);

    // Check if 3D point is in front of camera.
    if (pcx_2 > std::numeric_limits<double>::epsilon()) {
      const double x_0 = points2D[i].ray_in_cam(0);
      const double x_1 = points2D[i].ray_in_cam(1);
      const double x_2 = points2D[i].ray_in_cam(2);

      if (residual_type_ == ResidualType::CosineDistance) {
        const double inv_pcx_norm =
            1.0 / std::sqrt(pcx_0 * pcx_0 + pcx_1 * pcx_1 + pcx_2 * pcx_2);
        const double inv_x_norm =
            1.0 / std::sqrt(x_0 * x_0 + x_1 * x_1 + x_2 * x_2);
        const double cosine_dist =
            1 - inv_pcx_norm * inv_x_norm *
                    (pcx_0 * x_0 + pcx_1 * x_1 + pcx_2 * x_2);
        (*residuals)[i] = cosine_dist * cosine_dist;
      } else if (residual_type_ == ResidualType::ReprojectionError) {
        const double inv_pcx_2 = 1.0 / pcx_2;
        const double inv_x_2 = 1.0 / x_2;
        const double dx_0 = x_0 * inv_x_2 - pcx_0 * inv_pcx_2;
        const double dx_1 = x_1 * inv_x_2 - pcx_1 * inv_pcx_2;
        (*residuals)[i] = dx_0 * dx_0 + dx_1 * dx_1;
      } else {
        LOG(FATAL_THROW) << "Invalid residual type";
      }
    } else {
      (*residuals)[i] = std::numeric_limits<double>::max();
    }
  }
}

}  // namespace colmap
