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

#include "colmap/math/polynomial.h"
#include "colmap/util/logging.h"

#include <array>

#include <PoseLib/solvers/gp3p.h>

namespace colmap {

void GP3PEstimator::Estimate(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             std::vector<M_t>* rigs_from_world) {
  THROW_CHECK_EQ(points2D.size(), 3);
  THROW_CHECK_EQ(points3D.size(), 3);
  THROW_CHECK_NOTNULL(rigs_from_world);

  std::vector<Eigen::Vector3d> rays_in_rig(3);
  std::vector<Eigen::Vector3d> origins_in_world(3);
  for (int i = 0; i < 3; ++i) {
    const Rigid3d rig_from_cam = Inverse(points2D[i].cam_from_rig);
    rays_in_rig[i] =
        (rig_from_cam.rotation * points2D[i].ray_in_cam).normalized();
    origins_in_world[i] = rig_from_cam.translation;
  }

  std::vector<poselib::CameraPose> poses;
  const int num_poses =
      poselib::gp3p(origins_in_world, rays_in_rig, points3D, &poses);

  rigs_from_world->resize(num_poses);
  for (int i = 0; i < num_poses; ++i) {
    (*rigs_from_world)[i] = Rigid3d::FromMatrix(poses[i].Rt());
  }
}

void GP3PEstimator::Residuals(const std::vector<X_t>& points2D,
                              const std::vector<Y_t>& points3D,
                              const M_t& rig_from_world,
                              std::vector<double>* residuals) {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  residuals->resize(points2D.size(), 0);
  for (size_t i = 0; i < points2D.size(); ++i) {
    const Eigen::Vector3d point3D_in_cam =
        points2D[i].cam_from_rig * (rig_from_world * points3D[i]);
    // Check if 3D point is in front of camera.
    if (point3D_in_cam.z() > std::numeric_limits<double>::epsilon()) {
      if (residual_type == ResidualType::CosineDistance) {
        const double cosine_dist =
            1 - point3D_in_cam.normalized().dot(points2D[i].ray_in_cam);
        (*residuals)[i] = cosine_dist * cosine_dist;
      } else if (residual_type == ResidualType::ReprojectionError) {
        (*residuals)[i] = (point3D_in_cam.hnormalized() -
                           points2D[i].ray_in_cam.hnormalized())
                              .squaredNorm();
      } else {
        LOG(FATAL_THROW) << "Invalid residual type";
      }
    } else {
      (*residuals)[i] = std::numeric_limits<double>::max();
    }
  }
}

}  // namespace colmap
