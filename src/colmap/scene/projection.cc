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

#include "colmap/scene/projection.h"

#include "colmap/geometry/pose.h"
#include "colmap/math/matrix.h"

namespace colmap {

double CalculateSquaredReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Rigid3d& cam_from_world,
                                         const Camera& camera) {
  const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D;
  const std::optional<Eigen::Vector2d> proj_point2D =
      camera.ImgFromCam(point3D_in_cam);
  if (!proj_point2D) {
    return std::numeric_limits<double>::max();
  }
  return (*proj_point2D - point2D).squaredNorm();
}

double CalculateSquaredReprojectionError(
    const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const Camera& camera) {
  const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D.homogeneous();
  const std::optional<Eigen::Vector2d> proj_point2D =
      camera.ImgFromCam(point3D_in_cam);
  if (!proj_point2D) {
    return std::numeric_limits<double>::max();
  }
  return (*proj_point2D - point2D).squaredNorm();
}

double CalculateAngularError(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Rigid3d& cam_from_world,
                             const Camera& camera) {
  const std::optional<Eigen::Vector2d> cam_point = camera.CamFromImg(point2D);
  if (!cam_point) {
    return EIGEN_PI;
  }
  return CalculateNormalizedAngularError(*cam_point, point3D, cam_from_world);
}

double CalculateAngularError(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Matrix3x4d& cam_from_world,
                             const Camera& camera) {
  const std::optional<Eigen::Vector2d> cam_point = camera.CamFromImg(point2D);
  if (!cam_point) {
    return EIGEN_PI;
  }
  return CalculateNormalizedAngularError(*cam_point, point3D, cam_from_world);
}

double CalculateNormalizedAngularError(const Eigen::Vector2d& cam_point,
                                       const Eigen::Vector3d& point3D,
                                       const Rigid3d& cam_from_world) {
  const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D;
  return std::acos(cam_point.homogeneous().normalized().transpose() *
                   point3D_in_cam.normalized());
}

double CalculateNormalizedAngularError(
    const Eigen::Vector2d& cam_point,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world) {
  const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D.homogeneous();
  return std::acos(cam_point.homogeneous().normalized().transpose() *
                   point3D_in_cam.normalized());
}

bool HasPointPositiveDepth(const Eigen::Matrix3x4d& cam_from_world,
                           const Eigen::Vector3d& point3D) {
  return cam_from_world.row(2).dot(point3D.homogeneous()) >=
         std::numeric_limits<double>::epsilon();
}

}  // namespace colmap
