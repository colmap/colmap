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

#include <cmath>
#include <limits>

namespace colmap {
namespace {

// Pixel reprojection error is ill-defined for equirectangular (e.g.
// EQUIRECTANGULAR) cameras: the azimuth is discontinuous at the ±π seam and its
// pixel scale diverges towards the poles (azimuth pixels grow as
// 1/cos(elevation)). Instead measure the angular error between the observed
// bearing and the 3D point and convert it to an equivalent pixel error at the
// equator, consistent with the model's pixel<->angle scale used by
// CamFromImgThreshold. This makes the (squared) error continuous across the
// seam and uniform over the sphere.
double SquaredSphericalReprojectionError(const double angular_error,
                                         const Camera& camera) {
  const double pixels_per_radian =
      static_cast<double>(camera.width) / (2.0 * EIGEN_PI);
  const double pixel_error = angular_error * pixels_per_radian;
  return pixel_error * pixel_error;
}

}  // namespace

double CalculateSquaredReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Rigid3d& cam_from_world,
                                         const Camera& camera) {
  if (camera.IsSpherical()) {
    return SquaredSphericalReprojectionError(
        CalculateAngularReprojectionError(
            point2D, point3D, cam_from_world, camera),
        camera);
  }
  const std::optional<Eigen::Vector2d> proj_point2D =
      camera.ImgFromCam(cam_from_world * point3D);
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
  if (camera.IsSpherical()) {
    return SquaredSphericalReprojectionError(
        CalculateAngularReprojectionError(
            point2D, point3D, cam_from_world, camera),
        camera);
  }
  const std::optional<Eigen::Vector2d> proj_point2D =
      camera.ImgFromCam(cam_from_world * point3D.homogeneous());
  if (!proj_point2D) {
    return std::numeric_limits<double>::max();
  }
  return (*proj_point2D - point2D).squaredNorm();
}

double CalculateAngularReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Rigid3d& cam_from_world,
                                         const Camera& camera) {
  // Use the 3D bearing (full sphere) rather than the 2D CamFromImg, which
  // cannot represent back-hemisphere rays of omnidirectional (e.g.
  // EQUIRECTANGULAR) cameras. Identical to the legacy path for perspective
  // cameras.
  const std::optional<Eigen::Vector3d> cam_ray = camera.CamRayFromImg(point2D);
  if (!cam_ray) {
    return EIGEN_PI;
  }
  return CalculateAngularReprojectionError(*cam_ray, point3D, cam_from_world);
}

double CalculateAngularReprojectionError(
    const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const Camera& camera) {
  const std::optional<Eigen::Vector3d> cam_ray = camera.CamRayFromImg(point2D);
  if (!cam_ray) {
    return EIGEN_PI;
  }
  return CalculateAngularReprojectionError(*cam_ray, point3D, cam_from_world);
}

double CalculateAngularReprojectionError(const Eigen::Vector3d& cam_ray,
                                         const Eigen::Vector3d& point3D,
                                         const Rigid3d& cam_from_world) {
  const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D;
  const double cos_angle = cam_ray.transpose() * point3D_in_cam.normalized();
  return std::acos(std::clamp(cos_angle, -1.0, 1.0));
}

double CalculateAngularReprojectionError(
    const Eigen::Vector3d& cam_ray,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world) {
  const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D.homogeneous();
  const double cos_angle = cam_ray.transpose() * point3D_in_cam.normalized();
  return std::acos(std::clamp(cos_angle, -1.0, 1.0));
}

bool HasPointPositiveDepth(const Eigen::Matrix3x4d& cam_from_world,
                           const Eigen::Vector3d& point3D) {
  return cam_from_world.row(2).dot(point3D.homogeneous()) >=
         std::numeric_limits<double>::epsilon();
}

}  // namespace colmap
