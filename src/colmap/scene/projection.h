// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// Calculate the reprojection error.
//
// The reprojection error is the Euclidean distance between the observation
// in the image and the projection of the 3D point into the image. If the
// 3D point is behind the camera, then this function returns DBL_MAX.
double CalculateSquaredReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Rigid3d& cam_from_world,
                                         const Camera& camera);
double CalculateSquaredReprojectionError(
    const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const Camera& camera);

// Calculate the angular reprojection error.
//
// The angular error is the angle between the observed viewing ray and the
// actual viewing ray from the camera center to the 3D point.
double CalculateAngularReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Rigid3d& cam_from_world,
                                         const Camera& camera);
double CalculateAngularReprojectionError(
    const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const Camera& camera);

// Calculate angular reprojection error using normalized camera ray.
//
// The angular error is the angle between the observed viewing ray and the
// actual viewing ray from the camera center to the 3D point.
double CalculateAngularReprojectionError(const Eigen::Vector3d& cam_ray,
                                         const Eigen::Vector3d& point3D,
                                         const Rigid3d& cam_from_world);
double CalculateAngularReprojectionError(
    const Eigen::Vector3d& cam_ray,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world);

// Check if 3D point passes cheirality constraint,
// i.e. it lies in front of the camera and not in the image plane.
//
// @param cam_from_world  3x4 projection matrix.
// @param point3D         3D point as 3x1 vector.
//
// @return                True if point lies in front of camera.
bool HasPointPositiveDepth(const Eigen::Matrix3x4d& cam_from_world,
                           const Eigen::Vector3d& point3D);

}  // namespace colmap
