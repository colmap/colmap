// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_BASE_PROJECTION_H_
#define COLMAP_SRC_BASE_PROJECTION_H_

#include "colmap/base/camera.h"

#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// Compose projection matrix from rotation and translation components.
//
// The projection matrix transforms 3D world to image points.
//
// @param qvec           Unit Quaternion rotation coefficients (w, x, y, z).
// @param tvec           3x1 translation vector.
//
// @return               3x4 projection matrix.
Eigen::Matrix3x4d ComposeProjectionMatrix(const Eigen::Vector4d& qvec,
                                          const Eigen::Vector3d& tvec);

// Compose projection matrix from rotation matrix and translation components).
//
// The projection matrix transforms 3D world to image points.
//
// @param R              3x3 rotation matrix.
// @param t              3x1 translation vector.
//
// @return               3x4 projection matrix.
Eigen::Matrix3x4d ComposeProjectionMatrix(const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& T);

// Invert projection matrix, defined as:
//
//    P = [R | t] with R \in SO(3) and t \in R^3
//
// and the inverse projection matrix as:
//
//    P' = [R^T | -R^T t]
//
// @param proj_matrix    3x4 projection matrix.
//
// @return               3x4 inverse projection matrix.
Eigen::Matrix3x4d InvertProjectionMatrix(const Eigen::Matrix3x4d& proj_matrix);

// Compute the closes rotation matrix with the closest Frobenius norm by setting
// the singular values of the given matrix to 1.
Eigen::Matrix3d ComputeClosestRotationMatrix(const Eigen::Matrix3d& matrix);

// Decompose projection matrix into intrinsic camera matrix, rotation matrix and
// translation vector. Returns false if decomposition fails. This implementation
// is inspired by the OpenCV implementation with some additional checks.
bool DecomposeProjectionMatrix(const Eigen::Matrix3x4d& proj_matrix,
                               Eigen::Matrix3d* K,
                               Eigen::Matrix3d* R,
                               Eigen::Vector3d* T);

// Project 3D point to image.
//
// @param points3D         3D world point as 3x1 vector.
// @param proj_matrix      3x4 projection matrix.
// @param camera           Camera used to project to image plane.
//
// @return                 Projected image point.
Eigen::Vector2d ProjectPointToImage(const Eigen::Vector3d& point3D,
                                    const Eigen::Matrix3x4d& proj_matrix,
                                    const Camera& camera);

// Calculate the reprojection error.
//
// The reprojection error is the Euclidean distance between the observation
// in the image and the projection of the 3D point into the image. If the
// 3D point is behind the camera, then this function returns DBL_MAX.
double CalculateSquaredReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Eigen::Vector4d& qvec,
                                         const Eigen::Vector3d& tvec,
                                         const Camera& camera);
double CalculateSquaredReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Eigen::Matrix3x4d& proj_matrix,
                                         const Camera& camera);

// Calculate the angular error.
//
// The angular error is the angle between the observed viewing ray and the
// actual viewing ray from the camera center to the 3D point.
double CalculateAngularError(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Vector4d& qvec,
                             const Eigen::Vector3d& tvec,
                             const Camera& camera);
double CalculateAngularError(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Matrix3x4d& proj_matrix,
                             const Camera& camera);

// Calculate angulate error using normalized image points.
//
// The angular error is the angle between the observed viewing ray and the
// actual viewing ray from the camera center to the 3D point.
double CalculateNormalizedAngularError(const Eigen::Vector2d& point2D,
                                       const Eigen::Vector3d& point3D,
                                       const Eigen::Vector4d& qvec,
                                       const Eigen::Vector3d& tvec);
double CalculateNormalizedAngularError(const Eigen::Vector2d& point2D,
                                       const Eigen::Vector3d& point3D,
                                       const Eigen::Matrix3x4d& proj_matrix);

// Calculate depth of 3D point with respect to camera.
//
// The depth is defined as the Euclidean distance of a 3D point from the
// camera and is positive if the 3D point is in front and negative if
// behind of the camera.
//
// @param proj_matrix     3x4 projection matrix.
// @param point3D         3D point as 3x1 vector.
//
// @return                Depth of 3D point.
double CalculateDepth(const Eigen::Matrix3x4d& proj_matrix,
                      const Eigen::Vector3d& point3D);

// Check if 3D point passes cheirality constraint,
// i.e. it lies in front of the camera and not in the image plane.
//
// @param proj_matrix     3x4 projection matrix.
// @param point3D         3D point as 3x1 vector.
//
// @return                True if point lies in front of camera.
bool HasPointPositiveDepth(const Eigen::Matrix3x4d& proj_matrix,
                           const Eigen::Vector3d& point3D);

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_PROJECTION_H_
