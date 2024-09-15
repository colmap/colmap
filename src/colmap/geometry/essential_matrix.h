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

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Decompose an essential matrix into the possible rotations and translations.
//
// The first pose is assumed to be P = [I | 0] and the set of four other
// possible second poses are defined as: {[R1 | t], [R2 | t],
//                                        [R1 | -t], [R2 | -t]}
//
// @param E          3x3 essential matrix.
// @param R1         First possible 3x3 rotation matrix.
// @param R2         Second possible 3x3 rotation matrix.
// @param t          3x1 possible translation vector (also -t possible).
void DecomposeEssentialMatrix(const Eigen::Matrix3d& E,
                              Eigen::Matrix3d* R1,
                              Eigen::Matrix3d* R2,
                              Eigen::Vector3d* t);

// Recover the most probable pose from the given essential matrix.
//
// The pose of the first image is assumed to be P = [I | 0].
//
// @param E               3x3 essential matrix.
// @param points1         First set of corresponding points.
// @param points2         Second set of corresponding points.
// @param inlier_mask     Only points with `true` in the inlier mask are
//                        considered in the cheirality test. Size of the
//                        inlier mask must match the number of points N.
// @param cam2_from_cam1  Relative camera transformation.
// @param points3D        Triangulated 3D points infront of camera.
void PoseFromEssentialMatrix(const Eigen::Matrix3d& E,
                             const std::vector<Eigen::Vector2d>& points1,
                             const std::vector<Eigen::Vector2d>& points2,
                             Rigid3d* cam2_from_cam1,
                             std::vector<Eigen::Vector3d>* points3D);

// Compose essential matrix from relative camera poses.
//
// Assumes that first camera pose has projection matrix P = [I | 0], and
// pose of second camera is given as transformation from world to camera system.
//
// @param R             3x3 rotation matrix.
// @param t             3x1 translation vector.
//
// @return              3x3 essential matrix.
Eigen::Matrix3d EssentialMatrixFromPose(const Rigid3d& cam2_from_cam1);

// Find optimal image points, such that:
//
//     optimal_point1^t * E * optimal_point2 = 0
//
// as described in:
//
//   Lindstrom, P., "Triangulation made easy",
//   Computer Vision and Pattern Recognition (CVPR),
//   2010 IEEE Conference on , vol., no., pp.1554,1561, 13-18 June 2010
//
// @param E                Essential or fundamental matrix.
// @param point1           Corresponding 2D point in first image.
// @param point2           Corresponding 2D point in second image.
// @param optimal_point1   Estimated optimal image point in the first image.
// @param optimal_point2   Estimated optimal image point in the second image.
void FindOptimalImageObservations(const Eigen::Matrix3d& E,
                                  const Eigen::Vector2d& point1,
                                  const Eigen::Vector2d& point2,
                                  Eigen::Vector2d* optimal_point1,
                                  Eigen::Vector2d* optimal_point2);

// Compute the location of the epipole in homogeneous coordinates.
//
// @param E           3x3 essential matrix.
// @param left_image  If true, epipole in left image is computed,
//                    else in right image.
//
// @return            Epipole in homogeneous coordinates.
Eigen::Vector3d EpipoleFromEssentialMatrix(const Eigen::Matrix3d& E,
                                           bool left_image);

// Invert the essential matrix, i.e. if the essential matrix E describes the
// transformation from camera A to B, the inverted essential matrix E' describes
// the transformation from camera B to A.
//
// @param E      3x3 essential matrix.
//
// @return       Inverted essential matrix.
Eigen::Matrix3d InvertEssentialMatrix(const Eigen::Matrix3d& matrix);

// Composes the fundamental matrix from image 1 to 2 from the essential matrix
// and two camera's calibrations.
Eigen::Matrix3d FundamentalFromEssentialMatrix(const Eigen::Matrix3d& K2,
                                               const Eigen::Matrix3d& E,
                                               const Eigen::Matrix3d& K1);

}  // namespace colmap
