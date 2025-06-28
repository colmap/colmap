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
#include "colmap/math/math.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Triangulate 3D point from corresponding image point observations.
//
// Implementation of the direct linear transform triangulation method in
//   R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision,
//   Cambridge Univ. Press, 2003.
//
// @param cam_from_world1   Pose of the first camera as 3x4 matrix.
// @param cam_from_world2   Pose of the second camera as 3x4 matrix.
// @param cam_point1        Corresponding 2D point in first image.
// @param cam_point2        Corresponding 2D point in second image.
// @param point3D           Triangulated 3D point.
//
// @return                  Whether triangulation was successful.
bool TriangulatePoint(const Eigen::Matrix3x4d& cam_from_world1,
                      const Eigen::Matrix3x4d& cam_from_world2,
                      const Eigen::Vector2d& cam_point1,
                      const Eigen::Vector2d& cam_point2,
                      Eigen::Vector3d* point3D);

// Triangulate 3D mid-point from corresponding camera ray observations.
//
// @param cam2_from_cam1    Relative pose between camera pair.
// @param cam_ray1          Corresponding 2D ray in first camera.
// @param cam_ray2          Corresponding 2D ray in second camera.
// @param point3D           Triangulated 3D point in first camera.
//
// @return                  Whether triangulation was successful.
bool TriangulateMidPoint(const Rigid3d& cam2_from_cam1,
                         const Eigen::Vector3d& cam_ray1,
                         const Eigen::Vector3d& cam_ray2,
                         Eigen::Vector3d* point3D_in_cam1);

// Triangulate point from multiple views minimizing the L2 error.
//
// @param cams_from_world   Projection matrices of multi-view observations.
// @param cam_points        Image observations of multi-view observations.
// @param point3D           Triangulated 3D point.
//
// @return                  Whether triangulation was successful.
bool TriangulateMultiViewPoint(
    const span<const Eigen::Matrix3x4d>& cams_from_world,
    const span<const Eigen::Vector2d>& cam_points,
    Eigen::Vector3d* point3D);

// Triangulate optimal 3D point from corresponding image point observations by
// finding the optimal image observations.
//
// Note that camera poses should be very good in order for this method to yield
// good results. Otherwise just use `TriangulatePoint`.
//
// Implementation of the method described in
//   P. Lindstrom, "Triangulation Made Easy," IEEE Computer Vision and Pattern
//   Recognition 2010, pp. 1554-1561, June 2010.
//
// @param cam_from_world1   Projection matrix of the first image as 3x4 matrix.
// @param cam_from_world2   Projection matrix of the second image as 3x4 matrix.
// @param point1            Corresponding 2D point in first image.
// @param point2            Corresponding 2D point in second image.
// @param point3D           Triangulated 3D point.
//
// @return                  Whether triangulation was successful.
bool TriangulateOptimalPoint(const Eigen::Matrix3x4d& cam_from_world1,
                             const Eigen::Matrix3x4d& cam_from_world2,
                             const Eigen::Vector2d& point1,
                             const Eigen::Vector2d& point2,
                             Eigen::Vector3d* point3D);

// Calculate angle in radians between the two rays of a triangulated point.
double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center1,
                                   const Eigen::Vector3d& proj_center2,
                                   const Eigen::Vector3d& point3D);
std::vector<double> CalculateTriangulationAngles(
    const Eigen::Vector3d& proj_center1,
    const Eigen::Vector3d& proj_center2,
    const std::vector<Eigen::Vector3d>& points3D);

}  // namespace colmap
