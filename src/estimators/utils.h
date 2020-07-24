// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#ifndef COLMAP_SRC_ESTIMATORS_UTILS_H_
#define COLMAP_SRC_ESTIMATORS_UTILS_H_

#include <vector>

#include <Eigen/Core>

#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

// Center and normalize image points.
//
// The points are transformed in a two-step procedure that is expressed
// as a transformation matrix. The matrix of the resulting points is usually
// better conditioned than the matrix of the original points.
//
// Center the image points, such that the new coordinate system has its
// origin at the centroid of the image points.
//
// Normalize the image points, such that the mean distance from the points
// to the coordinate system is sqrt(2).
//
// @param points          Image coordinates.
// @param normed_points   Transformed image coordinates.
// @param matrix          3x3 transformation matrix.
void CenterAndNormalizeImagePoints(const std::vector<Eigen::Vector2d>& points,
                                   std::vector<Eigen::Vector2d>* normed_points,
                                   Eigen::Matrix3d* matrix);

// Calculate the residuals of a set of corresponding points and a given
// fundamental or essential matrix.
//
// Residuals are defined as the squared Sampson error.
//
// @param points1     First set of corresponding points as Nx2 matrix.
// @param points2     Second set of corresponding points as Nx2 matrix.
// @param E           3x3 fundamental or essential matrix.
// @param residuals   Output vector of residuals.
void ComputeSquaredSampsonError(const std::vector<Eigen::Vector2d>& points1,
                                const std::vector<Eigen::Vector2d>& points2,
                                const Eigen::Matrix3d& E,
                                std::vector<double>* residuals);

// Calculate the squared reprojection error given a set of 2D-3D point
// correspondences and a projection matrix. Returns DBL_MAX if a 3D point is
// behind the given camera.
//
// @param points2D      Normalized 2D image points.
// @param points3D      3D world points.
// @param proj_matrix   3x4 projection matrix.
// @param residuals     Output vector of residuals.
void ComputeSquaredReprojectionError(
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const Eigen::Matrix3x4d& proj_matrix, std::vector<double>* residuals);

}  // namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_UTILS_H_
