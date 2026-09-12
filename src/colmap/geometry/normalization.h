// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// Computes axis aligned bounding box for coordinates within the given
// percentile range. Computes the centroid as the mean within the box.
std::pair<Eigen::AlignedBox3d, Eigen::Vector3d> ComputeBoundingBoxAndCentroid(
    double min_percentile,
    double max_percentile,
    std::vector<double> coords_x,
    std::vector<double> coords_y,
    std::vector<double> coords_z);

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
// @param points            Image coordinates.
// @param normed_points     Transformed image coordinates.
// @param normed_from_orig  3x3 transformation matrix.
void CenterAndNormalizeImagePoints(const std::vector<Eigen::Vector2d>& points,
                                   std::vector<Eigen::Vector2d>* normed_points,
                                   Eigen::Matrix3d* normed_from_orig);

}  // namespace colmap
