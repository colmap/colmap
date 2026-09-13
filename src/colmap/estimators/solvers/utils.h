// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/pose.h"
#include "colmap/util/eigen_alignment.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Solve an Nx9 (N >= 8) epipolar system and enforce rank 2. The minimal
// N == 8 case takes the exact QR nullspace, while N > 8 takes the least-squares
// SVD nullspace.
Eigen::Matrix3d SolveEpipolarConstraintMatrix(
    const Eigen::Matrix<double, Eigen::Dynamic, 9>& A);

// Convert centered image points to normalized rays.
std::vector<Eigen::Vector3d> CalibratedRays(
    const std::vector<Eigen::Vector2d>& points, double focal);

// Extract rays while discarding their measurement Jacobians.
std::vector<Eigen::Vector3d> RaysFromCamRaysWithJac(
    const std::vector<CamRayWithJac>& cam_rays_with_jac);

}  // namespace colmap
