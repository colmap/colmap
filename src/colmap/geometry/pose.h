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
#include "colmap/geometry/sim3.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>

namespace colmap {

// Average unit vectors by finding the principal component of the outer product
// sum matrix. Uses SVD to find the direction with maximum variance.
// Supports optional weights (uniform weights if empty).
// Result is sign-corrected to align with the majority of input vectors.
//
// @param vectors        Matrix where each column is a unit vector.
// @param weights        Non-negative weights (uniform if empty).
//
// @return               The average unit vector.
Eigen::VectorXd AverageUnitVectors(const Eigen::MatrixXd& vectors,
                                   const std::vector<double>& weights = {});

// Convenience function to average 3D direction vectors.
//
// @param directions     The 3D direction vectors to be averaged.
// @param weights        Non-negative weights (uniform if empty).
//
// @return               The average direction vector.
Eigen::Vector3d AverageDirections(
    const std::vector<Eigen::Vector3d>& directions,
    const std::vector<double>& weights = {});

// Compute the closes rotation matrix with the closest Frobenius norm by setting
// the singular values of the given matrix to 1.
Eigen::Matrix3d ComputeClosestRotationMatrix(const Eigen::Matrix3d& matrix);

// Decompose projection matrix into intrinsic camera matrix, rotation matrix and
// translation vector. Returns false if decomposition fails.
bool DecomposeProjectionMatrix(const Eigen::Matrix3x4d& proj_matrix,
                               Eigen::Matrix3d* K,
                               Eigen::Matrix3d* R,
                               Eigen::Vector3d* T);

// Convert rotation matrix to/from angle axis.
Eigen::Vector3d RotationMatrixToAngleAxis(const Eigen::Matrix3d& R);
Eigen::Matrix3d AngleAxisToRotationMatrix(const Eigen::Vector3d& w);

// Convert 3D rotation matrix to Euler angles.
//
// The convention `R = Rx * Ry * Rz` is used,
// using a right-handed coordinate system.
//
// @param R              3x3 rotation matrix.
// @param rx, ry, rz     Euler angles in radians.
void RotationMatrixToEulerAngles(const Eigen::Matrix3d& R,
                                 double* rx,
                                 double* ry,
                                 double* rz);

// Convert Euler angles to 3D rotation matrix.
//
// The convention `R = Rz * Ry * Rx` is used,
// using a right-handed coordinate system.
//
// @param rx, ry, rz     Euler angles in radians.
//
// @return               3x3 rotation matrix.
Eigen::Matrix3d EulerAnglesToRotationMatrix(double rx, double ry, double rz);

// Compute the weighted average of multiple Quaternions according to:
//
//    Markley, F. Landis, et al. "Averaging quaternions."
//    Journal of Guidance, Control, and Dynamics 30.4 (2007): 1193-1197.
//
// @param quats         The Quaternions to be averaged.
// @param weights       Non-negative weights.
//
// @return              The average Quaternion.
Eigen::Quaterniond AverageQuaternions(
    const std::vector<Eigen::Quaterniond>& quats,
    const std::vector<double>& weights);

// Linearly interpolate camera pose.
Rigid3d InterpolateCameraPoses(const Rigid3d& cam1_from_world,
                               const Rigid3d& cam2_from_world,
                               double t);

// Perform cheirality constraint test, i.e., determine which of the triangulated
// correspondences lie in front of both cameras.
//
// @param cam2_from_cam1  Relative camera transformation.
// @param cam_rays1       First set of corresponding rays.
// @param cam_rays2       Second set of corresponding rays.
// @param points3D        Points that lie in front of both cameras.
bool CheckCheirality(const Rigid3d& cam2_from_cam1,
                     const std::vector<Eigen::Vector3d>& cam_rays1,
                     const std::vector<Eigen::Vector3d>& cam_rays2,
                     std::vector<Eigen::Vector3d>* points3D);

Rigid3d TransformCameraWorld(const Sim3d& new_from_old_world,
                             const Rigid3d& cam_from_world);

// Compute a gravity-aligned rotation matrix from a gravity direction via
// Householder QR. The second column of the output matrix is the gravity
// direction. This rotation transforms from a coordinate frame where gravity
// is aligned with the Y axis to the world frame with the given gravity
// direction.
//
// @param gravity        Normalized gravity direction vector.
//
// @return               3x3 rotation matrix with Y-axis aligned to gravity.
Eigen::Matrix3d GravityAlignedRotation(const Eigen::Vector3d& gravity);

// Extract yaw angle (rotation about Y-axis) from a gravity-aligned rotation
// matrix, i.e., a rotation where gravity is aligned with the Y axis.
//
// @param rotation       3x3 rotation matrix.
//
// @return               Yaw angle in radians.
double YAxisAngleFromRotation(const Eigen::Matrix3d& rotation);

// Construct gravity-aligned rotation matrix from yaw angle, i.e., a rotation
// about the Y-axis (gravity direction).
//
// @param angle          Yaw angle in radians.
//
// @return               3x3 rotation matrix.
Eigen::Matrix3d RotationFromYAxisAngle(double angle);

}  // namespace colmap
