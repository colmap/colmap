// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_BASE_POSE_H_
#define COLMAP_SRC_BASE_POSE_H_

#include <vector>

#include <Eigen/Core>

#include "util/types.h"

namespace colmap {

// Convert 3D rotation matrix to Euler angles.
//
// The convention `R = Rx * Ry * Rz` is used,
// using a right-handed coordinate system.
//
// @param R              3x3 rotation matrix.
// @param rx, ry, rz     Euler angles in radians.
void RotationMatrixToEulerAngles(const Eigen::Matrix3d& R, double* rx,
                                 double* ry, double* rz);

// Convert Euler angles to 3D rotation matrix.
//
// The convention `R = Rz * Ry * Rx` is used,
// using a right-handed coordinate system.
//
// @param rx, ry, rz     Euler angles in radians.
//
// @return               3x3 rotation matrix.
Eigen::Matrix3d EulerAnglesToRotationMatrix(const double rx, const double ry,
                                            const double rz);

// Convert 3D rotation matrix to Quaternion representation.
//
// @param rot_mat        3x3 rotation matrix.
//
// @return               Unit Quaternion rotation coefficients (w, x, y, z).
Eigen::Vector4d RotationMatrixToQuaternion(const Eigen::Matrix3d& rot_mat);

// Convert Quaternion representation to 3D rotation matrix.
//
// @param qvec           Unit Quaternion rotation coefficients (w, x, y, z).
//
// @return               3x3 rotation matrix.
Eigen::Matrix3d QuaternionToRotationMatrix(const Eigen::Vector4d& qvec);

// Normalize Quaternion vector.
//
// @param qvec          Quaternion rotation coefficients (w, x, y, z).
//
// @return              Unit Quaternion rotation coefficients (w, x, y, z).
Eigen::Vector4d NormalizeQuaternion(const Eigen::Vector4d& qvec);

// Invert Quaternion vector to return Quaternion of inverse rotation.
//
// @param qvec          Quaternion rotation coefficients (w, x, y, z).
//
// @return              Inverse Quaternion rotation coefficients (w, x, y, z).
Eigen::Vector4d InvertQuaternion(const Eigen::Vector4d& qvec);

// Concatenate Quaternion rotations such that the rotation of `qvec1` is applied
// befor the rotation of `qvec2`.
//
// @param qvec1         Quaternion rotation coefficients (w, x, y, z).
// @param qvec2         Quaternion rotation coefficients (w, x, y, z).
//
// @return              Concatenated Quaternion coefficients (w, x, y, z).
Eigen::Vector4d ConcatenateQuaternions(const Eigen::Vector4d& qvec1,
                                       const Eigen::Vector4d& qvec2);

// Transform point by quaternion rotation.
//
// @param qvec          Quaternion rotation coefficients (w, x, y, z).
// @param point         Point to rotate.
//
// @return              Rotated point.
Eigen::Vector3d QuaternionRotatePoint(const Eigen::Vector4d& qvec,
                                      const Eigen::Vector3d& point);

// Extract camera projection center from projection matrix, i.e. the projection
// center in world coordinates `-R^T t`.
//
// @param                3x4 projection matrix.
//
// @return               3x1 camera projection center.
Eigen::Vector3d ProjectionCenterFromMatrix(
    const Eigen::Matrix3x4d& proj_matrix);

// Extract camera projection center from projection parameters.
//
// @param qvec           Unit Quaternion rotation coefficients (w, x, y, z).
// @param tvec           3x1 translation vector.
//
// @return               3x1 camera projection center.
Eigen::Vector3d ProjectionCenterFromParameters(const Eigen::Vector4d& qvec,
                                               const Eigen::Vector3d& tvec);

// Linearly interpolate camera pose.
//
// @param qvec1, tvec1      Camera pose at t0 = 0.
// @param qvec2, tvec2      Camera pose at t1 = 1.
// @param t                 Interpolation time.
// @param qveci, tveci      Camera pose at time t.
void InterpolatePose(const Eigen::Vector4d& qvec1, const Eigen::Vector3d& tvec1,
                     const Eigen::Vector4d& qvec2, const Eigen::Vector3d& tvec2,
                     const double t, Eigen::Vector4d* qveci,
                     Eigen::Vector3d* tveci);

// Calculate baseline vector from first to second pose.
//
// The given rotation and orientation is expected as the
// world to camera transformation.
//
// @param qvec1           Unit Quaternion rotation coefficients (w, x, y, z).
// @param tvec1           3x1 translation vector.
// @param qvec2           Unit Quaternion rotation coefficients (w, x, y, z).
// @param tvec2           3x1 translation vector.
//
// @return                Baseline vector from 1 to 2.
Eigen::Vector3d CalculateBaseline(const Eigen::Vector4d& qvec1,
                                  const Eigen::Vector3d& tvec1,
                                  const Eigen::Vector4d& qvec2,
                                  const Eigen::Vector3d& tvec2);

// Perform cheirality constraint test, i.e., determine which of the triangulated
// correspondences lie in front of of both cameras. The first camera has the
// projection matrix P1 = [I | 0] and the second camera has the projection
// matrix P2 = [R | t].
//
// @param R            3x3 rotation matrix of second projection matrix.
// @param t            3x1 translation vector of second projection matrix.
// @param points1      First set of corresponding points.
// @param points2      Second set of corresponding points.
// @param points3D     Points that lie in front of both cameras.
bool CheckCheirality(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                     const std::vector<Eigen::Vector3d>& points1,
                     const std::vector<Eigen::Vector3d>& points2,
                     std::vector<Eigen::Vector3d>* points3D);
}  // namespace colmap

#endif  // COLMAP_SRC_BASE_POSE_H_
