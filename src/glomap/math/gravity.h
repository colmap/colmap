#pragma once

#include <vector>

#include <Eigen/Core>

namespace glomap {

// Compute rotation matrix from gravity direction via Householder QR.
// The second column of the output matrix is the gravity direction.
Eigen::Matrix3d RotationFromGravity(const Eigen::Vector3d& gravity);

// Extract yaw angle (rotation about Y-axis) from a gravity-aligned rotation
// matrix, i.e., a rotation where gravity is aligned with the Y axis.
double YAxisAngleFromRotation(const Eigen::Matrix3d& rotation);

// Construct gravity-aligned rotation matrix from yaw angle, i.e., a rotation
// about the Y-axis (gravity direction).
Eigen::Matrix3d RotationFromYAxisAngle(double angle);

// Compute the average direction from normalized gravity direction vectors.
Eigen::Vector3d AverageGravityDirection(
    const std::vector<Eigen::Vector3d>& gravities);

}  // namespace glomap
