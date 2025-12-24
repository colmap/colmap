#pragma once

#include <Eigen/Core>

#include <vector>

namespace glomap {

// Compute rotation matrix from gravity direction via Householder QR.
// The second column of the output matrix is the gravity direction.
Eigen::Matrix3d RotationFromGravity(const Eigen::Vector3d& gravity);

// Extract yaw angle from a gravity-aligned rotation matrix.
double YawFromRotation(const Eigen::Matrix3d& rotation);

// Construct rotation matrix from yaw angle (rotation about Y-axis).
Eigen::Matrix3d RotationFromYaw(double yaw);

// Compute the average direction from normalized gravity direction vectors.
Eigen::Vector3d AverageGravityDirection(
    const std::vector<Eigen::Vector3d>& gravities);

}  // namespace glomap
