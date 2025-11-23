#pragma once

#include <Eigen/Core>

namespace glomap {

// Get the aligment rotation matrix by QR decomposition
// The second col of output is gravity direction
Eigen::Matrix3d GetAlignRot(const Eigen::Vector3d& gravity);

// Get the rotation angle for an upright rotation matrix
double RotUpToAngle(const Eigen::Matrix3d& R_up);

// Get the upright rotation matrix from a rotation angle
Eigen::Matrix3d AngleToRotUp(double angle);

// Estimate the average gravity direction from a set of gravity directions
Eigen::Vector3d AverageGravity(const std::vector<Eigen::Vector3d>& gravities);

double CalcAngle(const Eigen::Vector3d& gravity1,
                 const Eigen::Vector3d& gravity2);
}  // namespace glomap
