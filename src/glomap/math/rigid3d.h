#pragma once

#include "colmap/geometry/rigid3.h"

#include "glomap/scene/types.h"
#include "glomap/types.h"

#include <Eigen/Geometry>

namespace glomap {

// Calculate the rotation angle difference between two poses
double CalcAngle(const Rigid3d& pose1, const Rigid3d& pose2);

// Calculate the center difference between two poses
double CalcTrans(const Rigid3d& pose1, const Rigid3d& pose2);

// Calculatet the translation direction difference between two poses
double CalcTransAngle(const Rigid3d& pose1, const Rigid3d& pose2);

// Calculate the rotation angle difference between two rotations
double CalcAngle(const Eigen::Matrix3d& rotation1,
                 const Eigen::Matrix3d& rotation2);

// Convert degree to radian
double DegToRad(double degree);

// Convert radian to degree
double RadToDeg(double radian);

// Convert pose to angle axis
Eigen::Vector3d Rigid3dToAngleAxis(const Rigid3d& pose);

// Convert rotation matrix to angle axis
Eigen::Vector3d RotationToAngleAxis(const Eigen::Matrix3d& rot);

// Convert angle axis to rotation matrix
Eigen::Matrix3d AngleAxisToRotation(const Eigen::Vector3d& aa);

// Calculate the center of the pose
Eigen::Vector3d CenterFromPose(const Rigid3d& pose);

}  // namespace glomap
