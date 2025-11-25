#pragma once

#include "glomap/scene/types.h"
#include "glomap/types.h"

#include <Eigen/Geometry>

namespace glomap {

// Convert rotation matrix to angle axis
Eigen::Vector3d RotationToAngleAxis(const Eigen::Matrix3d& rot);

// Convert angle axis to rotation matrix
Eigen::Matrix3d AngleAxisToRotation(const Eigen::Vector3d& aa);

}  // namespace glomap
