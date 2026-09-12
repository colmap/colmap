// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>

#include <Eigen/Geometry>

namespace colmap {

// Divide a bounding box into equal-sized sub-boxes.
//
// @param bbox    The bounding box to divide.
// @param split   Number of splits along each axis (x, y, z).
//
// @return        Vector of sub-boxes covering the original box.
std::vector<Eigen::AlignedBox3d> ComputeEqualPartsBboxes(
    const Eigen::AlignedBox3d& bbox, const Eigen::Vector3i& split);

}  // namespace colmap
