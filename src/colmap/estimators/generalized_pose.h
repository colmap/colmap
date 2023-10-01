// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/estimators/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/camera.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Estimate generalized absolute pose from 2D-3D correspondences.
//
// @param options              RANSAC options.
// @param points2D             Corresponding 2D points.
// @param points3D             Corresponding 3D points.
// @param camera_idxs          Index of the rig camera for each correspondence.
// @param cams_from_rig        Relative pose from rig to each camera frame.
// @param cameras              Cameras for which to estimate pose.
// @param rig_from_world       Estimated rig from world pose.
// @param num_inliers          Number of inliers in RANSAC.
// @param inlier_mask          Inlier mask for 2D-3D correspondences.
//
// @return                     Whether pose is estimated successfully.
bool EstimateGeneralizedAbsolutePose(
    const RANSACOptions& options,
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const std::vector<size_t>& camera_idxs,
    const std::vector<Rigid3d>& cams_from_rig,
    const std::vector<Camera>& cameras,
    Rigid3d* rig_from_world,
    size_t* num_inliers,
    std::vector<char>* inlier_mask);

// Refine generalized absolute pose (optionally focal lengths)
// from 2D-3D correspondences.
//
// @param options              Refinement options.
// @param inlier_mask          Inlier mask for 2D-3D correspondences.
// @param points2D             Corresponding 2D points.
// @param points3D             Corresponding 3D points.
// @param camera_idxs          Index of the rig camera for each correspondence.
// @param cams_from_rig        Relative pose from rig to each camera frame.
// @param rig_from_world       Estimated rig from world pose.
// @param cameras              Cameras for which to estimate pose. Modified
//                             in-place to store the estimated focal lengths.
// @param rig_from_world_cov   Estimated 6x6 covariance matrix of
//                             the rotation (as axis-angle, in tangent space)
//                             and translation terms (optional).
//
// @return                     Whether the solution is usable.
bool RefineGeneralizedAbsolutePose(
    const AbsolutePoseRefinementOptions& options,
    const std::vector<char>& inlier_mask,
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const std::vector<size_t>& camera_idxs,
    const std::vector<Rigid3d>& cams_from_rig,
    Rigid3d* rig_from_world,
    std::vector<Camera>* cameras,
    Eigen::Matrix6d* rig_from_world_cov = nullptr);

}  // namespace colmap
