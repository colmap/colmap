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

#include "colmap/estimators/generalized_relative_pose.h"
#include "colmap/estimators/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/camera.h"
#include "colmap/util/eigen_alignment.h"

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

// Estimate generalized relative pose from 2D-2D correspondences.
//
// @param options              RANSAC options.
// @param points2D1            Corresponding 2D points.
// @param points2D2            Corresponding 2D points.
// @param camera_idxs1         Index of the rig camera for each correspondence.
// @param camera_idxs2         Index of the rig camera for each correspondence.
// @param cams_from_rig        Relative pose from rig to each camera frame.
// @param cameras              Cameras for which to estimate pose.
// @param rig2_from_rig1       Estimated rig2 from rig1 pose, if at least one of
//                             the rigs is non-panoramic.
// @param pano2_from_pano1     Estimated rig2 from rig1 pose, if the rigs are
//                             both panoramic.
// @param num_inliers          Number of inliers in RANSAC.
// @param inlier_mask          Inlier mask for 2D-2D correspondences.
//
// @return                     Whether pose is estimated successfully.
bool EstimateGeneralizedRelativePose(
    const RANSACOptions& ransac_options,
    const std::vector<Eigen::Vector2d>& points2D1,
    const std::vector<Eigen::Vector2d>& points2D2,
    const std::vector<size_t>& camera_idxs1,
    const std::vector<size_t>& camera_idxs2,
    const std::vector<Rigid3d>& cams_from_rig,
    const std::vector<Camera>& cameras,
    std::optional<Rigid3d>* rig2_from_rig1,
    std::optional<Rigid3d>* pano2_from_pano1,
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

struct StructureLessAbsolutePoseEstimationOptions {
  // Options used for RANSAC.
  RANSACOptions ransac_options;

  StructureLessAbsolutePoseEstimationOptions() {
    ransac_options.max_error = 12.0;
    // Use high confidence to avoid preemptive termination o RANSAC
    // - too early termination may lead to bad registration.
    ransac_options.min_num_trials = 100;
    ransac_options.max_num_trials = 10000;
    ransac_options.confidence = 0.99999;
  }

  void Check() const { ransac_options.Check(); }
};

// Estimate absolute camera pose using 2D-2D correspondences.
// The 2D-2D correspondences are assumed to be structureless, i.e. the
// 3D points are not known. Based on the following paper:
// "Structure from Motion Using Structure-less Resection", Zheng and Wu, 2015.
bool EstimateStructureLessAbsolutePose(
    const StructureLessAbsolutePoseEstimationOptions& options,
    const std::vector<GRNPObservation>& points_world,
    const std::vector<GRNPObservation>& points_cam,
    const Camera& camera,
    Rigid3d* cam_from_world,
    size_t* num_inliers,
    std::vector<char>* inlier_mask);

}  // namespace colmap
