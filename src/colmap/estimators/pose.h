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
#include "colmap/optim/loransac.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

struct AbsolutePoseEstimationOptions {
  // Whether to estimate the focal length.
  bool estimate_focal_length = false;

  // Options used for P3P RANSAC.
  RANSACOptions ransac_options;

  AbsolutePoseEstimationOptions() {
    ransac_options.max_error = 12.0;
    // Use high confidence to avoid preemptive termination of P3P RANSAC
    // - too early termination may lead to bad registration.
    ransac_options.min_num_trials = 100;
    ransac_options.max_num_trials = 10000;
    ransac_options.confidence = 0.99999;
  }

  void Check() const { ransac_options.Check(); }
};

struct AbsolutePoseRefinementOptions {
  // Convergence criterion.
  double gradient_tolerance = 1.0;

  // Maximum number of solver iterations.
  int max_num_iterations = 100;

  // Scaling factor determines at which residual robustification takes place.
  double loss_function_scale = 1.0;

  // Whether to refine the focal length parameter group.
  bool refine_focal_length = false;

  // Whether to refine the extra parameter group.
  bool refine_extra_params = false;

  // Whether to print final summary.
  bool print_summary = false;

  void Check() const {
    THROW_CHECK_GE(gradient_tolerance, 0.0);
    THROW_CHECK_GE(max_num_iterations, 0);
    THROW_CHECK_GE(loss_function_scale, 0.0);
  }
};

// Estimate absolute pose (optionally focal length) from 2D-3D correspondences.
//
// Focal length estimation is performed using discrete sampling around the
// focal length of the given camera. The focal length that results in the
// maximal number of inliers is assigned to the given camera.
//
// @param options              Absolute pose estimation options.
// @param points2D             Corresponding 2D points.
// @param points3D             Corresponding 3D points.
// @param cam_from_world       Estimated absolute camera pose.
// @param camera               Camera for which to estimate pose. Modified
//                             in-place to store the estimated focal length.
// @param num_inliers          Number of inliers in RANSAC.
// @param inlier_mask          Inlier mask for 2D-3D correspondences.
//
// @return                     Whether pose is estimated successfully.
bool EstimateAbsolutePose(const AbsolutePoseEstimationOptions& options,
                          const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D,
                          Rigid3d* cam_from_world,
                          Camera* camera,
                          size_t* num_inliers,
                          std::vector<char>* inlier_mask);

// Estimate relative from 2D-2D correspondences.
//
// Pose of first camera is assumed to be at the origin without rotation. Pose
// of second camera is given as world-to-image transformation,
// i.e. `x2 = [R | t] * X2`.
//
// @param ransac_options       RANSAC options.
// @param cam_rays1            Corresponding 2D rays.
// @param cam_rays2            Corresponding 2D rays.
// @param cam2_from_cam1       Estimated pose between cameras.
// @param num_inliers          Number of inliers in RANSAC.
// @param inlier_mask          Inlier mask for 2D-2D correspondences.
//
// @return                     Whether pose is estimated successfully.
bool EstimateRelativePose(const RANSACOptions& ransac_options,
                          const std::vector<Eigen::Vector3d>& cam_rays1,
                          const std::vector<Eigen::Vector3d>& cam_rays2,
                          Rigid3d* cam2_from_cam1,
                          size_t* num_inliers,
                          std::vector<char>* inlier_mask);

// Refine absolute pose (optionally focal length) from 2D-3D correspondences.
//
// @param options              Refinement options.
// @param inlier_mask          Inlier mask for 2D-3D correspondences.
// @param points2D             Corresponding 2D points.
// @param points3D             Corresponding 3D points.
// @param cam_from_world       Refined absolute camera pose.
// @param camera               Camera for which to estimate pose. Modified
//                             in-place to store the estimated focal length.
// @param cam_from_world_cov   Estimated 6x6 covariance matrix of
//                             the rotation (as axis-angle, in tangent space)
//                             and translation terms (optional).
//
// @return                     Whether the solution is usable.
bool RefineAbsolutePose(const AbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        Rigid3d* cam_from_world,
                        Camera* camera,
                        Eigen::Matrix6d* cam_from_world_cov = nullptr);

// Refine relative pose of two cameras.
//
// Minimizes the Sampson error between corresponding normalized points using
// a robust cost function, i.e. the corresponding points need not necessarily
// be inliers given a sufficient initial guess for the relative pose.
//
// Assumes that first camera pose has projection matrix P = [I | 0], and
// pose of second camera is given as transformation from world to camera system.
//
// Assumes that the given translation vector is normalized, and refines
// the translation up to an unknown scale (i.e. refined translation vector
// is a unit vector again).
//
// @param options          Solver options.
// @param inlier_mask      Inlier mask for 2D-2D correspondences.
// @param cam_points1      First set of corresponding normalized points.
// @param cam_points2      Second set of corresponding normalized points.
// @param cam_from_world   Refined pose between cameras.
//
// @return                 Flag indicating if solution is usable.
bool RefineRelativePose(const ceres::Solver::Options& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector3d>& cam_rays1,
                        const std::vector<Eigen::Vector3d>& cam_rays2,
                        Rigid3d* cam_from_world);

// Refine essential matrix.
//
// Decomposes the essential matrix into rotation and translation components
// and refines the relative pose using the function `RefineRelativePose`.
//
// @param E                3x3 essential matrix.
// @param cam_rays1        First set of corresponding normalized rays.
// @param cam_rays2        Second set of corresponding normalized rays.
// @param inlier_mask      Inlier mask for corresponding rays.
// @param options          Solver options.
//
// @return                 Flag indicating if solution is usable.
bool RefineEssentialMatrix(const ceres::Solver::Options& options,
                           const std::vector<Eigen::Vector3d>& cam_rays1,
                           const std::vector<Eigen::Vector3d>& cam_rays2,
                           const std::vector<char>& inlier_mask,
                           Eigen::Matrix3d* E);

}  // namespace colmap
