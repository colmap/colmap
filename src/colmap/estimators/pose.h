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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_ESTIMATORS_POSE_H_
#define COLMAP_SRC_ESTIMATORS_POSE_H_

#include "colmap/base/camera.h"
#include "colmap/base/camera_models.h"
#include "colmap/optim/loransac.h"
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

  // Number of discrete samples for focal length estimation.
  size_t num_focal_length_samples = 30;

  // Minimum focal length ratio for discrete focal length sampling
  // around focal length of given camera.
  double min_focal_length_ratio = 0.2;

  // Maximum focal length ratio for discrete focal length sampling
  // around focal length of given camera.
  double max_focal_length_ratio = 5;

  // Number of threads for parallel estimation of focal length.
  int num_threads = ThreadPool::kMaxNumThreads;

  // Options used for P3P RANSAC.
  RANSACOptions ransac_options;

  void Check() const {
    CHECK_GT(num_focal_length_samples, 0);
    CHECK_GT(min_focal_length_ratio, 0);
    CHECK_GT(max_focal_length_ratio, 0);
    CHECK_LT(min_focal_length_ratio, max_focal_length_ratio);
    ransac_options.Check();
  }
};

struct AbsolutePoseRefinementOptions {
  // Convergence criterion.
  double gradient_tolerance = 1.0;

  // Maximum number of solver iterations.
  int max_num_iterations = 100;

  // Scaling factor determines at which residual robustification takes place.
  double loss_function_scale = 1.0;

  // Whether to refine the focal length parameter group.
  bool refine_focal_length = true;

  // Whether to refine the extra parameter group.
  bool refine_extra_params = true;

  // Whether to print final summary.
  bool print_summary = true;

  void Check() const {
    CHECK_GE(gradient_tolerance, 0.0);
    CHECK_GE(max_num_iterations, 0);
    CHECK_GE(loss_function_scale, 0.0);
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
// @param qvec                 Estimated rotation component as
//                             unit Quaternion coefficients (w, x, y, z).
// @param tvec                 Estimated translation component.
// @param camera               Camera for which to estimate pose. Modified
//                             in-place to store the estimated focal length.
// @param num_inliers          Number of inliers in RANSAC.
// @param inlier_mask          Inlier mask for 2D-3D correspondences.
//
// @return                     Whether pose is estimated successfully.
bool EstimateAbsolutePose(const AbsolutePoseEstimationOptions& options,
                          const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D,
                          Eigen::Vector4d* qvec,
                          Eigen::Vector3d* tvec,
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
// @param points1              Corresponding 2D points.
// @param points2              Corresponding 2D points.
// @param qvec                 Estimated rotation component as
//                             unit Quaternion coefficients (w, x, y, z).
// @param tvec                 Estimated translation component.
//
// @return                     Number of RANSAC inliers.
size_t EstimateRelativePose(const RANSACOptions& ransac_options,
                            const std::vector<Eigen::Vector2d>& points1,
                            const std::vector<Eigen::Vector2d>& points2,
                            Eigen::Vector4d* qvec,
                            Eigen::Vector3d* tvec);

// Refine absolute pose (optionally focal length) from 2D-3D correspondences.
//
// @param options              Refinement options.
// @param inlier_mask          Inlier mask for 2D-3D correspondences.
// @param points2D             Corresponding 2D points.
// @param points3D             Corresponding 3D points.
// @param qvec                 Estimated rotation component as
//                             unit Quaternion coefficients (w, x, y, z).
// @param tvec                 Estimated translation component.
// @param camera               Camera for which to estimate pose. Modified
//                             in-place to store the estimated focal length.
// @param rot_tvec_covariance  Estimated 6x6 covariance matrix of
//                             the rotation (as axis-angle, in tangent space)
//                             and translation terms (optional).
//
// @return                     Whether the solution is usable.
bool RefineAbsolutePose(const AbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        Eigen::Vector4d* qvec,
                        Eigen::Vector3d* tvec,
                        Camera* camera,
                        Eigen::Matrix6d* rot_tvec_covariance = nullptr);

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
// @param points1          First set of corresponding points.
// @param points2          Second set of corresponding points.
// @param qvec             Unit Quaternion rotation coefficients (w, x, y, z).
// @param tvec             3x1 translation vector.
//
// @return                 Flag indicating if solution is usable.
bool RefineRelativePose(const ceres::Solver::Options& options,
                        const std::vector<Eigen::Vector2d>& points1,
                        const std::vector<Eigen::Vector2d>& points2,
                        Eigen::Vector4d* qvec,
                        Eigen::Vector3d* tvec);

// Refine generalized absolute pose (optionally focal lengths)
// from 2D-3D correspondences.
//
// @param options              Refinement options.
// @param inlier_mask          Inlier mask for 2D-3D correspondences.
// @param points2D             Corresponding 2D points.
// @param points3D             Corresponding 3D points.
// @param camera_idxs          Index of the rig camera for each correspondence.
// @param rig_qvecs            Relative rotations from rig to each camera frame
// @param rig_tvecs            Relative translations from rig
//                             to each camera frame.
// @param qvec                 Estimated rotation component of the rig as
//                             unit Quaternion coefficients (w, x, y, z).
// @param tvec                 Estimated translation component of the rig.
// @param cameras              Cameras for which to estimate pose. Modified
//                             in-place to store the estimated focal lengths.
// @param rot_tvec_covariance  Estimated 6x6 covariance matrix of
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
    const std::vector<Eigen::Vector4d>& rig_qvecs,
    const std::vector<Eigen::Vector3d>& rig_tvecs,
    Eigen::Vector4d* qvec,
    Eigen::Vector3d* tvec,
    std::vector<Camera>* cameras,
    Eigen::Matrix6d* rot_tvec_covariance = nullptr);

}  // namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_POSE_H_
