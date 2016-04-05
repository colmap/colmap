// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_ESTIMATORS_POSE_H_
#define COLMAP_SRC_ESTIMATORS_POSE_H_

#include <vector>

#include <Eigen/Core>

#include <ceres/ceres.h>

#include "base/camera.h"
#include "base/camera_models.h"
#include "optim/ransac.h"
#include "util/logging.h"
#include "util/threading.h"
#include "util/types.h"

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
                          Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                          Camera* camera, size_t* num_inliers,
                          std::vector<bool>* inlier_mask);

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
                            Eigen::Vector4d* qvec, Eigen::Vector3d* tvec);

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
//
// @return                     Whether the solution is usable.
bool RefineAbsolutePose(const AbsolutePoseRefinementOptions& options,
                        const std::vector<bool>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                        Camera* camera);

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
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec);

}  // namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_POSE_H_
