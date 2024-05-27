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

#include "colmap/estimators/generalized_pose.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/cost_functions.h"
#include "colmap/estimators/generalized_absolute_pose.h"
#include "colmap/estimators/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/matrix.h"
#include "colmap/optim/ransac.h"
#include "colmap/optim/support_measurement.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {
namespace {

double ComputeMaxErrorInCamera(const std::vector<size_t>& camera_idxs,
                               const std::vector<Camera>& cameras,
                               const double max_error_px) {
  THROW_CHECK_GT(max_error_px, 0.0);
  double max_error_cam = 0.;
  for (const auto& camera_idx : camera_idxs) {
    max_error_cam += cameras[camera_idx].CamFromImgThreshold(max_error_px);
  }
  return max_error_cam / camera_idxs.size();
}

bool LowerVector3d(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
  if (v1.x() < v2.x()) {
    return true;
  } else if (v1.x() == v2.x()) {
    if (v1.y() < v2.y()) {
      return true;
    } else if (v1.y() == v2.y()) {
      return v1.z() < v2.z();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

std::vector<size_t> ComputeUniquePointIds(
    const std::vector<Eigen::Vector3d>& points3D) {
  std::vector<size_t> point3D_ids(points3D.size());
  std::iota(point3D_ids.begin(), point3D_ids.end(), 0);
  std::sort(point3D_ids.begin(), point3D_ids.end(), [&](size_t i, size_t j) {
    return LowerVector3d(points3D[i], points3D[j]);
  });

  std::vector<size_t>::iterator unique_it = point3D_ids.begin();
  std::vector<size_t>::iterator current_it = point3D_ids.begin();
  std::vector<size_t> unique_point3D_ids(points3D.size());
  while (current_it != point3D_ids.end()) {
    if (!points3D[*unique_it].isApprox(points3D[*current_it], 1e-5)) {
      unique_it = current_it;
    }
    unique_point3D_ids[*current_it] = unique_it - point3D_ids.begin();
    current_it++;
  }
  return unique_point3D_ids;
}

}  // namespace

bool EstimateGeneralizedAbsolutePose(
    const RANSACOptions& options,
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const std::vector<size_t>& camera_idxs,
    const std::vector<Rigid3d>& cams_from_rig,
    const std::vector<Camera>& cameras,
    Rigid3d* rig_from_world,
    size_t* num_inliers,
    std::vector<char>* inlier_mask) {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_EQ(points2D.size(), camera_idxs.size());
  THROW_CHECK_EQ(cams_from_rig.size(), cameras.size());
  THROW_CHECK_GE(*std::min_element(camera_idxs.begin(), camera_idxs.end()), 0);
  THROW_CHECK_LT(*std::max_element(camera_idxs.begin(), camera_idxs.end()),
                 cameras.size());
  options.Check();
  if (points2D.size() == 0) {
    return false;
  }

  std::vector<GP3PEstimator::X_t> rig_points2D(points2D.size());
  for (size_t i = 0; i < points2D.size(); i++) {
    const size_t camera_idx = camera_idxs[i];
    rig_points2D[i].ray_in_cam =
        cameras[camera_idx].CamFromImg(points2D[i]).homogeneous().normalized();
    rig_points2D[i].cam_from_rig = cams_from_rig[camera_idx];
  }

  // Associate unique ids to each 3D point.
  // Needed for UniqueInlierSupportMeasurer to avoid counting the same
  // 3D point multiple times due to FoV overlap in rig.
  // TODO(sarlinpe): Allow passing unique_point3D_ids as argument.
  const std::vector<size_t> unique_point3D_ids =
      ComputeUniquePointIds(points3D);

  // Average of the errors over the cameras, weighted by the number of
  // correspondences
  RANSACOptions options_copy(options);
  options_copy.max_error =
      ComputeMaxErrorInCamera(camera_idxs, cameras, options.max_error);

  RANSAC<GP3PEstimator, UniqueInlierSupportMeasurer> ransac(options_copy);
  ransac.support_measurer.SetUniqueSampleIds(unique_point3D_ids);
  ransac.estimator.residual_type =
      GP3PEstimator::ResidualType::ReprojectionError;
  const auto report = ransac.Estimate(rig_points2D, points3D);
  if (!report.success) {
    return false;
  }
  *rig_from_world = report.model;
  *num_inliers = report.support.num_unique_inliers;
  *inlier_mask = report.inlier_mask;
  return true;
}

bool RefineGeneralizedAbsolutePose(const AbsolutePoseRefinementOptions& options,
                                   const std::vector<char>& inlier_mask,
                                   const std::vector<Eigen::Vector2d>& points2D,
                                   const std::vector<Eigen::Vector3d>& points3D,
                                   const std::vector<size_t>& camera_idxs,
                                   const std::vector<Rigid3d>& cams_from_rig,
                                   Rigid3d* rig_from_world,
                                   std::vector<Camera>* cameras,
                                   Eigen::Matrix6d* rig_from_world_cov) {
  THROW_CHECK_EQ(points2D.size(), inlier_mask.size());
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_EQ(points2D.size(), camera_idxs.size());
  THROW_CHECK_EQ(cams_from_rig.size(), cameras->size());
  THROW_CHECK_GE(*std::min_element(camera_idxs.begin(), camera_idxs.end()), 0);
  THROW_CHECK_LT(*std::max_element(camera_idxs.begin(), camera_idxs.end()),
                 cameras->size());
  options.Check();

  const auto loss_function =
      std::make_unique<ceres::CauchyLoss>(options.loss_function_scale);

  std::vector<double*> cameras_params_data;
  for (size_t i = 0; i < cameras->size(); i++) {
    cameras_params_data.push_back(cameras->at(i).params.data());
  }
  std::vector<size_t> camera_counts(cameras->size(), 0);
  double* rig_from_world_rotation = rig_from_world->rotation.coeffs().data();
  double* rig_from_world_translation = rig_from_world->translation.data();

  std::vector<Eigen::Vector3d> points3D_copy = points3D;
  std::vector<Rigid3d> cams_from_rig_copy = cams_from_rig;

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  for (size_t i = 0; i < points2D.size(); ++i) {
    // Skip outlier observations
    if (!inlier_mask[i]) {
      continue;
    }
    const size_t camera_idx = camera_idxs[i];
    camera_counts[camera_idx] += 1;

    problem.AddResidualBlock(
        CameraCostFunction<RigReprojErrorCostFunction>(
            cameras->at(camera_idx).model_id, points2D[i]),
        loss_function.get(),
        cams_from_rig_copy[camera_idx].rotation.coeffs().data(),
        cams_from_rig_copy[camera_idx].translation.data(),
        rig_from_world_rotation,
        rig_from_world_translation,
        points3D_copy[i].data(),
        cameras_params_data[camera_idx]);
    problem.SetParameterBlockConstant(points3D_copy[i].data());
  }

  if (problem.NumResiduals() > 0) {
    SetQuaternionManifold(&problem, rig_from_world_rotation);

    // Camera parameterization.
    for (size_t i = 0; i < cameras->size(); i++) {
      if (camera_counts[i] == 0) continue;
      Camera& camera = cameras->at(i);

      // We don't optimize the rig parameters (it's likely under-constrained)
      problem.SetParameterBlockConstant(
          cams_from_rig_copy[i].rotation.coeffs().data());
      problem.SetParameterBlockConstant(
          cams_from_rig_copy[i].translation.data());

      if (!options.refine_focal_length && !options.refine_extra_params) {
        problem.SetParameterBlockConstant(camera.params.data());
      } else {
        // Always set the principal point as fixed.
        std::vector<int> camera_params_const;
        const span<const size_t> principal_point_idxs =
            camera.PrincipalPointIdxs();
        camera_params_const.insert(camera_params_const.end(),
                                   principal_point_idxs.begin(),
                                   principal_point_idxs.end());

        if (!options.refine_focal_length) {
          const span<const size_t> focal_length_idxs = camera.FocalLengthIdxs();
          camera_params_const.insert(camera_params_const.end(),
                                     focal_length_idxs.begin(),
                                     focal_length_idxs.end());
        }

        if (!options.refine_extra_params) {
          const span<const size_t> extra_params_idxs = camera.ExtraParamsIdxs();
          camera_params_const.insert(camera_params_const.end(),
                                     extra_params_idxs.begin(),
                                     extra_params_idxs.end());
        }

        if (camera_params_const.size() == camera.params.size()) {
          problem.SetParameterBlockConstant(camera.params.data());
        } else {
          SetSubsetManifold(static_cast<int>(camera.params.size()),
                            camera_params_const,
                            &problem,
                            camera.params.data());
        }
      }
    }
  }

  ceres::Solver::Options solver_options;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.logging_type = ceres::LoggingType::SILENT;

  // The overhead of creating threads is too large.
  solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  if (options.print_summary || VLOG_IS_ON(1)) {
    PrintSolverSummary(summary, "Pose refinement report");
  }

  if (problem.NumResiduals() > 0 && rig_from_world_cov != nullptr) {
    ceres::Covariance::Options options;
    ceres::Covariance covariance(options);
    std::vector<const double*> parameter_blocks = {rig_from_world_rotation,
                                                   rig_from_world_translation};
    if (!covariance.Compute(parameter_blocks, &problem)) {
      return false;
    }
    covariance.GetCovarianceMatrixInTangentSpace(parameter_blocks,
                                                 rig_from_world_cov->data());
  }

  return summary.IsSolutionUsable();
}

}  // namespace colmap
