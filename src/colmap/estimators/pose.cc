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

#include "colmap/estimators/pose.h"

#include "colmap/estimators/absolute_pose.h"
#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/cost_functions.h"
#include "colmap/estimators/essential_matrix.h"
#include "colmap/estimators/manifold.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/sensor/models.h"
#include "colmap/util/logging.h"

namespace colmap {

bool EstimateAbsolutePose(const AbsolutePoseEstimationOptions& options,
                          const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D,
                          Rigid3d* cam_from_world,
                          Camera* camera,
                          size_t* num_inliers,
                          std::vector<char>* inlier_mask) {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  options.Check();

  *num_inliers = 0;
  inlier_mask->clear();

  if (options.estimate_focal_length) {
    // TODO(jsch): Implement non-minimal solver for LORANSAC refinement.
    // Experiments showed marginal difference between RANSAC/LORANSAC for PNPF
    // after refining the estimates of this function using RefineAbsolutePose.
    const Eigen::Vector2d principal_point(camera->PrincipalPointX(),
                                          camera->PrincipalPointY());
    std::vector<Eigen::Vector2d> points2D_centered(points2D.size());
    for (size_t i = 0; i < points2D.size(); ++i) {
      points2D_centered[i] = points2D[i] - principal_point;
    }
    RANSAC<P4PFEstimator> ransac(options.ransac_options);
    auto report = ransac.Estimate(points2D_centered, points3D);
    if (report.success) {
      *cam_from_world =
          Rigid3d(Eigen::Quaterniond(report.model.cam_from_world.leftCols<3>()),
                  report.model.cam_from_world.col(3));
      for (const size_t idx : camera->FocalLengthIdxs()) {
        camera->params[idx] = report.model.focal_length;
      }
      *num_inliers = report.support.num_inliers;
      *inlier_mask = std::move(report.inlier_mask);
      return true;
    }
  } else {
    std::vector<P3PEstimator::X_t> points2D_with_rays(points2D.size());
    for (size_t i = 0; i < points2D.size(); ++i) {
      points2D_with_rays[i].image_point = points2D[i];
      if (const std::optional<Eigen::Vector2d> cam_point =
              camera->CamFromImg(points2D[i]);
          cam_point) {
        points2D_with_rays[i].camera_ray =
            cam_point->homogeneous().normalized();
      } else {
        points2D_with_rays[i].camera_ray.setZero();
      }
    }

    ImgFromCamFunc img_from_cam_func =
        std::bind(&Camera::ImgFromCam, camera, std::placeholders::_1);
    LORANSAC<P3PEstimator, EPNPEstimator> ransac(
        options.ransac_options,
        P3PEstimator(img_from_cam_func),
        EPNPEstimator(img_from_cam_func));
    auto report = ransac.Estimate(points2D_with_rays, points3D);
    if (report.success) {
      *cam_from_world = Rigid3d(Eigen::Quaterniond(report.model.leftCols<3>()),
                                report.model.col(3));
      *num_inliers = report.support.num_inliers;
      *inlier_mask = std::move(report.inlier_mask);
      return true;
    }
  }

  return false;
}

bool EstimateRelativePose(const RANSACOptions& ransac_options,
                          const std::vector<Eigen::Vector2d>& cam_points1,
                          const std::vector<Eigen::Vector2d>& cam_points2,
                          Rigid3d* cam2_from_cam1,
                          size_t* num_inliers,
                          std::vector<char>* inlier_mask) {
  THROW_CHECK_EQ(cam_points1.size(), cam_points2.size());

  RANSAC<EssentialMatrixFivePointEstimator> ransac(ransac_options);
  auto report = ransac.Estimate(cam_points1, cam_points2);

  if (!report.success) {
    return false;
  }

  std::vector<Eigen::Vector2d> inliers1(report.support.num_inliers);
  std::vector<Eigen::Vector2d> inliers2(report.support.num_inliers);

  size_t j = 0;
  for (size_t i = 0; i < cam_points1.size(); ++i) {
    if (report.inlier_mask[i]) {
      inliers1[j] = cam_points1[i];
      inliers2[j] = cam_points2[i];
      j += 1;
    }
  }

  std::vector<Eigen::Vector3d> points3D;
  PoseFromEssentialMatrix(
      report.model, inliers1, inliers2, cam2_from_cam1, &points3D);

  if (cam2_from_cam1->rotation.coeffs().array().isNaN().any() ||
      cam2_from_cam1->translation.array().isNaN().any()) {
    return false;
  }

  *num_inliers = report.support.num_inliers;
  *inlier_mask = std::move(report.inlier_mask);

  return !points3D.empty();
}

bool RefineAbsolutePose(const AbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        Rigid3d* cam_from_world,
                        Camera* camera,
                        Eigen::Matrix6d* cam_from_world_cov) {
  THROW_CHECK_EQ(inlier_mask.size(), points2D.size());
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  options.Check();

  const auto loss_function =
      std::make_unique<ceres::CauchyLoss>(options.loss_function_scale);

  double* camera_params = camera->params.data();
  double* cam_from_world_rotation = cam_from_world->rotation.coeffs().data();
  double* cam_from_world_translation = cam_from_world->translation.data();

  // CostFunction assumes unit quaternions.
  cam_from_world->rotation.normalize();

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  for (size_t i = 0; i < points2D.size(); ++i) {
    // Skip outlier observations
    if (!inlier_mask[i]) {
      continue;
    }
    problem.AddResidualBlock(
        CreateCameraCostFunction<ReprojErrorConstantPoint3DCostFunctor>(
            camera->model_id, points2D[i], points3D[i]),
        loss_function.get(),
        cam_from_world_rotation,
        cam_from_world_translation,
        camera_params);
  }

  if (problem.NumResiduals() > 0) {
    SetQuaternionManifold(&problem, cam_from_world_rotation);

    // Camera parameterization.
    if (!options.refine_focal_length && !options.refine_extra_params) {
      problem.SetParameterBlockConstant(camera->params.data());
    } else {
      // Always set the principal point as fixed.
      std::vector<int> camera_params_const;
      const span<const size_t> principal_point_idxs =
          camera->PrincipalPointIdxs();
      camera_params_const.insert(camera_params_const.end(),
                                 principal_point_idxs.begin(),
                                 principal_point_idxs.end());

      if (!options.refine_focal_length) {
        const span<const size_t> focal_length_idxs = camera->FocalLengthIdxs();
        camera_params_const.insert(camera_params_const.end(),
                                   focal_length_idxs.begin(),
                                   focal_length_idxs.end());
      }

      if (!options.refine_extra_params) {
        const span<const size_t> extra_params_idxs = camera->ExtraParamsIdxs();
        camera_params_const.insert(camera_params_const.end(),
                                   extra_params_idxs.begin(),
                                   extra_params_idxs.end());
      }

      if (camera_params_const.size() == camera->params.size()) {
        problem.SetParameterBlockConstant(camera->params.data());
      } else {
        SetSubsetManifold(static_cast<int>(camera->params.size()),
                          camera_params_const,
                          &problem,
                          camera->params.data());
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

  if (!summary.IsSolutionUsable()) {
    return false;
  }

  if (problem.NumResiduals() > 0 && cam_from_world_cov != nullptr) {
    ceres::Covariance::Options options;
    ceres::Covariance covariance(options);
    std::vector<const double*> parameter_blocks = {cam_from_world_rotation,
                                                   cam_from_world_translation};
    if (!covariance.Compute(parameter_blocks, &problem)) {
      return false;
    }
    // The rotation covariance is estimated in the tangent space of the
    // quaternion, which corresponds to the 3-DoF axis-angle local
    // parameterization.
    covariance.GetCovarianceMatrixInTangentSpace(parameter_blocks,
                                                 cam_from_world_cov->data());
  }

  return true;
}

bool RefineRelativePose(const ceres::Solver::Options& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& cam_points1,
                        const std::vector<Eigen::Vector2d>& cam_points2,
                        Rigid3d* cam2_from_cam1) {
  THROW_CHECK_EQ(cam_points1.size(), cam_points2.size());
  THROW_CHECK_EQ(cam_points1.size(), inlier_mask.size());

  // CostFunction assumes unit quaternions.
  cam2_from_cam1->rotation.normalize();

  double* cam2_from_cam1_rotation = cam2_from_cam1->rotation.coeffs().data();
  double* cam2_from_cam1_translation = cam2_from_cam1->translation.data();

  constexpr double kMaxL2Error = 1.0;
  const auto loss_function = std::make_unique<ceres::CauchyLoss>(kMaxL2Error);

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  for (size_t i = 0; i < cam_points1.size(); ++i) {
    // Skip outlier observations
    if (!inlier_mask[i]) {
      continue;
    }
    ceres::CostFunction* cost_function =
        SampsonErrorCostFunctor::Create(cam_points1[i], cam_points2[i]);
    problem.AddResidualBlock(cost_function,
                             loss_function.get(),
                             cam2_from_cam1_rotation,
                             cam2_from_cam1_translation);
  }

  SetQuaternionManifold(&problem, cam2_from_cam1_rotation);
  SetSphereManifold<3>(&problem, cam2_from_cam1_translation);

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  return summary.IsSolutionUsable();
}

bool RefineEssentialMatrix(const ceres::Solver::Options& options,
                           const std::vector<Eigen::Vector2d>& points1,
                           const std::vector<Eigen::Vector2d>& points2,
                           const std::vector<char>& inlier_mask,
                           Eigen::Matrix3d* E) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  THROW_CHECK_EQ(points1.size(), inlier_mask.size());

  // Extract inlier points for decomposing the essential matrix into
  // rotation and translation components.

  size_t num_inliers = 0;
  for (const auto inlier : inlier_mask) {
    if (inlier) {
      num_inliers += 1;
    }
  }

  std::vector<Eigen::Vector2d> inlier_points1(num_inliers);
  std::vector<Eigen::Vector2d> inlier_points2(num_inliers);
  size_t j = 0;
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      inlier_points1[j] = points1[i];
      inlier_points2[j] = points2[i];
      j += 1;
    }
  }

  // Extract relative pose from essential matrix.
  Rigid3d cam2_from_cam1;
  std::vector<Eigen::Vector3d> points3D;
  PoseFromEssentialMatrix(
      *E, inlier_points1, inlier_points2, &cam2_from_cam1, &points3D);

  if (points3D.size() == 0) {
    return false;
  }

  // Refine essential matrix, use all points so that refinement is able to
  // consider points as inliers that were originally outliers.

  const bool refinement_success =
      RefineRelativePose(options,
                         std::vector<char>(num_inliers, true),
                         inlier_points1,
                         inlier_points2,
                         &cam2_from_cam1);

  if (!refinement_success) {
    return false;
  }

  *E = EssentialMatrixFromPose(cam2_from_cam1);

  return true;
}

}  // namespace colmap
