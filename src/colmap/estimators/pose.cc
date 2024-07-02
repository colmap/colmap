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

#include "colmap/estimators/pose.h"

#include "colmap/estimators/absolute_pose.h"
#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/cost_functions.h"
#include "colmap/estimators/essential_matrix.h"
#include "colmap/estimators/manifold.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/pose.h"
#include "colmap/math/matrix.h"
#include "colmap/sensor/models.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"

namespace colmap {
namespace {

typedef LORANSAC<P3PEstimator, EPNPEstimator> AbsolutePoseRANSAC;

void EstimateAbsolutePoseKernel(const Camera& camera,
                                const double focal_length_factor,
                                const std::vector<Eigen::Vector2d>& points2D,
                                const std::vector<Eigen::Vector3d>& points3D,
                                const RANSACOptions& options,
                                AbsolutePoseRANSAC::Report* report) {
  // Scale the focal length by the given factor.
  Camera scaled_camera = camera;
  for (const size_t idx : camera.FocalLengthIdxs()) {
    scaled_camera.params[idx] *= focal_length_factor;
  }

  // Normalize image coordinates with current camera hypothesis.
  std::vector<Eigen::Vector2d> points2D_in_cam(points2D.size());
  for (size_t i = 0; i < points2D.size(); ++i) {
    points2D_in_cam[i] = scaled_camera.CamFromImg(points2D[i]);
  }

  // Estimate pose for given focal length.
  auto custom_options = options;
  custom_options.max_error =
      scaled_camera.CamFromImgThreshold(options.max_error);
  AbsolutePoseRANSAC ransac(custom_options);
  *report = ransac.Estimate(points2D_in_cam, points3D);
}

}  // namespace

bool EstimateAbsolutePose(const AbsolutePoseEstimationOptions& options,
                          const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D,
                          Rigid3d* cam_from_world,
                          Camera* camera,
                          size_t* num_inliers,
                          std::vector<char>* inlier_mask) {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  options.Check();

  std::vector<double> focal_length_factors;
  if (options.estimate_focal_length) {
    // Generate focal length factors using a quadratic function,
    // such that more samples are drawn for small focal lengths
    focal_length_factors.reserve(options.num_focal_length_samples + 1);
    const double fstep = 1.0 / options.num_focal_length_samples;
    const double fscale =
        options.max_focal_length_ratio - options.min_focal_length_ratio;
    double focal = 0.;
    for (size_t i = 0; i <= options.num_focal_length_samples;
         ++i, focal += fstep) {
      focal_length_factors.push_back(options.min_focal_length_ratio +
                                     fscale * focal * focal);
    }
  } else {
    focal_length_factors.reserve(1);
    focal_length_factors.push_back(1);
  }

  std::vector<std::future<void>> futures;
  futures.resize(focal_length_factors.size());
  std::vector<typename AbsolutePoseRANSAC::Report,
              Eigen::aligned_allocator<typename AbsolutePoseRANSAC::Report>>
      reports;
  reports.resize(focal_length_factors.size());

  ThreadPool thread_pool(std::min(
      options.num_threads, static_cast<int>(focal_length_factors.size())));

  for (size_t i = 0; i < focal_length_factors.size(); ++i) {
    futures[i] = thread_pool.AddTask(EstimateAbsolutePoseKernel,
                                     *camera,
                                     focal_length_factors[i],
                                     points2D,
                                     points3D,
                                     options.ransac_options,
                                     &reports[i]);
  }

  double focal_length_factor = 0;
  Eigen::Matrix3x4d cam_from_world_matrix;
  *num_inliers = 0;
  inlier_mask->clear();

  // Find best model among all focal lengths.
  for (size_t i = 0; i < focal_length_factors.size(); ++i) {
    futures[i].get();
    const auto report = reports[i];
    if (report.success && report.support.num_inliers > *num_inliers) {
      *num_inliers = report.support.num_inliers;
      cam_from_world_matrix = report.model;
      *inlier_mask = report.inlier_mask;
      focal_length_factor = focal_length_factors[i];
    }
  }

  if (*num_inliers == 0) {
    return false;
  }

  // Scale output camera with best estimated focal length.
  if (options.estimate_focal_length && *num_inliers > 0) {
    for (const size_t idx : camera->FocalLengthIdxs()) {
      camera->params[idx] *= focal_length_factor;
    }
  }

  *cam_from_world =
      Rigid3d(Eigen::Quaterniond(cam_from_world_matrix.leftCols<3>()),
              cam_from_world_matrix.col(3));

  if (cam_from_world->rotation.coeffs().array().isNaN().any() ||
      cam_from_world->translation.array().isNaN().any()) {
    return false;
  }

  return true;
}

size_t EstimateRelativePose(const RANSACOptions& ransac_options,
                            const std::vector<Eigen::Vector2d>& points1,
                            const std::vector<Eigen::Vector2d>& points2,
                            Rigid3d* cam2_from_cam1) {
  RANSAC<EssentialMatrixFivePointEstimator> ransac(ransac_options);
  const auto report = ransac.Estimate(points1, points2);

  if (!report.success) {
    return 0;
  }

  std::vector<Eigen::Vector2d> inliers1(report.support.num_inliers);
  std::vector<Eigen::Vector2d> inliers2(report.support.num_inliers);

  size_t j = 0;
  for (size_t i = 0; i < points1.size(); ++i) {
    if (report.inlier_mask[i]) {
      inliers1[j] = points1[i];
      inliers2[j] = points2[i];
      j += 1;
    }
  }

  Eigen::Matrix3d cam2_from_cam1_rot_mat;
  std::vector<Eigen::Vector3d> points3D;
  PoseFromEssentialMatrix(report.model,
                          inliers1,
                          inliers2,
                          &cam2_from_cam1_rot_mat,
                          &cam2_from_cam1->translation,
                          &points3D);

  cam2_from_cam1->rotation = Eigen::Quaterniond(cam2_from_cam1_rot_mat);

  if (cam2_from_cam1->rotation.coeffs().array().isNaN().any() ||
      cam2_from_cam1->translation.array().isNaN().any()) {
    return 0;
  }

  return points3D.size();
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
  double* rig_from_world_rotation = cam_from_world->rotation.coeffs().data();
  double* rig_from_world_translation = cam_from_world->translation.data();

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  for (size_t i = 0; i < points2D.size(); ++i) {
    // Skip outlier observations
    if (!inlier_mask[i]) {
      continue;
    }
    problem.AddResidualBlock(
        CameraCostFunction<ReprojErrorConstantPoint3DCostFunction>(
            camera->model_id, points2D[i], points3D[i]),
        loss_function.get(),
        rig_from_world_rotation,
        rig_from_world_translation,
        camera_params);
  }

  if (problem.NumResiduals() > 0) {
    SetQuaternionManifold(&problem, rig_from_world_rotation);

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

  if (problem.NumResiduals() > 0 && cam_from_world_cov != nullptr) {
    ceres::Covariance::Options options;
    ceres::Covariance covariance(options);
    std::vector<const double*> parameter_blocks = {rig_from_world_rotation,
                                                   rig_from_world_translation};
    if (!covariance.Compute(parameter_blocks, &problem)) {
      return false;
    }
    // The rotation covariance is estimated in the tangent space of the
    // quaternion, which corresponds to the 3-DoF axis-angle local
    // parameterization.
    covariance.GetCovarianceMatrixInTangentSpace(parameter_blocks,
                                                 cam_from_world_cov->data());
  }

  return summary.IsSolutionUsable();
}

bool RefineRelativePose(const ceres::Solver::Options& options,
                        const std::vector<Eigen::Vector2d>& points1,
                        const std::vector<Eigen::Vector2d>& points2,
                        Rigid3d* cam2_from_cam1) {
  THROW_CHECK_EQ(points1.size(), points2.size());

  // CostFunction assumes unit quaternions.
  cam2_from_cam1->rotation.normalize();

  double* cam2_from_cam1_rotation = cam2_from_cam1->rotation.coeffs().data();
  double* cam2_from_cam1_translation = cam2_from_cam1->translation.data();

  const double kMaxL2Error = 1.0;
  ceres::LossFunction* loss_function = new ceres::CauchyLoss(kMaxL2Error);

  ceres::Problem problem;

  for (size_t i = 0; i < points1.size(); ++i) {
    ceres::CostFunction* cost_function =
        SampsonErrorCostFunction::Create(points1[i], points2[i]);
    problem.AddResidualBlock(cost_function,
                             loss_function,
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
  Eigen::Matrix3d cam2_from_cam1_rot_mat;
  std::vector<Eigen::Vector3d> points3D;
  PoseFromEssentialMatrix(*E,
                          inlier_points1,
                          inlier_points2,
                          &cam2_from_cam1_rot_mat,
                          &cam2_from_cam1.translation,
                          &points3D);
  cam2_from_cam1.rotation = Eigen::Quaterniond(cam2_from_cam1_rot_mat);

  if (points3D.size() == 0) {
    return false;
  }

  // Refine essential matrix, use all points so that refinement is able to
  // consider points as inliers that were originally outliers.

  const bool refinement_success = RefineRelativePose(
      options, inlier_points1, inlier_points2, &cam2_from_cam1);

  if (!refinement_success) {
    return false;
  }

  *E = EssentialMatrixFromPose(cam2_from_cam1);

  return true;
}

}  // namespace colmap
