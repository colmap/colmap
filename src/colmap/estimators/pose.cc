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

#include "colmap/estimators/pose.h"

#include "colmap/base/camera_models.h"
#include "colmap/base/cost_functions.h"
#include "colmap/base/essential_matrix.h"
#include "colmap/base/pose.h"
#include "colmap/estimators/absolute_pose.h"
#include "colmap/estimators/essential_matrix.h"
#include "colmap/optim/bundle_adjustment.h"
#include "colmap/util/matrix.h"
#include "colmap/util/misc.h"
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
  const std::vector<size_t>& focal_length_idxs = camera.FocalLengthIdxs();
  for (const size_t idx : focal_length_idxs) {
    scaled_camera.Params(idx) *= focal_length_factor;
  }

  // Normalize image coordinates with current camera hypothesis.
  std::vector<Eigen::Vector2d> points2D_N(points2D.size());
  for (size_t i = 0; i < points2D.size(); ++i) {
    points2D_N[i] = scaled_camera.ImageToWorld(points2D[i]);
  }

  // Estimate pose for given focal length.
  auto custom_options = options;
  custom_options.max_error =
      scaled_camera.ImageToWorldThreshold(options.max_error);
  AbsolutePoseRANSAC ransac(custom_options);
  *report = ransac.Estimate(points2D_N, points3D);
}

}  // namespace

bool EstimateAbsolutePose(const AbsolutePoseEstimationOptions& options,
                          const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D,
                          Eigen::Vector4d* qvec,
                          Eigen::Vector3d* tvec,
                          Camera* camera,
                          size_t* num_inliers,
                          std::vector<char>* inlier_mask) {
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
  Eigen::Matrix3x4d proj_matrix;
  *num_inliers = 0;
  inlier_mask->clear();

  // Find best model among all focal lengths.
  for (size_t i = 0; i < focal_length_factors.size(); ++i) {
    futures[i].get();
    const auto report = reports[i];
    if (report.success && report.support.num_inliers > *num_inliers) {
      *num_inliers = report.support.num_inliers;
      proj_matrix = report.model;
      *inlier_mask = report.inlier_mask;
      focal_length_factor = focal_length_factors[i];
    }
  }

  if (*num_inliers == 0) {
    return false;
  }

  // Scale output camera with best estimated focal length.
  if (options.estimate_focal_length && *num_inliers > 0) {
    const std::vector<size_t>& focal_length_idxs = camera->FocalLengthIdxs();
    for (const size_t idx : focal_length_idxs) {
      camera->Params(idx) *= focal_length_factor;
    }
  }

  // Extract pose parameters.
  *qvec = RotationMatrixToQuaternion(proj_matrix.leftCols<3>());
  *tvec = proj_matrix.rightCols<1>();

  if (IsNaN(*qvec) || IsNaN(*tvec)) {
    return false;
  }

  return true;
}

size_t EstimateRelativePose(const RANSACOptions& ransac_options,
                            const std::vector<Eigen::Vector2d>& points1,
                            const std::vector<Eigen::Vector2d>& points2,
                            Eigen::Vector4d* qvec,
                            Eigen::Vector3d* tvec) {
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

  Eigen::Matrix3d R;

  std::vector<Eigen::Vector3d> points3D;
  PoseFromEssentialMatrix(
      report.model, inliers1, inliers2, &R, tvec, &points3D);

  *qvec = RotationMatrixToQuaternion(R);

  if (IsNaN(*qvec) || IsNaN(*tvec)) {
    return 0;
  }

  return points3D.size();
}

bool RefineAbsolutePose(const AbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        Eigen::Vector4d* qvec,
                        Eigen::Vector3d* tvec,
                        Camera* camera,
                        Eigen::Matrix6d* rot_tvec_covariance) {
  CHECK_EQ(inlier_mask.size(), points2D.size());
  CHECK_EQ(points2D.size(), points3D.size());
  options.Check();

  const auto loss_function =
      std::make_unique<ceres::CauchyLoss>(options.loss_function_scale);

  double* camera_params_data = camera->ParamsData();
  double* qvec_data = qvec->data();
  double* tvec_data = tvec->data();

  std::vector<Eigen::Vector3d> points3D_copy = points3D;

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  for (size_t i = 0; i < points2D.size(); ++i) {
    // Skip outlier observations
    if (!inlier_mask[i]) {
      continue;
    }

    ceres::CostFunction* cost_function = nullptr;

    switch (camera->ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                  \
  case CameraModel::kModelId:                                           \
    cost_function =                                                     \
        BundleAdjustmentCostFunction<CameraModel>::Create(points2D[i]); \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    problem.AddResidualBlock(cost_function,
                             loss_function.get(),
                             qvec_data,
                             tvec_data,
                             points3D_copy[i].data(),
                             camera_params_data);
    problem.SetParameterBlockConstant(points3D_copy[i].data());
  }

  if (problem.NumResiduals() > 0) {
    SetQuaternionManifold(&problem, qvec_data);

    // Camera parameterization.
    if (!options.refine_focal_length && !options.refine_extra_params) {
      problem.SetParameterBlockConstant(camera->ParamsData());
    } else {
      // Always set the principal point as fixed.
      std::vector<int> camera_params_const;
      const std::vector<size_t>& principal_point_idxs =
          camera->PrincipalPointIdxs();
      camera_params_const.insert(camera_params_const.end(),
                                 principal_point_idxs.begin(),
                                 principal_point_idxs.end());

      if (!options.refine_focal_length) {
        const std::vector<size_t>& focal_length_idxs =
            camera->FocalLengthIdxs();
        camera_params_const.insert(camera_params_const.end(),
                                   focal_length_idxs.begin(),
                                   focal_length_idxs.end());
      }

      if (!options.refine_extra_params) {
        const std::vector<size_t>& extra_params_idxs =
            camera->ExtraParamsIdxs();
        camera_params_const.insert(camera_params_const.end(),
                                   extra_params_idxs.begin(),
                                   extra_params_idxs.end());
      }

      if (camera_params_const.size() == camera->NumParams()) {
        problem.SetParameterBlockConstant(camera->ParamsData());
      } else {
        SetSubsetManifold(static_cast<int>(camera->NumParams()),
                          camera_params_const,
                          &problem,
                          camera->ParamsData());
      }
    }
  }

  ceres::Solver::Options solver_options;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::DENSE_QR;

  // The overhead of creating threads is too large.
  solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options.print_summary) {
    PrintHeading2("Pose refinement report");
    PrintSolverSummary(summary);
  }

  if (problem.NumResiduals() > 0 && rot_tvec_covariance != nullptr) {
    ceres::Covariance::Options options;
    ceres::Covariance covariance(options);
    std::vector<const double*> parameter_blocks = {qvec_data, tvec_data};
    if (!covariance.Compute(parameter_blocks, &problem)) {
      return false;
    }
    // The rotation covariance is estimated in the tangent space of the
    // quaternion, which corresponds to the 3-DoF axis-angle local
    // parameterization.
    covariance.GetCovarianceMatrixInTangentSpace(parameter_blocks,
                                                 rot_tvec_covariance->data());
  }

  return summary.IsSolutionUsable();
}

bool RefineRelativePose(const ceres::Solver::Options& options,
                        const std::vector<Eigen::Vector2d>& points1,
                        const std::vector<Eigen::Vector2d>& points2,
                        Eigen::Vector4d* qvec,
                        Eigen::Vector3d* tvec) {
  CHECK_EQ(points1.size(), points2.size());

  // CostFunction assumes unit quaternions.
  *qvec = NormalizeQuaternion(*qvec);

  const double kMaxL2Error = 1.0;
  ceres::LossFunction* loss_function = new ceres::CauchyLoss(kMaxL2Error);

  ceres::Problem problem;

  for (size_t i = 0; i < points1.size(); ++i) {
    ceres::CostFunction* cost_function =
        RelativePoseCostFunction::Create(points1[i], points2[i]);
    problem.AddResidualBlock(
        cost_function, loss_function, qvec->data(), tvec->data());
  }

  SetQuaternionManifold(&problem, qvec->data());
  SetSphereManifold<3>(&problem, tvec->data());

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  return summary.IsSolutionUsable();
}

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
    Eigen::Matrix6d* rot_tvec_covariance) {
  CHECK_EQ(points2D.size(), inlier_mask.size());
  CHECK_EQ(points2D.size(), points3D.size());
  CHECK_EQ(points2D.size(), camera_idxs.size());
  CHECK_EQ(rig_qvecs.size(), rig_tvecs.size());
  CHECK_EQ(rig_qvecs.size(), cameras->size());
  CHECK_GE(*std::min_element(camera_idxs.begin(), camera_idxs.end()), 0);
  CHECK_LT(*std::max_element(camera_idxs.begin(), camera_idxs.end()),
           cameras->size());
  options.Check();

  const auto loss_function =
      std::make_unique<ceres::CauchyLoss>(options.loss_function_scale);

  std::vector<double*> cameras_params_data;
  for (size_t i = 0; i < cameras->size(); i++) {
    cameras_params_data.push_back(cameras->at(i).ParamsData());
  }
  std::vector<size_t> camera_counts(cameras->size(), 0);
  double* qvec_data = qvec->data();
  double* tvec_data = tvec->data();

  std::vector<Eigen::Vector3d> points3D_copy = points3D;
  std::vector<Eigen::Vector4d> rig_qvecs_copy = rig_qvecs;
  std::vector<Eigen::Vector3d> rig_tvecs_copy = rig_tvecs;

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

    ceres::CostFunction* cost_function = nullptr;
    switch (cameras->at(camera_idx).ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                     \
  case CameraModel::kModelId:                                              \
    cost_function =                                                        \
        RigBundleAdjustmentCostFunction<CameraModel>::Create(points2D[i]); \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    problem.AddResidualBlock(cost_function,
                             loss_function.get(),
                             qvec_data,
                             tvec_data,
                             rig_qvecs_copy[camera_idx].data(),
                             rig_tvecs_copy[camera_idx].data(),
                             points3D_copy[i].data(),
                             cameras_params_data[camera_idx]);
    problem.SetParameterBlockConstant(points3D_copy[i].data());
  }

  if (problem.NumResiduals() > 0) {
    SetQuaternionManifold(&problem, qvec_data);

    // Camera parameterization.
    for (size_t i = 0; i < cameras->size(); i++) {
      if (camera_counts[i] == 0) continue;
      Camera& camera = cameras->at(i);

      // We don't optimize the rig parameters (it's likely under-constrained)
      problem.SetParameterBlockConstant(rig_qvecs_copy[i].data());
      problem.SetParameterBlockConstant(rig_tvecs_copy[i].data());

      if (!options.refine_focal_length && !options.refine_extra_params) {
        problem.SetParameterBlockConstant(camera.ParamsData());
      } else {
        // Always set the principal point as fixed.
        std::vector<int> camera_params_const;
        const std::vector<size_t>& principal_point_idxs =
            camera.PrincipalPointIdxs();
        camera_params_const.insert(camera_params_const.end(),
                                   principal_point_idxs.begin(),
                                   principal_point_idxs.end());

        if (!options.refine_focal_length) {
          const std::vector<size_t>& focal_length_idxs =
              camera.FocalLengthIdxs();
          camera_params_const.insert(camera_params_const.end(),
                                     focal_length_idxs.begin(),
                                     focal_length_idxs.end());
        }

        if (!options.refine_extra_params) {
          const std::vector<size_t>& extra_params_idxs =
              camera.ExtraParamsIdxs();
          camera_params_const.insert(camera_params_const.end(),
                                     extra_params_idxs.begin(),
                                     extra_params_idxs.end());
        }

        if (camera_params_const.size() == camera.NumParams()) {
          problem.SetParameterBlockConstant(camera.ParamsData());
        } else {
          SetSubsetManifold(static_cast<int>(camera.NumParams()),
                            camera_params_const,
                            &problem,
                            camera.ParamsData());
        }
      }
    }
  }

  ceres::Solver::Options solver_options;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::DENSE_QR;

  // The overhead of creating threads is too large.
  solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options.print_summary) {
    PrintHeading2("Pose refinement report");
    PrintSolverSummary(summary);
  }

  if (problem.NumResiduals() > 0 && rot_tvec_covariance != nullptr) {
    ceres::Covariance::Options options;
    ceres::Covariance covariance(options);
    std::vector<const double*> parameter_blocks = {qvec_data, tvec_data};
    if (!covariance.Compute(parameter_blocks, &problem)) {
      return false;
    }
    covariance.GetCovarianceMatrixInTangentSpace(parameter_blocks,
                                                 rot_tvec_covariance->data());
  }

  return summary.IsSolutionUsable();
}

}  // namespace colmap
