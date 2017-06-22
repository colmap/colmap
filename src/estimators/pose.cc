// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "estimators/pose.h"

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/essential_matrix.h"
#include "base/pose.h"
#include "estimators/epnp.h"
#include "estimators/essential_matrix.h"
#include "estimators/p3p.h"
#include "optim/bundle_adjustment.h"
#include "util/misc.h"
#include "util/threading.h"

namespace colmap {
namespace {

typedef LORANSAC<P3PEstimator, EPnPEstimator> AbsolutePoseRANSAC_t;

void EstimateAbsolutePoseKernel(const Camera& camera,
                                const double focal_length_factor,
                                const std::vector<Eigen::Vector2d>& points2D,
                                const std::vector<Eigen::Vector3d>& points3D,
                                const RANSACOptions& options,
                                AbsolutePoseRANSAC_t::Report* report) {
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
  AbsolutePoseRANSAC_t ransac(custom_options);
  *report = ransac.Estimate(points2D_N, points3D);
}

}  // namespace

bool EstimateAbsolutePose(const AbsolutePoseEstimationOptions& options,
                          const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D,
                          Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                          Camera* camera, size_t* num_inliers,
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
    for (double f = 0; f <= 1.0; f += fstep) {
      focal_length_factors.push_back(options.min_focal_length_ratio +
                                     fscale * f * f);
    }
  } else {
    focal_length_factors.reserve(1);
    focal_length_factors.push_back(1);
  }

  std::vector<std::future<void>> futures;
  futures.resize(focal_length_factors.size());
  std::vector<typename AbsolutePoseRANSAC_t::Report,
              Eigen::aligned_allocator<typename AbsolutePoseRANSAC_t::Report>>
      reports;
  reports.resize(focal_length_factors.size());

  ThreadPool thread_pool(std::min(
      options.num_threads, static_cast<int>(focal_length_factors.size())));

  for (size_t i = 0; i < focal_length_factors.size(); ++i) {
    futures[i] = thread_pool.AddTask(
        EstimateAbsolutePoseKernel, *camera, focal_length_factors[i], points2D,
        points3D, options.ransac_options, &reports[i]);
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
                            Eigen::Vector4d* qvec, Eigen::Vector3d* tvec) {
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
  PoseFromEssentialMatrix(report.model, inliers1, inliers2, &R, tvec,
                          &points3D);

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
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                        Camera* camera) {
  CHECK_EQ(inlier_mask.size(), points2D.size());
  CHECK_EQ(points2D.size(), points3D.size());
  options.Check();

  ceres::LossFunction* loss_function =
      new ceres::CauchyLoss(options.loss_function_scale);

  double* camera_params_data = camera->ParamsData();
  double* qvec_data = qvec->data();
  double* tvec_data = tvec->data();

  std::vector<Eigen::Vector3d> points3D_copy = points3D;

  ceres::Problem problem;

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
    problem.AddResidualBlock(cost_function, loss_function, qvec_data,   \
                             tvec_data, points3D_copy[i].data(),        \
                             camera_params_data);                       \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    problem.SetParameterBlockConstant(points3D_copy[i].data());
  }

  if (problem.NumResiduals() > 0) {
    // Quaternion parameterization.
    *qvec = NormalizeQuaternion(*qvec);
    ceres::LocalParameterization* quaternion_parameterization =
        new ceres::QuaternionParameterization;
    problem.SetParameterization(qvec_data, quaternion_parameterization);

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
        ceres::SubsetParameterization* camera_params_parameterization =
            new ceres::SubsetParameterization(
                static_cast<int>(camera->NumParams()), camera_params_const);
        problem.SetParameterization(camera->ParamsData(),
                                    camera_params_parameterization);
      }
    }
  }

  ceres::Solver::Options solver_options;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::DENSE_QR;

  // The overhead of creating threads is too large.
  solver_options.num_threads = 1;
  solver_options.num_linear_solver_threads = 1;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options.print_summary) {
    PrintHeading2("Pose refinement report");
    PrintSolverSummary(summary);
  }

  return summary.IsSolutionUsable();
}

bool RefineRelativePose(const ceres::Solver::Options& options,
                        const std::vector<Eigen::Vector2d>& points1,
                        const std::vector<Eigen::Vector2d>& points2,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec) {
  CHECK_EQ(points1.size(), points2.size());

  // CostFunction assumes unit quaternions.
  *qvec = NormalizeQuaternion(*qvec);

  const double kMaxL2Error = 1.0;
  ceres::LossFunction* loss_function = new ceres::CauchyLoss(kMaxL2Error);

  ceres::Problem problem;

  for (size_t i = 0; i < points1.size(); ++i) {
    ceres::CostFunction* cost_function =
        RelativePoseCostFunction::Create(points1[i], points2[i]);
    problem.AddResidualBlock(cost_function, loss_function, qvec->data(),
                             tvec->data());
  }

  ceres::LocalParameterization* quaternion_parameterization =
      new ceres::QuaternionParameterization;
  problem.SetParameterization(qvec->data(), quaternion_parameterization);

  ceres::HomogeneousVectorParameterization* homogeneous_parameterization =
      new ceres::HomogeneousVectorParameterization(3);
  problem.SetParameterization(tvec->data(), homogeneous_parameterization);

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  return summary.IsSolutionUsable();
}

}  // namespace colmap
