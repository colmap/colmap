// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/pose.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/pose_prior.h"
#include "colmap/estimators/cost_functions/reprojection_error.h"
#include "colmap/estimators/cost_functions/sampson_error.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/estimators/solvers/absolute_pose.h"
#include "colmap/estimators/solvers/essential_matrix.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/optim/loransac.h"
#include "colmap/util/logging.h"

namespace colmap {
namespace {

typedef LORANSAC<P3PEstimator, EPNPEstimator> AbsolutePoseRANSAC;
typedef RANSAC<CovariantP3PEstimator, MEstimatorSupportMeasurer>
    CovariantAbsolutePoseRANSAC;

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

void EstimateCovariantAbsolutePoseKernel(
    const Camera& camera,
    const double focal_length_factor,
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const std::vector<Eigen::Matrix3d>& points3D_cov,
    const RANSACOptions& options,
    AbsolutePoseRANSAC::Report* report) {
  constexpr double kSigmaInlierFactor = 3.0;

  // Scale the focal length by the given factor.
  Camera scaled_camera = camera;
  for (const size_t idx : camera.FocalLengthIdxs()) {
    scaled_camera.params[idx] *= focal_length_factor;
  }

  const double max_error_in_cam =
      scaled_camera.CamFromImgThreshold(options.max_error);
  const Eigen::Matrix2d point2D_cov = (max_error_in_cam / kSigmaInlierFactor) *
                                      (max_error_in_cam / kSigmaInlierFactor) *
                                      Eigen::Matrix2d::Identity();

  // Normalize image coordinates with current camera hypothesis.
  std::vector<std::pair<Eigen::Vector2d, Eigen::Matrix2d>> points2D_with_cov(
      points2D.size());
  for (size_t i = 0; i < points2D.size(); ++i) {
    points2D_with_cov[i] = {scaled_camera.CamFromImg(points2D[i]), point2D_cov};
  }

  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> points3D_with_cov(
      points3D.size());
  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D_with_cov[i] = {points3D[i], points3D_cov[i]};
  }

  // Estimate pose for given focal length.
  auto custom_options = options;
  // TODO: Do we need to account for the log(det(cov)) term in the residual?
  custom_options.max_error = kSigmaInlierFactor;
  CovariantAbsolutePoseRANSAC ransac(custom_options);
  const auto covariant_report =
      ransac.Estimate(points2D_with_cov, points3D_with_cov);

  report->success = covariant_report.success;
  report->num_trials = covariant_report.num_trials;
  report->support.num_inliers = covariant_report.support.num_inliers;
  report->inlier_mask = covariant_report.inlier_mask;
  report->model = covariant_report.model;
}

}  // namespace

bool EstimateAbsolutePose(const AbsolutePoseEstimationOptions& options,
                          const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D,
                          const std::vector<Eigen::Matrix3d>* points3D_cov,
                          Rigid3d* cam_from_world,
                          Camera* camera,
                          size_t* num_inliers,
                          std::vector<char>* inlier_mask) {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  if (points3D_cov != nullptr) {
    THROW_CHECK_EQ(points2D.size(), points3D_cov->size());
  }

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
    if (points3D_cov == nullptr) {
      futures[i] = thread_pool.AddTask(EstimateAbsolutePoseKernel,
                                       *camera,
                                       focal_length_factors[i],
                                       points2D,
                                       points3D,
                                       options.ransac_options,
                                       &reports[i]);
    } else {
      futures[i] = thread_pool.AddTask(EstimateCovariantAbsolutePoseKernel,
                                       *camera,
                                       focal_length_factors[i],
                                       points2D,
                                       points3D,
                                       *points3D_cov,
                                       options.ransac_options,
                                       &reports[i]);
    }
  }

  double focal_length_factor = 0;
  Eigen::Matrix3x4d cam_from_world_matrix;
  *num_inliers = 0;
  inlier_mask->clear();

  if (options.estimate_focal_length) {
    const Eigen::Vector2d principal_point(camera->PrincipalPointX(),
                                          camera->PrincipalPointY());
    std::vector<Eigen::Vector2d> points2D_centered(points2D.size());
    for (size_t i = 0; i < points2D.size(); ++i) {
      points2D_centered[i] = points2D[i] - principal_point;
    }
    const span<const size_t> focal_length_idxs = camera->FocalLengthIdxs();
    const bool share_focal_length = focal_length_idxs.size() == 1;
    LORANSAC<P4PFEstimator, P4PFEstimator> ransac(
        options.ransac_options,
        P4PFEstimator(share_focal_length),
        P4PFEstimator(share_focal_length));
    auto report = ransac.Estimate(points2D_centered, points3D);
    if (report.success) {
      *cam_from_world = report.model.cam_from_world;
      for (size_t k = 0; k < focal_length_idxs.size(); ++k) {
        camera->params[focal_length_idxs[k]] = report.model.focal_lengths[k];
      }
      *num_inliers = report.support.num_inliers;
      *inlier_mask = std::move(report.inlier_mask);
      return true;
    }
  } else {
    std::vector<P3PEstimator::X_t> points2D_with_rays(points2D.size());
    for (size_t i = 0; i < points2D.size(); ++i) {
      points2D_with_rays[i].image_point = points2D[i];
      points2D_with_rays[i].camera_ray =
          camera->CamRayFromImg(points2D[i]).value_or(Eigen::Vector3d::Zero());
    }

    ImgFromCamFunc img_from_cam_func =
        [camera](const Eigen::Vector3d& cam_point) {
          return camera->ImgFromCam(cam_point);
        };
    LORANSAC<P3PEstimator, P3PEstimator> ransac(
        options.ransac_options,
        P3PEstimator(img_from_cam_func),
        P3PEstimator(img_from_cam_func));
    auto report = ransac.Estimate(points2D_with_rays, points3D);
    if (report.success) {
      *cam_from_world = report.model;
      *num_inliers = report.support.num_inliers;
      *inlier_mask = std::move(report.inlier_mask);
      return true;
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

  LOG(INFO) << "Absolute pose estimation with " << *num_inliers << " inliers";

  return true;
}

bool EstimateRelativePose(const RANSACOptions& ransac_options,
                          const std::vector<CamRayWithJac>& cam_rays1_with_jac,
                          const std::vector<CamRayWithJac>& cam_rays2_with_jac,
                          Rigid3d* cam2_from_cam1,
                          size_t* num_inliers,
                          std::vector<char>* inlier_mask) {
  THROW_CHECK_EQ(cam_rays1_with_jac.size(), cam_rays2_with_jac.size());

  LORANSAC<EssentialMatrixTangentSampsonEstimator,
           EssentialMatrixTangentSampsonEstimator>
      ransac(ransac_options);
  auto report = ransac.Estimate(cam_rays1_with_jac, cam_rays2_with_jac);

  if (!report.success) {
    return false;
  }

  std::vector<Eigen::Vector3d> inlier_cam_rays1;
  std::vector<Eigen::Vector3d> inlier_cam_rays2;
  inlier_cam_rays1.reserve(report.support.num_inliers);
  inlier_cam_rays2.reserve(report.support.num_inliers);
  for (size_t i = 0; i < cam_rays1_with_jac.size(); ++i) {
    if (report.inlier_mask[i]) {
      inlier_cam_rays1.push_back(cam_rays1_with_jac[i].ray);
      inlier_cam_rays2.push_back(cam_rays2_with_jac[i].ray);
    }
  }

  std::vector<int> valid_indices;
  PoseFromEssentialMatrix(report.model,
                          inlier_cam_rays1,
                          inlier_cam_rays2,
                          cam2_from_cam1,
                          &valid_indices);

  if (cam2_from_cam1->rotation().coeffs().array().isNaN().any() ||
      cam2_from_cam1->translation().array().isNaN().any()) {
    return false;
  }

  *num_inliers = report.support.num_inliers;
  *inlier_mask = std::move(report.inlier_mask);

  return !valid_indices.empty();
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

  // CostFunction assumes unit quaternions.
  cam_from_world->rotation().normalize();

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
        cam_from_world->params.data(),
        camera->params.data());
  }

  if (options.use_position_prior) {
    problem.AddResidualBlock(
        CovarianceWeightedCostFunctor<AbsolutePosePositionPriorCostFunctor>::
            Create(options.position_prior_covariance,
                   options.position_prior_in_world),
        nullptr,
        cam_from_world->params.data());
  }

  if (problem.NumResiduals() > 0) {
    if (problem.HasParameterBlock(camera->params.data())) {
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
          const span<const size_t> focal_length_idxs =
              camera->FocalLengthIdxs();
          camera_params_const.insert(camera_params_const.end(),
                                     focal_length_idxs.begin(),
                                     focal_length_idxs.end());
        }

        if (!options.refine_extra_params) {
          const span<const size_t> extra_params_idxs =
              camera->ExtraParamsIdxs();
          camera_params_const.insert(camera_params_const.end(),
                                     extra_params_idxs.begin(),
                                     extra_params_idxs.end());
        }

        if (camera_params_const.size() == camera->params.size()) {
          problem.SetParameterBlockConstant(camera->params.data());
        } else {
          SetManifold(
              &problem,
              camera->params.data(),
              CreateSubsetManifold(camera->params.size(), camera_params_const));
        }
      }
    }

    SetManifold(&problem,
                cam_from_world->params.data(),
                CreateProductManifold(CreateEigenQuaternionManifold(),
                                      CreateEuclideanManifold<3>()));
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
    std::vector<const double*> parameter_blocks = {
        cam_from_world->params.data()};
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
                        const std::vector<CamRayWithJac>& cam_rays1_with_jac,
                        const std::vector<CamRayWithJac>& cam_rays2_with_jac,
                        Rigid3d* cam2_from_cam1) {
  THROW_CHECK_EQ(cam_rays1_with_jac.size(), cam_rays2_with_jac.size());
  THROW_CHECK_EQ(cam_rays1_with_jac.size(), inlier_mask.size());

  // CostFunction assumes unit quaternions.
  cam2_from_cam1->rotation().normalize();

  // No robust loss: the observations are already RANSAC-gated inliers, so the
  // refinement is plain least squares.
  ceres::Problem problem;

  for (size_t i = 0; i < cam_rays1_with_jac.size(); ++i) {
    // Skip outliers and unprojectable (zero) rays, which the residual scores as
    // a perfect fit rather than rejecting.
    if (!inlier_mask[i] || cam_rays1_with_jac[i].ray.isZero() ||
        cam_rays2_with_jac[i].ray.isZero()) {
      continue;
    }
    ceres::CostFunction* cost_function = TangentSampsonErrorCostFunctor::Create(
        cam_rays1_with_jac[i], cam_rays2_with_jac[i]);
    problem.AddResidualBlock(cost_function,
                             /*loss_function=*/nullptr,
                             cam2_from_cam1->params.data());
  }

  SetManifold(&problem,
              cam2_from_cam1->params.data(),
              CreateProductManifold(CreateEigenQuaternionManifold(),
                                    CreateSphereManifold<3>()));

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  return summary.IsSolutionUsable();
}

bool RefineEssentialMatrix(const ceres::Solver::Options& options,
                           const std::vector<CamRayWithJac>& cam_rays1_with_jac,
                           const std::vector<CamRayWithJac>& cam_rays2_with_jac,
                           const std::vector<char>& inlier_mask,
                           Eigen::Matrix3d* E) {
  THROW_CHECK_EQ(cam_rays1_with_jac.size(), cam_rays2_with_jac.size());
  THROW_CHECK_EQ(cam_rays1_with_jac.size(), inlier_mask.size());

  // Collect inliers. PoseFromEssentialMatrix needs the bare bearings. The
  // refinement additionally needs their unprojection Jacobians.
  std::vector<CamRayWithJac> inlier_cam_rays1_with_jac;
  std::vector<CamRayWithJac> inlier_cam_rays2_with_jac;
  std::vector<Eigen::Vector3d> inlier_rays1;
  std::vector<Eigen::Vector3d> inlier_rays2;
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      inlier_cam_rays1_with_jac.push_back(cam_rays1_with_jac[i]);
      inlier_cam_rays2_with_jac.push_back(cam_rays2_with_jac[i]);
      inlier_rays1.push_back(cam_rays1_with_jac[i].ray);
      inlier_rays2.push_back(cam_rays2_with_jac[i].ray);
    }
  }

  // Extract relative pose from essential matrix.
  Rigid3d cam2_from_cam1;
  std::vector<int> valid_indices;
  PoseFromEssentialMatrix(
      *E, inlier_rays1, inlier_rays2, &cam2_from_cam1, &valid_indices);

  if (valid_indices.empty()) {
    return false;
  }

  // Refine over all inliers (robustness came from the RANSAC selection).
  if (!RefineRelativePose(
          options,
          std::vector<char>(inlier_cam_rays1_with_jac.size(), true),
          inlier_cam_rays1_with_jac,
          inlier_cam_rays2_with_jac,
          &cam2_from_cam1)) {
    return false;
  }

  *E = EssentialMatrixFromPose(cam2_from_cam1);

  return true;
}

}  // namespace colmap
