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

// Fitting of camera intrinsics to dense image-point/camera-ray correspondences,
// following
// AnyCalib (https://github.com/javrtg/AnyCalib, Apache-2.0):
//
//   Tirado-Garin & Civera, "AnyCalib: On-Manifold Learning for Model-Agnostic
//   Single-View Camera Calibration", ICCV 2025.
//
// The linear solvers below are ports of the `fit` methods of
// `anycalib/cameras/pinhole.py` / `anycalib/cameras/radial.py`. The nonlinear
// refinement minimizes pixel residuals of the projected rays with Ceres
// (autodiff) directly on any COLMAP perspective model. This differs from
// upstream `GaussNewtonCalib`, which refines tangent-space residuals with
// per-model analytic Jacobians; the pixel formulation needs no per-model
// derivatives, as every model's projection is already templated for Jets.

#include "colmap/calibration/ray_fitting.h"

#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>

#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace colmap {
namespace {

// Whether a camera ray is within the admissible field of view (AnyCalib
// `check_within_fov` in `pinhole.py`).
bool IsWithinFov(const Eigen::Vector3d& cam_ray, double max_fov_deg) {
  const double max_fov = std::min(max_fov_deg, 179.0) * EIGEN_PI / 180.0;
  return cam_ray.head<2>().norm() < cam_ray.z() * std::tan(0.5 * max_fov);
}

struct LinearSolution {
  Eigen::VectorXd params;
  bool success = false;
};

LinearSolution SolveNormalEquations(const Eigen::MatrixXd& AtA,
                                    const Eigen::VectorXd& Atb) {
  LinearSolution result;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(AtA);
  if (ldlt.info() != Eigen::Success) {
    return result;
  }
  result.params = ldlt.solve(Atb);
  result.success = result.params.allFinite();
  return result;
}

struct PinholeFit {
  double fx = 0;
  double fy = 0;
  double cx = 0;
  double cy = 0;
  bool success = false;
};

// Closed-form pinhole fit, ported from `Pinhole.fit` (unknown principal
// point, no covariances): solves for inverse focals and scaled principal
// point in the reparameterized error space.
PinholeFit FitPinholeLinear(const std::vector<Eigen::Vector2d>& img_points,
                            const std::vector<Eigen::Vector3d>& cam_rays,
                            double max_fov_deg) {
  PinholeFit result;
  const double eps = std::numeric_limits<double>::epsilon();

  Eigen::Vector2d norm_factor(0, 0);
  for (const auto& img_point : img_points) {
    norm_factor.x() = std::max(norm_factor.x(), img_point.x());
    norm_factor.y() = std::max(norm_factor.y(), img_point.y());
  }
  if (norm_factor.x() <= 0 || norm_factor.y() <= 0) {
    return result;
  }

  Eigen::Matrix4d AtA = Eigen::Matrix4d::Zero();
  Eigen::Vector4d Atb = Eigen::Vector4d::Zero();
  for (size_t i = 0; i < img_points.size(); ++i) {
    if (!IsWithinFov(cam_rays[i], max_fov_deg)) {
      continue;
    }
    const Eigen::Vector2d proj =
        cam_rays[i].head<2>() / std::max(cam_rays[i].z(), eps);
    Eigen::Matrix<double, 2, 4> A = Eigen::Matrix<double, 2, 4>::Zero();
    A(0, 0) = img_points[i].x() / norm_factor.x();
    A(1, 1) = img_points[i].y() / norm_factor.y();
    A(0, 2) = -1;
    A(1, 3) = -1;
    AtA += A.transpose() * A;
    Atb += A.transpose() * proj;
  }

  const LinearSolution sol = SolveNormalEquations(AtA, Atb);
  if (!sol.success || sol.params.x() == 0 || sol.params.y() == 0) {
    return result;
  }
  result.fx = norm_factor.x() / sol.params.x();
  result.fy = norm_factor.y() / sol.params.y();
  result.cx = sol.params.z() * result.fx;
  result.cy = sol.params.w() * result.fy;
  result.success = std::isfinite(result.fx) && std::isfinite(result.fy) &&
                   std::isfinite(result.cx) && std::isfinite(result.cy) &&
                   result.fx > 0 && result.fy > 0;
  return result;
}

struct RadialFit {
  double fx = 0;
  double fy = 0;
  double cx = 0;
  double cy = 0;
  Eigen::VectorXd k;
  bool success = false;
};

// Closed-form radial fit, ported from `Radial.fit` (unknown principal point,
// no covariances). Projection: u = fx * X/Z * (1 + k1 r^2 + ...) + cx.
RadialFit FitRadialLinear(const std::vector<Eigen::Vector2d>& img_points,
                          const std::vector<Eigen::Vector3d>& cam_rays,
                          int num_k,
                          double max_fov_deg) {
  RadialFit result;
  const double eps = std::numeric_limits<double>::epsilon();

  Eigen::Vector2d fac(0, 0);
  for (const auto& img_point : img_points) {
    fac.x() = std::max(fac.x(), img_point.x());
    fac.y() = std::max(fac.y(), img_point.y());
  }
  if (fac.x() <= 0 || fac.y() <= 0) {
    return result;
  }

  const int dim = 4 + num_k;
  Eigen::MatrixXd AtA = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::VectorXd Atb = Eigen::VectorXd::Zero(dim);
  for (size_t i = 0; i < img_points.size(); ++i) {
    if (!IsWithinFov(cam_rays[i], max_fov_deg)) {
      continue;
    }
    const Eigen::Vector2d proj =
        cam_rays[i].head<2>() / std::max(cam_rays[i].z(), eps);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, dim);
    A(0, 0) = img_points[i].x() / fac.x();
    A(1, 1) = img_points[i].y() / fac.y();
    A(0, 2) = -1;
    A(1, 3) = -1;
    const double radii_u2 = proj.squaredNorm();
    Eigen::Vector2d proj_radii = -proj * radii_u2;
    for (int j = 0; j < num_k; ++j) {
      A(0, 4 + j) = proj_radii.x();
      A(1, 4 + j) = proj_radii.y();
      proj_radii *= radii_u2;
    }
    AtA += A.transpose() * A;
    Atb += A.transpose() * proj;
  }

  const LinearSolution sol = SolveNormalEquations(AtA, Atb);
  if (!sol.success || sol.params.x() == 0 || sol.params.y() == 0) {
    return result;
  }
  result.fx = fac.x() / sol.params.x();
  result.fy = fac.y() / sol.params.y();
  result.cx = sol.params.z() * result.fx;
  result.cy = sol.params.w() * result.fy;
  result.k = sol.params.tail(num_k);
  result.success = std::isfinite(result.fx) && std::isfinite(result.fy) &&
                   std::isfinite(result.cx) && std::isfinite(result.cy) &&
                   result.k.allFinite() && result.fx > 0 && result.fy > 0;
  return result;
}

// Whether the model's distortion starts with radial k1[, k2] coefficients
// matching AnyCalib's radial projection, so closed-form radial init applies.
bool HasRadialDistortionPrefix(CameraModelId model_id, int* num_k) {
  switch (model_id) {
    case CameraModelId::kSimpleRadial:
      *num_k = 1;
      return true;
    case CameraModelId::kRadial:
    case CameraModelId::kOpenCV:
    case CameraModelId::kFullOpenCV:
      *num_k = 2;
      return true;
    default:
      return false;
  }
}

// Pixel residuals of projecting the observed ray: r_i = project(params,
// cam_ray_i) - img_point_i. Templated on the camera model (cf. the reprojection
// error costs), enabling static Ceres autodiff without runtime model dispatch.
// Autodiff is possible because projection (as opposed to unprojection) is
// templated for Jets in every model.
template <typename CameraModel>
struct RayReprojectionResidual {
  RayReprojectionResidual(const Eigen::Vector2d& img_point,
                          const Eigen::Vector3d& cam_ray)
      : img_point(img_point), cam_ray(cam_ray) {}

  template <typename T>
  bool operator()(const T* params, T* residuals) const {
    T x, y;
    if (!CameraModel::ImgFromCam(params,
                                 T(cam_ray.x()),
                                 T(cam_ray.y()),
                                 T(cam_ray.z()),
                                 &x,
                                 &y,
                                 /*check_cheirality=*/true)) {
      // Projection validity can depend on the optimized parameters (e.g. for
      // DIVISION and EUCM). Reject an invalid trial step rather than making
      // its residual disappear and thereby rewarding invalid parameters.
      return false;
    }
    residuals[0] = x - T(img_point.x());
    residuals[1] = y - T(img_point.y());
    return true;
  }

  Eigen::Vector2d img_point;
  Eigen::Vector3d cam_ray;
};

// Closed-form initialization of `model_id` parameters: radial fit for radial
// models, pinhole fit otherwise (with pinhole fallback if radial fails).
// Extra parameters start at zero (undistorted), except radial k1[, k2] and
// the FOV/EUCM inits below.
bool InitializeCameraParams(CameraModelId model_id,
                            const std::vector<Eigen::Vector2d>& img_points,
                            const std::vector<Eigen::Vector3d>& cam_rays,
                            double max_fov_deg,
                            std::vector<double>* params) {
  THROW_CHECK_NOTNULL(params);
  const span<const size_t> focal_idxs = CameraModelFocalLengthIdxs(model_id);
  const span<const size_t> pp_idxs = CameraModelPrincipalPointIdxs(model_id);
  if (focal_idxs.size() > 2 || pp_idxs.size() != 2) {
    LOG(ERROR) << "Unsupported parameter layout for "
               << CameraModelIdToName(model_id);
    return false;
  }

  double fx = 0, fy = 0, cx = 0, cy = 0;
  Eigen::VectorXd k_init;
  int num_k = 0;
  bool use_radial_init = HasRadialDistortionPrefix(model_id, &num_k);
  if (use_radial_init) {
    const RadialFit fit =
        FitRadialLinear(img_points, cam_rays, num_k, max_fov_deg);
    if (fit.success) {
      fx = fit.fx;
      fy = fit.fy;
      cx = fit.cx;
      cy = fit.cy;
      k_init = fit.k;
    } else {
      use_radial_init = false;
    }
  }
  if (!use_radial_init) {
    const PinholeFit fit = FitPinholeLinear(img_points, cam_rays, max_fov_deg);
    if (!fit.success) {
      return false;
    }
    fx = fit.fx;
    fy = fit.fy;
    cx = fit.cx;
    cy = fit.cy;
  }

  params->assign(CameraModelNumParams(model_id), 0.0);
  if (focal_idxs.size() == 1) {
    (*params)[focal_idxs[0]] = 0.5 * (fx + fy);
  } else {
    (*params)[focal_idxs[0]] = fx;
    (*params)[focal_idxs[1]] = fy;
  }
  (*params)[pp_idxs[0]] = cx;
  (*params)[pp_idxs[1]] = cy;
  const span<const size_t> extra_idxs = CameraModelExtraParamsIdxs(model_id);
  if (use_radial_init) {
    for (int j = 0; j < num_k; ++j) {
      (*params)[extra_idxs[j]] = k_init[j];
    }
  }
  // Non-zero inits where zero is a stationary point of the cost: FOV's
  // distortion factor depends on omega^2 to first order, and EUCM's
  // denominator has vanishing alpha/beta derivatives at the origin.
  if (model_id == CameraModelId::kFOV) {
    (*params)[extra_idxs[0]] = 0.5;
  } else if (model_id == CameraModelId::kEUCM) {
    (*params)[extra_idxs[0]] = 0.5;
    (*params)[extra_idxs[1]] = 1.0;
  }
  return true;
}

// Nonlinear refinement of `params` (in: initialization, out: refined
// parameters, or the initialization if refinement failed). Only points
// projectable at the initialization become residual blocks; the set stays
// fixed during optimization, as Ceres requires a static problem structure.
// Returns true unless refinement failed or made the cost worse.
template <typename CameraModel>
bool RefineCameraParams(const std::vector<Eigen::Vector2d>& img_points,
                        const std::vector<Eigen::Vector3d>& cam_rays,
                        const RayFittingOptions& options,
                        std::vector<double>* params,
                        double* initial_cost,
                        double* final_cost) {
  THROW_CHECK_NOTNULL(params);
  THROW_CHECK_NOTNULL(initial_cost);
  THROW_CHECK_NOTNULL(final_cost);
  THROW_CHECK_EQ(params->size(), CameraModel::num_params);
  const std::vector<double> init_params = *params;

  ceres::Problem problem;
  size_t num_residuals = 0;
  std::vector<size_t> residual_indices;
  residual_indices.reserve(img_points.size());
  for (size_t i = 0; i < img_points.size(); ++i) {
    Eigen::Vector2d projection;
    if (!CameraModel::ImgFromCam(params->data(),
                                 cam_rays[i].x(),
                                 cam_rays[i].y(),
                                 cam_rays[i].z(),
                                 &projection.x(),
                                 &projection.y(),
                                 /*check_cheirality=*/true)) {
      continue;
    }
    auto* cost_function =
        new ceres::AutoDiffCostFunction<RayReprojectionResidual<CameraModel>,
                                        2,
                                        CameraModel::num_params>(
            new RayReprojectionResidual<CameraModel>(img_points[i],
                                                     cam_rays[i]));
    problem.AddResidualBlock(
        cost_function, /*loss_function=*/nullptr, params->data());
    residual_indices.push_back(i);
    num_residuals += 2;
  }
  if (num_residuals == 0) {
    return false;
  }

  for (const size_t idx : CameraModel::focal_length_idxs) {
    problem.SetParameterLowerBound(
        params->data(), idx, std::numeric_limits<double>::epsilon());
  }
  if constexpr (CameraModel::model_id == CameraModelId::kEUCM) {
    const size_t alpha_idx = CameraModel::extra_params_idxs[0];
    const size_t beta_idx = CameraModel::extra_params_idxs[1];
    problem.SetParameterLowerBound(params->data(), alpha_idx, 0.0);
    problem.SetParameterUpperBound(params->data(), alpha_idx, 1.0);
    problem.SetParameterLowerBound(
        params->data(), beta_idx, std::numeric_limits<double>::epsilon());
  }

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.max_num_iterations = options.max_num_iterations;
  // Parallelism happens across images in the calling controller.
  solver_options.num_threads = 1;
  solver_options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  // Report mean squared pixel residuals (Ceres costs are halved sums of
  // squares).
  *initial_cost = 2 * summary.initial_cost / num_residuals;
  *final_cost = 2 * summary.final_cost / num_residuals;
  // Keep the refinement unless it made the cost worse. Ceres only accepts
  // non-increasing steps, so equality means the initialization was already
  // optimal (or refinement was disabled with zero iterations).
  if (summary.IsSolutionUsable() && std::isfinite(*final_cost) &&
      *final_cost <= *initial_cost &&
      std::all_of(params->begin(), params->end(), [](const double param) {
        return std::isfinite(param);
      })) {
    for (const size_t i : residual_indices) {
      Eigen::Vector2d projection;
      if (!CameraModel::ImgFromCam(params->data(),
                                   cam_rays[i].x(),
                                   cam_rays[i].y(),
                                   cam_rays[i].z(),
                                   &projection.x(),
                                   &projection.y(),
                                   /*check_cheirality=*/true) ||
          !projection.allFinite()) {
        *params = init_params;
        return false;
      }
    }
    return true;
  }
  *params = init_params;
  return false;
}

}  // namespace

bool RayFittingOptions::Check() const {
  CHECK_OPTION_GE(max_num_iterations, 0);
  CHECK_OPTION_GT(max_num_points, 0);
  CHECK_OPTION_GT(max_fov_deg, 0.0);
  CHECK_OPTION_LT(max_fov_deg, 180.0);
  return true;
}

std::vector<double> ReverseScaleAndShiftParams(
    CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector2d& scale_xy,
    const Eigen::Vector2d& shift_xy) {
  THROW_CHECK(CameraModelVerifyParams(model_id, params));
  std::vector<double> out = params;
  const span<const size_t> focal_idxs = CameraModelFocalLengthIdxs(model_id);
  const span<const size_t> pp_idxs = CameraModelPrincipalPointIdxs(model_id);
  THROW_CHECK_EQ(pp_idxs.size(), 2);
  if (focal_idxs.size() == 1) {
    out[focal_idxs[0]] /= 0.5 * (scale_xy.x() + scale_xy.y());
  } else {
    THROW_CHECK_EQ(focal_idxs.size(), 2);
    out[focal_idxs[0]] /= scale_xy.x();
    out[focal_idxs[1]] /= scale_xy.y();
  }
  out[pp_idxs[0]] = (out[pp_idxs[0]] - shift_xy.x()) / scale_xy.x();
  out[pp_idxs[1]] = (out[pp_idxs[1]] - shift_xy.y()) / scale_xy.y();
  return out;
}

std::vector<size_t> StrideSubsampleIndices(size_t num_points,
                                           size_t max_num_points) {
  if (num_points <= max_num_points) {
    std::vector<size_t> indices(num_points);
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
  }
  const size_t step =
      (num_points + max_num_points - 1) / max_num_points;  // ceil div
  std::vector<size_t> indices;
  indices.reserve((num_points + step - 1) / step);
  for (size_t i = 0; i < num_points; i += step) {
    indices.push_back(i);
  }
  return indices;
}

FittedCamera FitCameraFromRays(CameraModelId model_id,
                               const std::vector<Eigen::Vector2d>& img_points,
                               const std::vector<Eigen::Vector3d>& cam_rays,
                               const RayFittingOptions& options) {
  FittedCamera result;
  THROW_CHECK(options.Check());
  if (img_points.empty() || img_points.size() != cam_rays.size()) {
    return result;
  }
  if (!CameraModelIsPerspective(model_id)) {
    LOG(ERROR) << "Ray fitting only supports perspective camera models";
    return result;
  }

  // Stride-subsample dense correspondences.
  const std::vector<size_t> indices =
      StrideSubsampleIndices(img_points.size(), options.max_num_points);
  std::vector<Eigen::Vector2d> sampled_img_points;
  std::vector<Eigen::Vector3d> sampled_cam_rays;
  sampled_img_points.reserve(indices.size());
  sampled_cam_rays.reserve(indices.size());
  for (const size_t i : indices) {
    sampled_img_points.push_back(img_points[i]);
    sampled_cam_rays.push_back(cam_rays[i]);
  }

  std::vector<double> params;
  if (!InitializeCameraParams(model_id,
                              sampled_img_points,
                              sampled_cam_rays,
                              options.max_fov_deg,
                              &params)) {
    return result;
  }

  bool refined = false;
  switch (model_id) {
#define CAMERA_MODEL_CASE(Model)                              \
  case Model::model_id:                                       \
    refined = RefineCameraParams<Model>(sampled_img_points,   \
                                        sampled_cam_rays,     \
                                        options,              \
                                        &params,              \
                                        &result.initial_cost, \
                                        &result.final_cost);  \
    break;
    PERSPECTIVE_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
    default:
      LOG(ERROR) << "Unsupported camera model for refinement: "
                 << CameraModelIdToName(model_id);
      return result;
  }

  result.params = params;
  result.success = refined;
  return result;
}

}  // namespace colmap
