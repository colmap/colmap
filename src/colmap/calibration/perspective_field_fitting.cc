// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/calibration/perspective_field_fitting.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/math/math.h"
#include "colmap/sensor/models.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace colmap {

bool PerspectiveField::Check() const {
  const Eigen::Index num_points = points2D_in_img.rows();
  if (up_in_img.rows() != num_points || latitude.size() != num_points) {
    return false;
  }
  if (up_confidence.size() != 0 && up_confidence.size() != num_points) {
    return false;
  }
  if (latitude_confidence.size() != 0 &&
      latitude_confidence.size() != num_points) {
    return false;
  }
  if ((width > 0 || height > 0) &&
      static_cast<Eigen::Index>(width) * height != num_points) {
    return false;
  }
  return true;
}

bool PerspectiveFieldFittingOptions::Check() const {
  CHECK_OPTION_GT(max_num_iterations, 0);
  CHECK_OPTION_GT(stride, 0);
  CHECK_OPTION_GE(min_confidence, 0.0);
  return true;
}

PerspectiveField SubsamplePerspectiveField(const PerspectiveField& field,
                                           const int stride,
                                           const bool max_pool_confidence,
                                           const double min_confidence) {
  THROW_CHECK(field.Check());
  THROW_CHECK_GT(stride, 0);

  const Eigen::Index num_points = field.NumPoints();
  const bool has_up_conf = field.up_confidence.size() == num_points;
  const bool has_lat_conf = field.latitude_confidence.size() == num_points;

  auto combined_conf = [&](Eigen::Index idx) -> double {
    const double c_up = has_up_conf ? field.up_confidence(idx) : 1.0;
    const double c_lat = has_lat_conf ? field.latitude_confidence(idx) : 1.0;
    return std::sqrt(std::max(0.0, c_up) * std::max(0.0, c_lat));
  };

  if (stride == 1 && min_confidence <= 0.0) {
    return field;
  }

  std::vector<Eigen::Index> selected_indices;
  if (field.width > 0 && field.height > 0) {
    const int H = field.height;
    const int W = field.width;
    selected_indices.reserve(((H + stride - 1) / stride) *
                             ((W + stride - 1) / stride));
    for (int r0 = 0; r0 < H; r0 += stride) {
      const int r1 = std::min(r0 + stride, H);
      for (int c0 = 0; c0 < W; c0 += stride) {
        const int c1 = std::min(c0 + stride, W);
        if (!max_pool_confidence) {
          const int r = r0 + (r1 - r0) / 2;
          const int c = c0 + (c1 - c0) / 2;
          const Eigen::Index idx = static_cast<Eigen::Index>(r) * W + c;
          if (combined_conf(idx) >= min_confidence) {
            selected_indices.push_back(idx);
          }
          continue;
        }
        Eigen::Index best_idx = -1;
        double best_score = -1.0;
        for (int r = r0; r < r1; ++r) {
          for (int c = c0; c < c1; ++c) {
            const Eigen::Index idx = static_cast<Eigen::Index>(r) * W + c;
            const double score = combined_conf(idx);
            if (score >= min_confidence && score > best_score) {
              best_score = score;
              best_idx = idx;
            }
          }
        }
        if (best_idx >= 0) {
          selected_indices.push_back(best_idx);
        }
      }
    }
  } else {
    selected_indices.reserve((num_points + stride - 1) / stride);
    for (Eigen::Index idx = 0; idx < num_points; idx += stride) {
      if (combined_conf(idx) >= min_confidence) {
        selected_indices.push_back(idx);
      }
    }
  }

  const Eigen::Index out_n = static_cast<Eigen::Index>(selected_indices.size());
  PerspectiveField out;
  out.width = 0;
  out.height = 0;
  out.points2D_in_img.resize(out_n, 2);
  out.up_in_img.resize(out_n, 2);
  out.latitude.resize(out_n);
  if (has_up_conf) {
    out.up_confidence.resize(out_n);
  }
  if (has_lat_conf) {
    out.latitude_confidence.resize(out_n);
  }

  for (Eigen::Index i = 0; i < out_n; ++i) {
    const Eigen::Index src_idx = selected_indices[i];
    out.points2D_in_img.row(i) = field.points2D_in_img.row(src_idx);
    out.up_in_img.row(i) = field.up_in_img.row(src_idx);
    out.latitude(i) = field.latitude(src_idx);
    if (has_up_conf) {
      out.up_confidence(i) = field.up_confidence(src_idx);
    }
    if (has_lat_conf) {
      out.latitude_confidence(i) = field.latitude_confidence(src_idx);
    }
  }
  return out;
}

namespace {

// Initialize uncalibrated focal length to 0.7 * max(width, height) and zero
// distortion, mirroring GeoCalib's get_trivial_estimation in lm_optimizer.py.
void InitializeFocalLengthFromPerspectiveField(Camera* camera) {
  if (camera->has_prior_focal_length ||
      CameraModelIsSpherical(camera->model_id)) {
    return;
  }
  const double max_dim =
      static_cast<double>(std::max(camera->width, camera->height));
  if (max_dim <= 0.0) {
    return;
  }
  const double f_init = 0.7 * max_dim;
  for (const size_t idx : camera->FocalLengthIdxs()) {
    camera->params[idx] = f_init;
  }
}

// Closed-form linear least-squares initialization of gravity_in_rig on S^2
// from all subsampled perspective fields across the frame.
Eigen::Vector3d EstimateInitialGravityInRig(
    const std::vector<PerspectiveFieldCameraInput>& inputs,
    const std::vector<PerspectiveField>& subsampled_fields,
    const bool use_confidence) {
  Eigen::Matrix3d AtA = Eigen::Matrix3d::Zero();
  Eigen::Vector3d Atb = Eigen::Vector3d::Zero();

  for (size_t k = 0; k < inputs.size(); ++k) {
    const Camera& camera = *inputs[k].camera;
    const PerspectiveField& field = subsampled_fields[k];
    const Eigen::Matrix3d R_cam_from_rig =
        inputs[k].cam_from_rig.toRotationMatrix();
    const Eigen::Index num_points = field.NumPoints();
    const bool has_up_conf =
        use_confidence && field.up_confidence.size() == num_points;
    const bool has_lat_conf =
        use_confidence && field.latitude_confidence.size() == num_points;

    for (Eigen::Index i = 0; i < num_points; ++i) {
      const Eigen::Vector2d xy = field.points2D_in_img.row(i).transpose();
      const std::optional<Eigen::Vector3d> r_cam =
          CameraModelCamRayFromImg(camera.model_id, camera.params, xy);
      if (!r_cam.has_value()) {
        continue;
      }

      // 1. Latitude linear constraint:
      //   (R_cam_from_rig^T * r_cam)^T * g_rig = -sin(lat_obs)
      const double w_lat =
          has_lat_conf ? std::max(0.0, field.latitude_confidence(i)) : 1.0;
      if (w_lat > 0.0) {
        const Eigen::Vector3d a_lat = R_cam_from_rig.transpose() * (*r_cam);
        const double b_lat = -std::sin(field.latitude(i));
        AtA += w_lat * (a_lat * a_lat.transpose());
        Atb += w_lat * b_lat * a_lat;
      }

      // 2. Up-vector orthogonal constraint:
      //   u_tilde = -J_uvw * g_cam is parallel to up_obs, so
      //   [-up_obs.y, up_obs.x] * J_uvw * R_cam_from_rig * g_rig = 0.
      const double w_up =
          has_up_conf ? std::max(0.0, field.up_confidence(i)) : 1.0;
      if (w_up > 0.0) {
        Eigen::Matrix2x3d J_uvw;
        if (CameraModelImgFromCamWithJac(camera.model_id,
                                         camera.params,
                                         *r_cam,
                                         &J_uvw,
                                         /*check_cheirality=*/false)
                .has_value()) {
          const Eigen::Vector2d up = field.up_in_img.row(i).transpose();
          const Eigen::Vector2d up_perp(-up.y(), up.x());
          Eigen::Vector3d a_up =
              (up_perp.transpose() * J_uvw * R_cam_from_rig).transpose();
          const double a_up_norm = a_up.norm();
          if (a_up_norm > 1e-12) {
            a_up /= a_up_norm;
            AtA += w_up * (a_up * a_up.transpose());
          }
        }
      }
    }
  }

  Eigen::Vector3d g_rig(0.0, 1.0, 0.0);
  if (Atb.squaredNorm() > 1e-12) {
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(AtA);
    if (eig.info() == Eigen::Success && eig.eigenvalues()(0) > 1e-8) {
      const Eigen::Vector3d g_sol = AtA.ldlt().solve(Atb);
      if (g_sol.norm() > 1e-6 && g_sol.allFinite()) {
        g_rig = g_sol.normalized();
      }
    }
  } else if (AtA.norm() > 1e-12) {
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(AtA);
    if (eig.info() == Eigen::Success) {
      g_rig = eig.eigenvectors().col(0);
    }
  }

  // Ensure sign consistency with observed 2D up-vectors.
  double up_dot_sum = 0.0;
  for (size_t k = 0; k < inputs.size(); ++k) {
    const Camera& camera = *inputs[k].camera;
    const PerspectiveField& field = subsampled_fields[k];
    const Eigen::Vector3d g_cam = inputs[k].cam_from_rig * g_rig;
    const Eigen::Index num_points = field.NumPoints();
    const bool has_up_conf =
        use_confidence && field.up_confidence.size() == num_points;

    for (Eigen::Index i = 0; i < num_points; ++i) {
      const double w_up =
          has_up_conf ? std::max(0.0, field.up_confidence(i)) : 1.0;
      if (w_up <= 0.0) {
        continue;
      }
      const Eigen::Vector2d xy = field.points2D_in_img.row(i).transpose();
      const std::optional<Eigen::Vector3d> r_cam =
          CameraModelCamRayFromImg(camera.model_id, camera.params, xy);
      if (!r_cam.has_value()) {
        continue;
      }
      Eigen::Matrix2x3d J_uvw;
      if (CameraModelImgFromCamWithJac(camera.model_id,
                                       camera.params,
                                       *r_cam,
                                       &J_uvw,
                                       /*check_cheirality=*/false)
              .has_value()) {
        const Eigen::Vector2d u_tilde = -(J_uvw * g_cam);
        const double norm = u_tilde.norm();
        if (norm > 1e-12) {
          const Eigen::Vector2d up = field.up_in_img.row(i).transpose();
          up_dot_sum += std::sqrt(w_up) * up.dot(u_tilde / norm);
        }
      }
    }
  }
  if (up_dot_sum < 0.0) {
    g_rig = -g_rig;
  }

  return g_rig;
}

template <typename T>
inline void ApplyScaledHuber2D(const double weight,
                               const double scale,
                               T* r0,
                               T* r1) {
  if (scale > 0.0) {
    const T s2 = (*r0) * (*r0) + (*r1) * (*r1);
    const T a = T(scale);
    const T a2 = a * a;
    if (s2 > a2) {
      const T s = ceres::sqrt(s2);
      const T factor = ceres::sqrt(T(2.0) * a * s - a2) / s;
      *r0 *= factor;
      *r1 *= factor;
    }
  }
  *r0 *= T(weight);
  *r1 *= T(weight);
}

template <typename T>
inline T ApplyScaledHuber1D(const double weight,
                            const double scale,
                            const T& r) {
  T out = r;
  if (scale > 0.0) {
    const T s2 = r * r;
    const T a = T(scale);
    const T a2 = a * a;
    if (s2 > a2) {
      const T s = ceres::sqrt(s2);
      out *= ceres::sqrt(T(2.0) * a * s - a2) / s;
    }
  }
  return T(weight) * out;
}

// Fast residual evaluation when camera intrinsics are fixed: r and J_uvw are
// precomputed once in double before Ceres runs. Works for all 18 camera models.
struct FixedCameraPerspectiveFieldCostFunctor {
  FixedCameraPerspectiveFieldCostFunctor(const Eigen::Vector3d& r,
                                         const Eigen::Matrix2x3d& J_uvw,
                                         const Eigen::Quaterniond& cam_from_rig,
                                         const Eigen::Vector2d& up_obs,
                                         const double up_weight,
                                         const double sin_lat_obs,
                                         const double lat_weight,
                                         const double loss_scale)
      : r_(r),
        J_uvw_(J_uvw),
        cam_from_rig_(cam_from_rig),
        up_obs_(up_obs),
        up_weight_(up_weight),
        sin_lat_obs_(sin_lat_obs),
        lat_weight_(lat_weight),
        loss_scale_(loss_scale) {}

  static ceres::CostFunction* Create(const Eigen::Vector3d& r,
                                     const Eigen::Matrix2x3d& J_uvw,
                                     const Eigen::Quaterniond& cam_from_rig,
                                     const Eigen::Vector2d& up_obs,
                                     const double up_weight,
                                     const double sin_lat_obs,
                                     const double lat_weight,
                                     const double loss_scale) {
    return new ceres::
        AutoDiffCostFunction<FixedCameraPerspectiveFieldCostFunctor, 3, 3>(
            new FixedCameraPerspectiveFieldCostFunctor(r,
                                                       J_uvw,
                                                       cam_from_rig,
                                                       up_obs,
                                                       up_weight,
                                                       sin_lat_obs,
                                                       lat_weight,
                                                       loss_scale));
  }

  template <typename T>
  bool operator()(const T* const gravity_in_rig_ptr, T* residuals) const {
    const Eigen::Map<const Eigen::Vector3<T>> g_in_rig(gravity_in_rig_ptr);
    const Eigen::Vector3<T> g_in_cam = cam_from_rig_.cast<T>() * g_in_rig;

    const Eigen::Vector2<T> u_tilde = -(J_uvw_.cast<T>() * g_in_cam);
    const T u_norm = ceres::sqrt(u_tilde.squaredNorm() + T(1e-12));
    const Eigen::Vector2<T> u_2d = u_tilde / u_norm;

    residuals[0] = T(up_obs_.x()) - u_2d.x();
    residuals[1] = T(up_obs_.y()) - u_2d.y();
    ApplyScaledHuber2D(up_weight_, loss_scale_, &residuals[0], &residuals[1]);

    const T sin_lat = -r_.cast<T>().dot(g_in_cam);
    residuals[2] =
        ApplyScaledHuber1D(lat_weight_, loss_scale_, T(sin_lat_obs_) - sin_lat);
    return true;
  }

 private:
  const Eigen::Vector3d r_;
  const Eigen::Matrix2x3d J_uvw_;
  const Eigen::Quaterniond cam_from_rig_;
  const Eigen::Vector2d up_obs_;
  const double up_weight_;
  const double sin_lat_obs_;
  const double lat_weight_;
  const double loss_scale_;
};

// Cost functor when camera intrinsics ARE refined: uses templated
// CamRayFromImg<T> and ImgFromCamWithJac<T> across all camera models.
template <typename CameraModel>
struct RefinedCameraPerspectiveFieldCostFunctor {
  RefinedCameraPerspectiveFieldCostFunctor(
      const Eigen::Vector2d& point2D,
      const Eigen::Quaterniond& cam_from_rig,
      const Eigen::Vector2d& up_obs,
      const double up_weight,
      const double sin_lat_obs,
      const double lat_weight,
      const double loss_scale)
      : point2D_(point2D),
        cam_from_rig_(cam_from_rig),
        up_obs_(up_obs),
        up_weight_(up_weight),
        sin_lat_obs_(sin_lat_obs),
        lat_weight_(lat_weight),
        loss_scale_(loss_scale) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D,
                                     const Eigen::Quaterniond& cam_from_rig,
                                     const Eigen::Vector2d& up_obs,
                                     const double up_weight,
                                     const double sin_lat_obs,
                                     const double lat_weight,
                                     const double loss_scale) {
    return new ceres::AutoDiffCostFunction<
        RefinedCameraPerspectiveFieldCostFunctor<CameraModel>,
        3,
        CameraModel::num_params,
        3>(
        new RefinedCameraPerspectiveFieldCostFunctor<CameraModel>(point2D,
                                                                  cam_from_rig,
                                                                  up_obs,
                                                                  up_weight,
                                                                  sin_lat_obs,
                                                                  lat_weight,
                                                                  loss_scale));
  }

  template <typename T>
  bool operator()(const T* const params,
                  const T* const gravity_in_rig_ptr,
                  T* residuals) const {
    T params_eval[CameraModel::num_params];
    for (size_t i = 0; i < CameraModel::num_params; ++i) {
      params_eval[i] = params[i];
    }
    if constexpr (CameraModel::focal_length_idxs.size() == 2) {
      params_eval[CameraModel::focal_length_idxs[1]] =
          params_eval[CameraModel::focal_length_idxs[0]];
    }

    Eigen::Vector3<T> r;
    if (!CameraModel::CamRayFromImg(params_eval,
                                    T(point2D_.x()),
                                    T(point2D_.y()),
                                    &r.x(),
                                    &r.y(),
                                    &r.z())) {
      return false;
    }

    const Eigen::Map<const Eigen::Vector3<T>> g_in_rig(gravity_in_rig_ptr);
    const Eigen::Vector3<T> g_in_cam = cam_from_rig_.cast<T>() * g_in_rig;

    // Directional derivative of ImgFromCam(params, r - t * g_in_cam) at t = 0:
    //   u_tilde = -J_uvw(params, r) * g_in_cam.
    // Evaluating ImgFromCam<T> at r +/- eps * g_in_cam propagates exact Ceres
    // Jet derivatives w.r.t. both camera parameters and gravity across all
    // camera models without requiring templated analytical Jacobians.
    constexpr double kStep = 1e-6;
    T x_plus, y_plus, x_minus, y_minus;
    if (!CameraModel::ImgFromCam(params_eval,
                                 r.x() - T(kStep) * g_in_cam.x(),
                                 r.y() - T(kStep) * g_in_cam.y(),
                                 r.z() - T(kStep) * g_in_cam.z(),
                                 &x_plus,
                                 &y_plus,
                                 /*check_cheirality=*/false) ||
        !CameraModel::ImgFromCam(params_eval,
                                 r.x() + T(kStep) * g_in_cam.x(),
                                 r.y() + T(kStep) * g_in_cam.y(),
                                 r.z() + T(kStep) * g_in_cam.z(),
                                 &x_minus,
                                 &y_minus,
                                 /*check_cheirality=*/false)) {
      return false;
    }

    const Eigen::Vector2<T> u_tilde((x_plus - x_minus) / T(2.0 * kStep),
                                    (y_plus - y_minus) / T(2.0 * kStep));
    const T u_norm = ceres::sqrt(u_tilde.squaredNorm() + T(1e-12));
    const Eigen::Vector2<T> u_2d = u_tilde / u_norm;

    residuals[0] = T(up_obs_.x()) - u_2d.x();
    residuals[1] = T(up_obs_.y()) - u_2d.y();
    ApplyScaledHuber2D(up_weight_, loss_scale_, &residuals[0], &residuals[1]);

    const T sin_lat = -r.dot(g_in_cam);
    residuals[2] =
        ApplyScaledHuber1D(lat_weight_, loss_scale_, T(sin_lat_obs_) - sin_lat);
    return true;
  }

 private:
  const Eigen::Vector2d point2D_;
  const Eigen::Quaterniond cam_from_rig_;
  const Eigen::Vector2d up_obs_;
  const double up_weight_;
  const double sin_lat_obs_;
  const double lat_weight_;
  const double loss_scale_;
};

void SetCameraParamsManifold(const PerspectiveFieldFittingOptions& options,
                             const Camera& camera,
                             ceres::Problem* problem,
                             double* params_data) {
  if (!problem->HasParameterBlock(params_data)) {
    return;
  }

  if (CameraModelIsSpherical(camera.model_id)) {
    problem->SetParameterBlockConstant(params_data);
    return;
  }

  std::vector<int> const_camera_params;
  const span<const size_t> focal_idxs = camera.FocalLengthIdxs();
  if (!options.refine_focal_length) {
    const_camera_params.insert(
        const_camera_params.end(), focal_idxs.begin(), focal_idxs.end());
  } else if (focal_idxs.size() == 2) {
    // A perspective field is rotationally symmetric around the gravity axis,
    // so only an isotropic focal length (fx = fy) is observable.
    const_camera_params.push_back(static_cast<int>(focal_idxs[1]));
  }
  if (!options.refine_principal_point) {
    const span<const size_t> idxs = camera.PrincipalPointIdxs();
    const_camera_params.insert(
        const_camera_params.end(), idxs.begin(), idxs.end());
  }
  const span<const size_t> extra_idxs = camera.ExtraParamsIdxs();
  if (!options.refine_extra_params) {
    const_camera_params.insert(
        const_camera_params.end(), extra_idxs.begin(), extra_idxs.end());
  } else if (extra_idxs.size() > 1) {
    // Refine only the primary radial distortion parameter (k1 / omega / alpha)
    // to prevent overfitting higher-order / tangential terms to a 1D latitude
    // profile.
    const_camera_params.insert(
        const_camera_params.end(), extra_idxs.begin() + 1, extra_idxs.end());
  }

  const size_t num_params = camera.params.size();
  if (const_camera_params.size() == num_params) {
    problem->SetParameterBlockConstant(params_data);
  } else if (!const_camera_params.empty()) {
    SetManifold(problem,
                params_data,
                CreateSubsetManifold(static_cast<int>(num_params),
                                     const_camera_params));
  }

  if (options.refine_focal_length && !focal_idxs.empty()) {
    problem->SetParameterLowerBound(
        params_data, static_cast<int>(focal_idxs[0]), 1e-3);
  }
}

void AddFixedCameraResiduals(const Camera& camera,
                             const PerspectiveField& field,
                             const Eigen::Quaterniond& cam_from_rig,
                             const bool use_confidence,
                             const double loss_scale,
                             double* gravity_in_rig_data,
                             ceres::Problem* problem) {
  const Eigen::Index num_points = field.NumPoints();
  const bool has_up_conf =
      use_confidence && field.up_confidence.size() == num_points;
  const bool has_lat_conf =
      use_confidence && field.latitude_confidence.size() == num_points;

  for (Eigen::Index i = 0; i < num_points; ++i) {
    const Eigen::Vector2d xy = field.points2D_in_img.row(i).transpose();
    const std::optional<Eigen::Vector3d> r =
        CameraModelCamRayFromImg(camera.model_id, camera.params, xy);
    if (!r.has_value()) {
      continue;
    }
    Eigen::Matrix2x3d J_uvw;
    if (!CameraModelImgFromCamWithJac(camera.model_id,
                                      camera.params,
                                      *r,
                                      &J_uvw,
                                      /*check_cheirality=*/false)
             .has_value()) {
      continue;
    }
    const Eigen::Vector2d up = field.up_in_img.row(i).transpose();
    const double up_w =
        has_up_conf ? std::sqrt(std::max(0.0, field.up_confidence(i))) : 1.0;
    const double sin_lat = std::sin(field.latitude(i));
    const double lat_w =
        has_lat_conf ? std::sqrt(std::max(0.0, field.latitude_confidence(i)))
                     : 1.0;

    problem->AddResidualBlock(
        FixedCameraPerspectiveFieldCostFunctor::Create(
            *r, J_uvw, cam_from_rig, up, up_w, sin_lat, lat_w, loss_scale),
        /*loss_function=*/nullptr,
        gravity_in_rig_data);
  }
}

void AddRefinedCameraResiduals(const PerspectiveField& field,
                               const Eigen::Quaterniond& cam_from_rig,
                               const bool use_confidence,
                               const double loss_scale,
                               Camera* camera,
                               double* gravity_in_rig_data,
                               ceres::Problem* problem) {
  const Eigen::Index num_points = field.NumPoints();
  const bool has_up_conf =
      use_confidence && field.up_confidence.size() == num_points;
  const bool has_lat_conf =
      use_confidence && field.latitude_confidence.size() == num_points;

  switch (camera->model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                     \
  case CameraModel::model_id:                                              \
    for (Eigen::Index i = 0; i < num_points; ++i) {                        \
      const Eigen::Vector2d xy = field.points2D_in_img.row(i).transpose(); \
      const Eigen::Vector2d up = field.up_in_img.row(i).transpose();       \
      const double up_w =                                                  \
          has_up_conf ? std::sqrt(std::max(0.0, field.up_confidence(i)))   \
                      : 1.0;                                               \
      const double sin_lat = std::sin(field.latitude(i));                  \
      const double lat_w =                                                 \
          has_lat_conf                                                     \
              ? std::sqrt(std::max(0.0, field.latitude_confidence(i)))     \
              : 1.0;                                                       \
      problem->AddResidualBlock(                                           \
          RefinedCameraPerspectiveFieldCostFunctor<CameraModel>::Create(   \
              xy, cam_from_rig, up, up_w, sin_lat, lat_w, loss_scale),     \
          /*loss_function=*/nullptr,                                       \
          camera->params.data(),                                           \
          gravity_in_rig_data);                                            \
    }                                                                      \
    break;

    PERSPECTIVE_CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
    default:
      LOG(FATAL_THROW) << "Unsupported camera model for intrinsic refinement: "
                       << CameraModelIdToName(camera->model_id);
  }
}

}  // namespace

FittedPerspectiveFields FitPerspectiveFields(
    const PerspectiveFieldFittingOptions& options,
    const std::vector<PerspectiveFieldCameraInput>& inputs) {
  THROW_CHECK(options.Check());
  THROW_CHECK(!inputs.empty());

  std::vector<PerspectiveField> subsampled_fields(inputs.size());
  size_t total_points = 0;
  for (size_t k = 0; k < inputs.size(); ++k) {
    THROW_CHECK_NOTNULL(inputs[k].field);
    THROW_CHECK_NOTNULL(inputs[k].camera);
    THROW_CHECK(inputs[k].field->Check());
    THROW_CHECK(inputs[k].camera->VerifyParams());

    subsampled_fields[k] =
        SubsamplePerspectiveField(*inputs[k].field,
                                  options.stride,
                                  options.max_pool_confidence,
                                  options.min_confidence);
    total_points += subsampled_fields[k].NumPoints();

    if (inputs[k].refine_camera && options.refine_focal_length &&
        !inputs[k].camera->has_prior_focal_length) {
      InitializeFocalLengthFromPerspectiveField(inputs[k].camera);
    }
  }

  FittedPerspectiveFields result;
  if (total_points == 0) {
    return result;
  }

  result.gravity_in_rig = EstimateInitialGravityInRig(
      inputs, subsampled_fields, options.use_confidence);

  ceres::Problem problem;

  const bool any_intrinsic_refined = options.refine_focal_length ||
                                     options.refine_principal_point ||
                                     options.refine_extra_params;

  for (size_t k = 0; k < inputs.size(); ++k) {
    if (subsampled_fields[k].NumPoints() == 0) {
      continue;
    }
    const bool refine_this_camera =
        inputs[k].refine_camera && any_intrinsic_refined &&
        !CameraModelIsSpherical(inputs[k].camera->model_id);

    if (refine_this_camera) {
      AddRefinedCameraResiduals(subsampled_fields[k],
                                inputs[k].cam_from_rig,
                                options.use_confidence,
                                options.loss_function_scale,
                                inputs[k].camera,
                                result.gravity_in_rig.data(),
                                &problem);
      SetCameraParamsManifold(options,
                              *inputs[k].camera,
                              &problem,
                              inputs[k].camera->params.data());
    } else {
      AddFixedCameraResiduals(*inputs[k].camera,
                              subsampled_fields[k],
                              inputs[k].cam_from_rig,
                              options.use_confidence,
                              options.loss_function_scale,
                              result.gravity_in_rig.data(),
                              &problem);
    }
  }

  if (!problem.HasParameterBlock(result.gravity_in_rig.data())) {
    return result;
  }
  SetManifold(
      &problem, result.gravity_in_rig.data(), CreateSphereManifold<3>());

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.minimizer_progress_to_stdout = options.print_summary;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  if (options.print_summary) {
    LOG(INFO) << summary.FullReport();
  }

  if (options.refine_focal_length) {
    for (size_t k = 0; k < inputs.size(); ++k) {
      if (inputs[k].refine_camera &&
          !CameraModelIsSpherical(inputs[k].camera->model_id)) {
        const span<const size_t> focal_idxs =
            inputs[k].camera->FocalLengthIdxs();
        if (focal_idxs.size() == 2) {
          inputs[k].camera->params[focal_idxs[1]] =
              inputs[k].camera->params[focal_idxs[0]];
        }
      }
    }
  }

  result.gravity_in_rig.normalize();
  result.success = summary.IsSolutionUsable();
  result.initial_cost = summary.initial_cost;
  result.final_cost = summary.final_cost;
  return result;
}

FittedPerspectiveFields FitPerspectiveField(
    const PerspectiveFieldFittingOptions& options,
    const PerspectiveField& field,
    Camera* camera,
    const bool refine_camera) {
  PerspectiveFieldCameraInput input;
  input.field = &field;
  input.camera = camera;
  input.cam_from_rig = Eigen::Quaterniond::Identity();
  input.refine_camera = refine_camera;
  return FitPerspectiveFields(options, {input});
}

PerspectiveField ComputePerspectiveFieldFromCameraAndGravity(
    const Camera& camera,
    const Eigen::Vector3d& gravity_in_cam,
    const int grid_width,
    const int grid_height) {
  THROW_CHECK(camera.VerifyParams());
  const int W = grid_width > 0 ? grid_width : static_cast<int>(camera.width);
  const int H = grid_height > 0 ? grid_height : static_cast<int>(camera.height);
  THROW_CHECK_GT(W, 0);
  THROW_CHECK_GT(H, 0);

  const Eigen::Vector3d g_in_cam = gravity_in_cam.normalized();
  const double scale_x =
      static_cast<double>(camera.width) / static_cast<double>(W);
  const double scale_y =
      static_cast<double>(camera.height) / static_cast<double>(H);

  const Eigen::Index num_pixels = static_cast<Eigen::Index>(W) * H;
  PerspectiveField field;
  field.width = W;
  field.height = H;
  field.points2D_in_img.resize(num_pixels, 2);
  field.up_in_img.resize(num_pixels, 2);
  field.latitude.resize(num_pixels);
  field.up_confidence = Eigen::VectorXd::Ones(num_pixels);
  field.latitude_confidence = Eigen::VectorXd::Ones(num_pixels);

  for (int r = 0; r < H; ++r) {
    for (int c = 0; c < W; ++c) {
      const Eigen::Index idx = static_cast<Eigen::Index>(r) * W + c;
      const Eigen::Vector2d xy((c + 0.5) * scale_x, (r + 0.5) * scale_y);
      field.points2D_in_img.row(idx) = xy.transpose();

      const std::optional<Eigen::Vector3d> ray =
          CameraModelCamRayFromImg(camera.model_id, camera.params, xy);
      Eigen::Matrix2x3d J_uvw;
      if (!ray.has_value() ||
          !CameraModelImgFromCamWithJac(camera.model_id,
                                        camera.params,
                                        *ray,
                                        &J_uvw,
                                        /*check_cheirality=*/false)
               .has_value()) {
        field.up_in_img.row(idx) = Eigen::RowVector2d(0.0, -1.0);
        field.latitude(idx) = 0.0;
        field.up_confidence(idx) = 0.0;
        field.latitude_confidence(idx) = 0.0;
        continue;
      }

      const Eigen::Vector2d u_tilde = -(J_uvw * g_in_cam);
      const double u_norm = std::sqrt(u_tilde.squaredNorm() + 1e-12);
      field.up_in_img.row(idx) = (u_tilde / u_norm).transpose();
      const double sin_lat = std::clamp(-ray->dot(g_in_cam), -1.0, 1.0);
      field.latitude(idx) = std::asin(sin_lat);
    }
  }
  return field;
}

}  // namespace colmap
