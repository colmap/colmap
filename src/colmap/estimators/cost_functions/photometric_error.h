// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/image/image_pyramid.h"
#include "colmap/sensor/models.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <ceres/ceres.h>

namespace colmap {

// Maximum number of camera parameters across all COLMAP camera models
// (RadTanThinPrismFisheye / FullOpenCV / ThinPrismFisheye have 12 params).
constexpr int kPhotometricMaxCamParams = 16;

// Maximum square patch side length P (M = P * P <= 64 samples). Bounding M at
// compile time lets Evaluate() keep all scratch matrices on the stack with
// zero heap allocations.
constexpr int kPhotometricMaxPatchSize = 8;
constexpr int kPhotometricMaxPatchSamples =
    kPhotometricMaxPatchSize * kPhotometricMaxPatchSize;

// Fixed per-landmark source patch data shared across all target views of the
// landmark at a given pyramid level.
struct PhotometricPatchData {
  // Full-resolution COLMAP pixel coordinates of the M patch samples in the
  // source image.
  Eigen::VectorXd src_xs;
  Eigen::VectorXd src_ys;

  // Precomputed zero-mean unit-norm source intensities Psi(I_src) of length M.
  Eigen::VectorXd src_norm;

  // Initial undistorted source rays (3 x M) at the start of the current pyramid
  // level (z = 1 for perspective models, unit sphere ray for spherical models).
  // Used directly when source intrinsics are constant, or as a thread-safe
  // warm-start for Newton undistortion when variable.
  Eigen::Matrix3Xd src_rays_init;

  // Snapshot of source camera parameters at the start of the current pyramid
  // level so Evaluate() can skip unprojection when intrinsics are unchanged.
  std::vector<double> src_cam_params_init;

  // Full-resolution anchor pixel (center of patch) used to reconstruct the
  // updated 3D point along the source ray after plane refinement.
  Eigen::Vector2d anchor_xy = Eigen::Vector2d::Zero();

  int NumSamples() const { return static_cast<int>(src_norm.size()); }
};

namespace photometric_internal {

// Stack-allocated dynamic-size Eigen types (zero heap allocation for
// M <= kPhotometricMaxPatchSamples).
using VectorMaxM = Eigen::Matrix<double,
                                 Eigen::Dynamic,
                                 1,
                                 Eigen::ColMajor,
                                 kPhotometricMaxPatchSamples,
                                 1>;
using Matrix3xMaxM = Eigen::Matrix<double,
                                   3,
                                   Eigen::Dynamic,
                                   Eigen::ColMajor,
                                   3,
                                   kPhotometricMaxPatchSamples>;
using MatrixMaxMx2 = Eigen::Matrix<double,
                                   Eigen::Dynamic,
                                   2,
                                   Eigen::ColMajor,
                                   kPhotometricMaxPatchSamples,
                                   2>;
using MatrixMaxMx3RowMajor = Eigen::Matrix<double,
                                           Eigen::Dynamic,
                                           3,
                                           Eigen::RowMajor,
                                           kPhotometricMaxPatchSamples,
                                           3>;
using MatrixMaxMx6RowMajor = Eigen::Matrix<double,
                                           Eigen::Dynamic,
                                           6,
                                           Eigen::RowMajor,
                                           kPhotometricMaxPatchSamples,
                                           6>;
using MatrixMaxMx7RowMajor = Eigen::Matrix<double,
                                           Eigen::Dynamic,
                                           7,
                                           Eigen::RowMajor,
                                           kPhotometricMaxPatchSamples,
                                           7>;
using MatrixMaxCamJac = Eigen::Matrix<double,
                                      Eigen::Dynamic,
                                      Eigen::Dynamic,
                                      Eigen::RowMajor,
                                      2 * kPhotometricMaxPatchSamples,
                                      kPhotometricMaxCamParams>;
using MatrixMaxScratch = Eigen::Matrix<double,
                                       Eigen::Dynamic,
                                       Eigen::Dynamic,
                                       Eigen::RowMajor,
                                       kPhotometricMaxPatchSamples,
                                       kPhotometricMaxCamParams>;

using MatrixMx3RowMajor =
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Matrix3x4RowMajor = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
using Matrix2x3RowMajor = Eigen::Matrix<double, 2, 3, Eigen::RowMajor>;
using RowVectorMaxCamParams = Eigen::Matrix<double,
                                            1,
                                            Eigen::Dynamic,
                                            Eigen::RowMajor,
                                            1,
                                            kPhotometricMaxCamParams>;

template <int kCols>
inline Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, kCols, Eigen::RowMajor>>
MapFixedColMatrix(double* ptr, const int rows) {
  return Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, kCols, Eigen::RowMajor>>(
      ptr, rows, kCols);
}

inline Eigen::Map<Eigen::RowMajorMatrixXd> MapRowMajorMatrix(double* ptr,
                                                             const int rows,
                                                             const int cols) {
  return Eigen::Map<Eigen::RowMajorMatrixXd>(ptr, rows, cols);
}

// Compute J_q = d(R(q) * p) / dq (3x4 row-major, Eigen xyzw storage).
inline Matrix3x4RowMajor QuaternionRotatePointJac(const Eigen::Quaterniond& q,
                                                  const Eigen::Vector3d& pt) {
  Matrix3x4RowMajor J;
  QuaternionRotatePointWithJac(q.coeffs().data(), pt.data(), J.data());
  return J;
}

// Compute J_q = d(R(q)^T * p) / dq (3x4 row-major, Eigen xyzw storage).
// Since R(q)^T = R(q^*) where q^* = (-qx, -qy, -qz, qw), apply
// QuaternionRotatePointWithJac(q^*, p) and negate the first 3 columns.
inline Matrix3x4RowMajor QuaternionInverseRotatePointJac(
    const Eigen::Quaterniond& q, const Eigen::Vector3d& pt) {
  Matrix3x4RowMajor J;
  QuaternionRotatePointWithJac(
      q.conjugate().coeffs().data(), pt.data(), J.data());
  J.leftCols<3>() = -J.leftCols<3>();
  return J;
}

// Bilinearly interpolate target intensities (and optional spatial gradients)
// and compute the LS-NCC normalized target patch:
//   p_hat = Psi(I) = (I - mean(I) * 1) / sigma,   sigma = ||I - mean(I) * 1||_2
// and residual vector `r = p_hat - src_norm`, where ||r||^2 = 2 * (1 - NCC).
inline bool InterpolateAndComputeLSNCC(
    const GrayscaleImageLevel& level,
    const Eigen::Ref<const Eigen::VectorXd>& xs,
    const Eigen::Ref<const Eigen::VectorXd>& ys,
    const Eigen::Ref<const Eigen::VectorXd>& src_norm,
    const double min_norm,
    const bool compute_grad,
    Eigen::Ref<Eigen::VectorXd> dxs,
    Eigen::Ref<Eigen::VectorXd> dys,
    Eigen::Ref<Eigen::VectorXd> tgt_norm,
    Eigen::Ref<Eigen::VectorXd> residuals,
    double& inv_sigma) {
  const Eigen::Index num_samples = xs.size();
  if (compute_grad) {
    for (Eigen::Index m = 0; m < num_samples; ++m) {
      const auto val =
          level.InterpolateBilinear(xs[m], ys[m], &dxs[m], &dys[m]);
      if (!val.has_value()) {
        return false;
      }
      tgt_norm[m] = *val;
    }
  } else {
    for (Eigen::Index m = 0; m < num_samples; ++m) {
      const auto val = level.InterpolateBilinear(xs[m], ys[m]);
      if (!val.has_value()) {
        return false;
      }
      tgt_norm[m] = *val;
    }
  }

  tgt_norm.array() -= tgt_norm.mean();
  const double sq_norm = tgt_norm.squaredNorm();
  if (!(sq_norm > 0.0 && sq_norm >= min_norm * min_norm)) {
    return false;
  }
  inv_sigma = 1.0 / std::sqrt(sq_norm);
  tgt_norm *= inv_sigma;
  residuals = tgt_norm - src_norm;
  return true;
}

// Project a raw intensity Jacobian G = dI/dtheta [M x C] through the LS-NCC
// normalization Jacobian:
//   dPsi / dI = (1 / sigma) * (I_M - (1/M) * 1 1^T - p_hat p_hat^T),
// where p_hat = Psi(I) (`tgt_norm`) and inv_sigma = 1 / sigma.
// Safe when `raw_jac` and `out_jac` alias (when `accumulate == false`).
template <typename DerivedG, typename DerivedOut>
inline void ProjectLSNCCJacobian(
    const Eigen::MatrixBase<DerivedG>& raw_jac,
    const Eigen::Ref<const Eigen::VectorXd>& tgt_norm,
    const double inv_sigma,
    Eigen::MatrixBase<DerivedOut>& out_jac,
    const bool accumulate) {
  constexpr int kCols = DerivedG::ColsAtCompileTime;
  using RowVec = std::conditional_t<kCols == Eigen::Dynamic,
                                    RowVectorMaxCamParams,
                                    Eigen::Matrix<double, 1, kCols>>;
  const Eigen::Index rows = raw_jac.rows();
  RowVec col_means = RowVec::Zero(raw_jac.cols());
  RowVec col_dots = RowVec::Zero(raw_jac.cols());
  for (Eigen::Index m = 0; m < rows; ++m) {
    col_means += raw_jac.row(m);
    col_dots += tgt_norm[m] * raw_jac.row(m);
  }
  col_means /= static_cast<double>(rows);
  if (accumulate) {
    for (Eigen::Index m = 0; m < rows; ++m) {
      out_jac.row(m) +=
          inv_sigma * (raw_jac.row(m) - col_means - tgt_norm[m] * col_dots);
    }
  } else {
    for (Eigen::Index m = 0; m < rows; ++m) {
      out_jac.row(m) =
          inv_sigma * (raw_jac.row(m) - col_means - tgt_norm[m] * col_dots);
    }
  }
}

// Apply 2D chain rule G[m, :] = gx[m] * J_raw[m, 0, :] + gy[m] * J_raw[m, 1, :]
// followed by LS-NCC projection J_Psi * G into `out_jac`.
inline void ChainRuleAndProjectLSNCC(
    const Eigen::Ref<const Eigen::VectorXd>& gx,
    const Eigen::Ref<const Eigen::VectorXd>& gy,
    const double* J_raw,
    const Eigen::Ref<const Eigen::VectorXd>& tgt_norm,
    const double inv_sigma,
    Eigen::Map<Eigen::RowMajorMatrixXd> out_jac,
    const bool accumulate) {
  const Eigen::Index num_samples = gx.size();
  const Eigen::Index num_cols = out_jac.cols();
  auto FillRaw = [&](Eigen::Ref<Eigen::RowMajorMatrixXd> dst) {
    for (Eigen::Index m = 0; m < num_samples; ++m) {
      const Eigen::Map<const Eigen::RowMajorMatrixXd> J_m(
          J_raw + static_cast<size_t>(m) * 2 * num_cols, 2, num_cols);
      dst.row(m) = gx[m] * J_m.row(0) + gy[m] * J_m.row(1);
    }
  };
  if (accumulate) {
    MatrixMaxScratch raw_jac(num_samples, num_cols);
    FillRaw(raw_jac);
    ProjectLSNCCJacobian(
        raw_jac, tgt_norm, inv_sigma, out_jac, /*accumulate=*/true);
  } else {
    FillRaw(out_jac);
    ProjectLSNCCJacobian(
        out_jac, tgt_norm, inv_sigma, out_jac, /*accumulate=*/false);
  }
}

// Project a batch of M 3D points `pts_in_cam` (3 x M) using
// `CameraModel::ImgFromCamWithJac`. `J_params_out` (optional, M * 2 *
// num_params) and `J_uvw_out` (optional, M * 6) are stored contiguously per
// sample m; pass nullptr to skip computing them.
template <typename CameraModel>
bool ProjectBatchImpl(const double* params,
                      const Eigen::Ref<const Eigen::Matrix3Xd>& pts_in_cam,
                      double* xs_out,
                      double* ys_out,
                      double* J_params_out,
                      double* J_uvw_out) {
  constexpr int kNumParams = static_cast<int>(CameraModel::num_params);
  const Eigen::Index num_samples = pts_in_cam.cols();
  for (Eigen::Index m = 0; m < num_samples; ++m) {
    double* jp =
        J_params_out == nullptr ? nullptr : (J_params_out + m * 2 * kNumParams);
    double* ju = J_uvw_out == nullptr ? nullptr : (J_uvw_out + m * 6);
    if (!CameraModel::ImgFromCamWithJac(params,
                                        pts_in_cam(0, m),
                                        pts_in_cam(1, m),
                                        pts_in_cam(2, m),
                                        &xs_out[m],
                                        &ys_out[m],
                                        jp,
                                        ju)) {
      return false;
    }
  }
  return true;
}

inline bool ProjectBatch(const CameraModelId model_id,
                         const double* params,
                         const Eigen::Ref<const Eigen::Matrix3Xd>& pts_in_cam,
                         double* xs_out,
                         double* ys_out,
                         double* J_params_out = nullptr,
                         double* J_uvw_out = nullptr) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)    \
  case CameraModel::model_id:             \
    return ProjectBatchImpl<CameraModel>( \
        params, pts_in_cam, xs_out, ys_out, J_params_out, J_uvw_out);
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
  return false;
}

// Unproject fixed source pixels (src_xs[m], src_ys[m]) to perspective rays
// (u_m, v_m, 1) via warm-started Gauss-Newton on `ImgFromCamWithJac`. When
// `d_uv_d_params_out != nullptr`, also computes the implicit-function Jacobian
// from F(uv, params) = pi(uv, 1; params) - x_src = 0:
//   d(u_m, v_m) / d(params) = -J_uv^{-1} * J_params   (2 x num_params per m).
// When `params` matches `init_rays`, the Newton loop exits on iteration 0 after
// a single `ImgFromCamWithJac` call. For spherical models (whose params are
// constant metadata), copies `init_rays` directly with zero Jacobian.
template <typename CameraModel>
bool UnprojectBatchWithJacImpl(
    const double* params,
    const Eigen::Ref<const Eigen::VectorXd>& src_xs,
    const Eigen::Ref<const Eigen::VectorXd>& src_ys,
    const Eigen::Ref<const Eigen::Matrix3Xd>& init_rays,
    Eigen::Ref<Eigen::Matrix3Xd> rays_out,
    double* d_uv_d_params_out) {
  constexpr int kNumParams = static_cast<int>(CameraModel::num_params);
  const Eigen::Index num_samples = init_rays.cols();
  const bool compute_jac = (d_uv_d_params_out != nullptr);

  if constexpr (!std::is_base_of_v<BasePerspectiveCameraModel<CameraModel>,
                                   CameraModel>) {
    rays_out = init_rays;
    if (compute_jac) {
      std::fill(d_uv_d_params_out,
                d_uv_d_params_out + num_samples * 2 * kNumParams,
                0.0);
    }
    return true;
  } else {
    constexpr int kMaxNewtonIters = 15;
    Matrix2x3RowMajor J_uvw;
    Eigen::Matrix<double, 2, kNumParams, Eigen::RowMajor> J_params;
    double* jp_out = compute_jac ? J_params.data() : nullptr;

    for (Eigen::Index m = 0; m < num_samples; ++m) {
      Eigen::Vector2d uv = init_rays.col(m).head<2>();
      const Eigen::Vector2d target(src_xs[m], src_ys[m]);
      // Scale pixel tolerance to 4 ULPs of the pixel coordinates so it stays
      // above double-precision rounding even at >6000 px while remaining tight
      // enough for small distortion perturbations near the image center.
      const double px_scale = std::max(1.0, target.lpNorm<1>());
      const double px_tol =
          4.0 * std::numeric_limits<double>::epsilon() * px_scale;
      const double tol_sq = px_tol * px_tol;

      bool converged = false;
      for (int iter = 0; iter < kMaxNewtonIters; ++iter) {
        Eigen::Vector2d proj;
        if (!CameraModel::ImgFromCamWithJac(params,
                                            uv.x(),
                                            uv.y(),
                                            1.0,
                                            &proj.x(),
                                            &proj.y(),
                                            jp_out,
                                            J_uvw.data())) {
          break;
        }
        const Eigen::Vector2d err = proj - target;
        if (err.squaredNorm() <= tol_sq) {
          converged = true;
          break;
        }
        const Eigen::Matrix2d J_uv = J_uvw.leftCols<2>();
        if (!(std::abs(J_uv.determinant()) > 1e-12)) {
          break;
        }
        const Eigen::Vector2d delta = -J_uv.inverse() * err;
        uv += delta;
        if (delta.squaredNorm() < 1e-28) {
          converged = true;
          break;
        }
      }

      if (!converged) {
        // Fallback to model's full CamFromImg once before declaring failure.
        if (!CameraModel::CamFromImg(
                params, target.x(), target.y(), &uv.x(), &uv.y())) {
          return false;
        }
        Eigen::Vector2d proj;
        if (!CameraModel::ImgFromCamWithJac(params,
                                            uv.x(),
                                            uv.y(),
                                            1.0,
                                            &proj.x(),
                                            &proj.y(),
                                            jp_out,
                                            J_uvw.data())) {
          return false;
        }
        if (!((proj - target).squaredNorm() < 1e-6)) {
          return false;
        }
      }

      rays_out.col(m) = uv.homogeneous();

      if (compute_jac) {
        const Eigen::Matrix2d J_uv = J_uvw.leftCols<2>();
        if (!(std::abs(J_uv.determinant()) > 1e-12)) {
          return false;
        }
        Eigen::Map<Eigen::Matrix<double, 2, kNumParams, Eigen::RowMajor>>(
            d_uv_d_params_out + m * 2 * kNumParams) =
            -J_uv.inverse() * J_params;
      }
    }
    return true;
  }
}

inline bool UnprojectBatchWithJac(
    const CameraModelId model_id,
    const double* params,
    const Eigen::Ref<const Eigen::VectorXd>& src_xs,
    const Eigen::Ref<const Eigen::VectorXd>& src_ys,
    const Eigen::Ref<const Eigen::Matrix3Xd>& init_rays,
    Eigen::Ref<Eigen::Matrix3Xd> rays_out,
    double* d_uv_d_params_out = nullptr) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)             \
  case CameraModel::model_id:                      \
    return UnprojectBatchWithJacImpl<CameraModel>( \
        params, src_xs, src_ys, init_rays, rays_out, d_uv_d_params_out);
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
  return false;
}

inline bool HasSourceCamChanged(const double* src_cam,
                                const std::vector<double>& src_cam_init) {
  return !std::equal(src_cam_init.begin(), src_cam_init.end(), src_cam);
}

// Parameter-block layout for PhotometricNCCCostFunction. Deduplicates shared
// cameras, frames, and rig sensors so Ceres never receives the same double*
// twice in one residual block.
struct SourceTargetBlockLayout {
  CameraModelId src_model_id = CameraModelId::kInvalid;
  CameraModelId tgt_model_id = CameraModelId::kInvalid;
  int src_num_cam_params = 0;
  int tgt_num_cam_params = 0;

  bool same_frame = false;
  bool src_is_ref_sensor = true;
  bool tgt_is_ref_sensor = true;
  bool same_sensor = false;
  bool same_camera = false;

  int plane_idx = 0;
  int src_rig_from_world_idx = -1;
  int tgt_rig_from_world_idx = -1;
  int src_cam_from_rig_idx = -1;
  int tgt_cam_from_rig_idx = -1;
  int src_cam_idx = -1;
  int tgt_cam_idx = -1;

  std::vector<int32_t> block_sizes;

  static SourceTargetBlockLayout Create(const CameraModelId src_model_id,
                                        const CameraModelId tgt_model_id,
                                        const frame_t src_frame_id,
                                        const frame_t tgt_frame_id,
                                        const sensor_t src_sensor_id,
                                        const sensor_t tgt_sensor_id,
                                        const bool src_is_ref_sensor,
                                        const bool tgt_is_ref_sensor,
                                        const camera_t src_camera_id,
                                        const camera_t tgt_camera_id) {
    SourceTargetBlockLayout layout;
    layout.src_model_id = src_model_id;
    layout.tgt_model_id = tgt_model_id;
    layout.src_num_cam_params =
        static_cast<int>(CameraModelNumParams(src_model_id));
    layout.tgt_num_cam_params =
        static_cast<int>(CameraModelNumParams(tgt_model_id));
    THROW_CHECK_LE(layout.src_num_cam_params, kPhotometricMaxCamParams);
    THROW_CHECK_LE(layout.tgt_num_cam_params, kPhotometricMaxCamParams);

    layout.same_frame = (src_frame_id == tgt_frame_id);
    layout.src_is_ref_sensor = src_is_ref_sensor;
    layout.tgt_is_ref_sensor = tgt_is_ref_sensor;
    layout.same_sensor = (!src_is_ref_sensor && !tgt_is_ref_sensor &&
                          src_sensor_id == tgt_sensor_id);
    layout.same_camera = (src_camera_id == tgt_camera_id);

    // Block 0: plane n in R^3.
    layout.plane_idx = 0;
    layout.block_sizes.push_back(3);

    if (!layout.same_frame) {
      layout.src_rig_from_world_idx =
          static_cast<int>(layout.block_sizes.size());
      layout.block_sizes.push_back(7);
      layout.tgt_rig_from_world_idx =
          static_cast<int>(layout.block_sizes.size());
      layout.block_sizes.push_back(7);
    }

    if (!layout.src_is_ref_sensor) {
      layout.src_cam_from_rig_idx = static_cast<int>(layout.block_sizes.size());
      layout.block_sizes.push_back(7);
    }
    if (!layout.tgt_is_ref_sensor) {
      if (layout.same_sensor) {
        layout.tgt_cam_from_rig_idx = layout.src_cam_from_rig_idx;
      } else {
        layout.tgt_cam_from_rig_idx =
            static_cast<int>(layout.block_sizes.size());
        layout.block_sizes.push_back(7);
      }
    }

    layout.src_cam_idx = static_cast<int>(layout.block_sizes.size());
    layout.block_sizes.push_back(layout.src_num_cam_params);
    if (layout.same_camera) {
      layout.tgt_cam_idx = layout.src_cam_idx;
    } else {
      layout.tgt_cam_idx = static_cast<int>(layout.block_sizes.size());
      layout.block_sizes.push_back(layout.tgt_num_cam_params);
    }

    return layout;
  }

  std::vector<double*> BuildParameterPointers(double* plane_ptr,
                                              double* src_rig_from_world_ptr,
                                              double* tgt_rig_from_world_ptr,
                                              double* src_cam_from_rig_ptr,
                                              double* tgt_cam_from_rig_ptr,
                                              double* src_cam_ptr,
                                              double* tgt_cam_ptr) const {
    std::vector<double*> ptrs;
    ptrs.reserve(block_sizes.size());
    ptrs.push_back(plane_ptr);
    if (!same_frame) {
      ptrs.push_back(src_rig_from_world_ptr);
      ptrs.push_back(tgt_rig_from_world_ptr);
    }
    if (!src_is_ref_sensor) {
      ptrs.push_back(src_cam_from_rig_ptr);
    }
    if (!tgt_is_ref_sensor && !same_sensor) {
      ptrs.push_back(tgt_cam_from_rig_ptr);
    }
    ptrs.push_back(src_cam_ptr);
    if (!same_camera) {
      ptrs.push_back(tgt_cam_ptr);
    }
    THROW_CHECK_EQ(ptrs.size(), block_sizes.size());
    return ptrs;
  }
};

// Unpacked SE(3) rigid transform (quaternion, rotation matrix, translation).
// Intentionally left uninitialized by default; construct via Identity() or
// FromParams() to avoid redundant writes on the hot evaluation path.
struct UnpackedPose {
  Eigen::Quaterniond q;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;

  static UnpackedPose Identity() {
    return {Eigen::Quaterniond::Identity(),
            Eigen::Matrix3d::Identity(),
            Eigen::Vector3d::Zero()};
  }

  static UnpackedPose FromParams(const double* params) {
    const Eigen::Quaterniond q = EigenQuaternionMap<double>(params);
    return {q, q.toRotationMatrix(), EigenVector3Map<double>(params + 4)};
  }
};

// Unpacked frame & rig transforms and composed relative transform
//   T_tgt_from_src = T_tgt_from_world * T_src_from_world^{-1}
// where T_cam_from_world = T_cam_from_rig * T_rig_from_world.
// All members are written once in FromParameters(); no default initializers.
struct RelativePoseState {
  UnpackedPose src_rig_from_world;
  UnpackedPose tgt_rig_from_world;
  UnpackedPose src_cam_from_rig;
  UnpackedPose tgt_cam_from_rig;

  Eigen::Matrix3d tgt_cam_from_world_rot;
  Eigen::Matrix3d tgt_cam_from_src_cam_rot;
  Eigen::Vector3d tgt_cam_from_src_cam_trans;

  static RelativePoseState FromParameters(const SourceTargetBlockLayout& layout,
                                          double const* const* parameters) {
    RelativePoseState state;
    if (layout.same_frame) {
      state.src_rig_from_world = UnpackedPose::Identity();
      state.tgt_rig_from_world = UnpackedPose::Identity();
    } else {
      state.src_rig_from_world =
          UnpackedPose::FromParams(parameters[layout.src_rig_from_world_idx]);
      state.tgt_rig_from_world =
          UnpackedPose::FromParams(parameters[layout.tgt_rig_from_world_idx]);
    }
    state.src_cam_from_rig =
        layout.src_is_ref_sensor
            ? UnpackedPose::Identity()
            : UnpackedPose::FromParams(parameters[layout.src_cam_from_rig_idx]);
    state.tgt_cam_from_rig =
        layout.tgt_is_ref_sensor
            ? UnpackedPose::Identity()
            : UnpackedPose::FromParams(parameters[layout.tgt_cam_from_rig_idx]);

    const Eigen::Matrix3d src_cam_from_world_rot =
        state.src_cam_from_rig.R * state.src_rig_from_world.R;
    const Eigen::Vector3d src_cam_from_world_trans =
        state.src_cam_from_rig.R * state.src_rig_from_world.t +
        state.src_cam_from_rig.t;
    state.tgt_cam_from_world_rot =
        state.tgt_cam_from_rig.R * state.tgt_rig_from_world.R;
    const Eigen::Vector3d tgt_cam_from_world_trans =
        state.tgt_cam_from_rig.R * state.tgt_rig_from_world.t +
        state.tgt_cam_from_rig.t;

    state.tgt_cam_from_src_cam_rot =
        state.tgt_cam_from_world_rot * src_cam_from_world_rot.transpose();
    state.tgt_cam_from_src_cam_trans =
        tgt_cam_from_world_trans -
        state.tgt_cam_from_src_cam_rot * src_cam_from_world_trans;
    return state;
  }
};

struct BlockJacobianPtrs {
  double* plane = nullptr;
  double* src_rig_from_world = nullptr;
  double* tgt_rig_from_world = nullptr;
  double* src_cam_from_rig = nullptr;
  double* tgt_cam_from_rig = nullptr;
  double* src_cam = nullptr;
  double* tgt_cam = nullptr;

  static BlockJacobianPtrs FromJacobians(const SourceTargetBlockLayout& layout,
                                         double** jacobians) {
    BlockJacobianPtrs jac;
    if (jacobians == nullptr) {
      return jac;
    }
    jac.plane = jacobians[layout.plane_idx];
    if (!layout.same_frame) {
      jac.src_rig_from_world = jacobians[layout.src_rig_from_world_idx];
      jac.tgt_rig_from_world = jacobians[layout.tgt_rig_from_world_idx];
    }
    if (!layout.src_is_ref_sensor) {
      jac.src_cam_from_rig = jacobians[layout.src_cam_from_rig_idx];
    }
    if (!layout.tgt_is_ref_sensor) {
      jac.tgt_cam_from_rig = jacobians[layout.tgt_cam_from_rig_idx];
    }
    jac.src_cam = jacobians[layout.src_cam_idx];
    jac.tgt_cam = jacobians[layout.tgt_cam_idx];
    return jac;
  }

  bool NeedAny() const {
    return plane || src_rig_from_world || tgt_rig_from_world ||
           src_cam_from_rig || tgt_cam_from_rig || src_cam || tgt_cam;
  }
  bool NeedJUvw() const {
    return plane || src_rig_from_world || tgt_rig_from_world ||
           src_cam_from_rig || tgt_cam_from_rig || src_cam;
  }
  bool NeedPoses() const {
    return src_rig_from_world || tgt_rig_from_world || src_cam_from_rig ||
           tgt_cam_from_rig;
  }
};

// Transform alpha-scaled 3D points backwards through `child_from_parent`:
//   p_parent_m = R^T * (p_child_m - alpha_m * t)
//              = R^T * p_child_m - (R^T * t) * alpha_m.
inline void TransformScaledPointsInverse(
    const UnpackedPose& child_from_parent,
    const Eigen::Ref<const Eigen::Matrix3Xd>& pts_in_child,
    const Eigen::Ref<const Eigen::VectorXd>& plane_dots,
    Matrix3xMaxM& pts_in_parent) {
  const Eigen::Matrix3d R_T = child_from_parent.R.transpose();
  const Eigen::Vector3d rotated_t = R_T * child_from_parent.t;
  pts_in_parent.noalias() = R_T.lazyProduct(pts_in_child);
  pts_in_parent.noalias() -= rotated_t.lazyProduct(plane_dots.transpose());
}

// Transform alpha-scaled 3D points forwards through `child_from_parent`:
//   p_child_m = R * p_parent_m + alpha_m * t.
inline void TransformScaledPointsForward(
    const UnpackedPose& child_from_parent,
    const Eigen::Ref<const Eigen::Matrix3Xd>& pts_in_parent,
    const Eigen::Ref<const Eigen::VectorXd>& plane_dots,
    Matrix3xMaxM& pts_in_child) {
  pts_in_child.noalias() = child_from_parent.R.lazyProduct(pts_in_parent);
  pts_in_child.noalias() +=
      child_from_parent.t.lazyProduct(plane_dots.transpose());
}

// Write the LS-NCC-projected 7-DOF Jacobian (quaternion xyzw + translation) for
// a forward pose `child_from_parent` along the target branch:
//   p_child_m = R(q) * p_parent_m + alpha_m * t
//   dI_m / dq = (dI_m / dp_child_m) * d(R(q) * p_parent_m) / dq
//   dI_m / dt = alpha_m * (dI_m / dp_child_m).
inline void WriteForwardPoseJacobian(
    const UnpackedPose& child_from_parent,
    const Eigen::Ref<const Eigen::Matrix3Xd>& pts_in_parent,
    const Eigen::Ref<const Eigen::VectorXd>& plane_dots,
    const Eigen::Ref<const MatrixMx3RowMajor>& dI_d_child_pt,
    const Eigen::Ref<const Eigen::VectorXd>& tgt_norm,
    const double inv_sigma,
    double* jac_out) {
  const int num_samples = static_cast<int>(plane_dots.size());
  auto J_out = MapFixedColMatrix<7>(jac_out, num_samples);
  for (int m = 0; m < num_samples; ++m) {
    J_out.row(m).head<4>().noalias() =
        dI_d_child_pt.row(m) *
        QuaternionRotatePointJac(child_from_parent.q, pts_in_parent.col(m));
    J_out.row(m).tail<3>() = plane_dots[m] * dI_d_child_pt.row(m);
  }
  ProjectLSNCCJacobian(J_out, tgt_norm, inv_sigma, J_out, /*accumulate=*/false);
}

// Write (or accumulate, when `same_sensor`) the LS-NCC-projected 7-DOF Jacobian
// for an inverted pose `child_from_parent` along the source branch:
//   shifted_m  = p_child_m - alpha_m * t
//   p_parent_m = R(q)^T * shifted_m
//   dI_m / dq  = (dI_m / dp_parent_m) * d(R(q)^T * shifted_m) / dq
//   dI_m / dt  = -alpha_m * (dI_m / dp_parent_m) * R(q)^T.
inline void WriteInversePoseJacobian(
    const UnpackedPose& child_from_parent,
    const Eigen::Ref<const Eigen::Matrix3Xd>& pts_in_child,
    const Eigen::Ref<const Eigen::VectorXd>& plane_dots,
    const Eigen::Ref<const MatrixMx3RowMajor>& dI_d_parent_pt,
    const Eigen::Ref<const Eigen::VectorXd>& tgt_norm,
    const double inv_sigma,
    double* jac_out,
    const bool accumulate) {
  const int num_samples = static_cast<int>(plane_dots.size());
  const Eigen::Matrix3d R_T = child_from_parent.R.transpose();
  auto FillRaw =
      [&](Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>>
              dst) {
        for (int m = 0; m < num_samples; ++m) {
          const Eigen::Vector3d shifted =
              pts_in_child.col(m) - plane_dots[m] * child_from_parent.t;
          dst.row(m).head<4>().noalias() =
              dI_d_parent_pt.row(m) *
              QuaternionInverseRotatePointJac(child_from_parent.q, shifted);
          dst.row(m).tail<3>().noalias() =
              (-plane_dots[m]) * (dI_d_parent_pt.row(m) * R_T);
        }
      };

  auto J_out = MapFixedColMatrix<7>(jac_out, num_samples);
  if (accumulate) {
    MatrixMaxMx7RowMajor raw_jac(num_samples, 7);
    FillRaw(raw_jac);
    ProjectLSNCCJacobian(
        raw_jac, tgt_norm, inv_sigma, J_out, /*accumulate=*/true);
  } else {
    FillRaw(J_out);
    ProjectLSNCCJacobian(
        J_out, tgt_norm, inv_sigma, J_out, /*accumulate=*/false);
  }
}

}  // namespace photometric_internal

// Analytical Least-Squares NCC (Woodford 2018) photometric bundle adjustment
// cost function using LSPBA's (Woodford & Rosten 2020) source-anchored plane
// parameterization n in R^3 (points on the plane satisfy n^T X_src = 1).
//
// Residual vector (length M = patch.NumSamples()):
//   r = Psi(I_tgt(x')) - Psi(I_src(x))
// where ||r||^2 = 2 * (1 - NCC) in [0, 4].
class PhotometricNCCCostFunction : public ceres::CostFunction {
 public:
  PhotometricNCCCostFunction(
      const PhotometricPatchData& patch,
      const GrayscaleImageLevel& target_level,
      photometric_internal::SourceTargetBlockLayout layout,
      const double min_target_norm)
      : patch_(patch),
        target_level_(target_level),
        layout_(std::move(layout)),
        min_target_norm_(min_target_norm) {
    const int num_samples = patch_.NumSamples();
    THROW_CHECK_GE(num_samples, 4);
    THROW_CHECK_LE(num_samples, kPhotometricMaxPatchSamples);
    set_num_residuals(num_samples);
    *mutable_parameter_block_sizes() = layout_.block_sizes;
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    const int num_samples = patch_.NumSamples();
    const int num_blocks = static_cast<int>(layout_.block_sizes.size());

    auto SetBarrierAndZeroJacobians = [&]() {
      // Worst-case ZNSSD residual: ||r||^2 = 4.0 (NCC = -1), zero gradient.
      // Prevents Levenberg-Marquardt from being rewarded for pushing patches
      // out of bounds, behind a camera, or into textureless regions.
      const double barrier = 2.0 / std::sqrt(static_cast<double>(num_samples));
      Eigen::Map<Eigen::VectorXd>(residuals, num_samples).setConstant(barrier);
      if (jacobians != nullptr) {
        for (int b = 0; b < num_blocks; ++b) {
          if (jacobians[b] != nullptr) {
            Eigen::Map<Eigen::VectorXd>(
                jacobians[b],
                static_cast<Eigen::Index>(num_samples) * layout_.block_sizes[b])
                .setZero();
          }
        }
      }
      return true;
    };

    const Eigen::Vector3d plane =
        EigenVector3Map<double>(parameters[layout_.plane_idx]);
    const double* src_cam = parameters[layout_.src_cam_idx];
    const double* tgt_cam = parameters[layout_.tgt_cam_idx];

    const auto pose = photometric_internal::RelativePoseState::FromParameters(
        layout_, parameters);
    const auto jac = photometric_internal::BlockJacobianPtrs::FromJacobians(
        layout_, jacobians);
    const bool need_any_jac = jac.NeedAny();
    const bool need_J_uvw = jac.NeedJUvw();

    // Step 1: Source unprojection (skip entirely when src_cam is unchanged and
    // no source-camera Jacobian is requested).
    const bool src_cam_changed = photometric_internal::HasSourceCamChanged(
        src_cam, patch_.src_cam_params_init);
    const bool run_unproject = src_cam_changed || jac.src_cam != nullptr;

    photometric_internal::Matrix3xMaxM src_rays_buf;
    photometric_internal::MatrixMaxCamJac d_uv_d_src_cam;
    if (run_unproject) {
      src_rays_buf.resize(3, num_samples);
      double* d_uv_ptr = nullptr;
      if (jac.src_cam != nullptr) {
        d_uv_d_src_cam.resize(2 * num_samples, layout_.src_num_cam_params);
        d_uv_ptr = d_uv_d_src_cam.data();
      }
      if (!photometric_internal::UnprojectBatchWithJac(layout_.src_model_id,
                                                       src_cam,
                                                       patch_.src_xs,
                                                       patch_.src_ys,
                                                       patch_.src_rays_init,
                                                       src_rays_buf,
                                                       d_uv_ptr)) {
        return SetBarrierAndZeroJacobians();
      }
    }
    const Eigen::Ref<const Eigen::Matrix3Xd> src_rays =
        run_unproject
            ? Eigen::Ref<const Eigen::Matrix3Xd>(src_rays_buf)
            : Eigen::Ref<const Eigen::Matrix3Xd>(patch_.src_rays_init);

    // Step 2: Warp source rays u_m through the plane homography.
    // Since X_src_m = u_m / alpha_m with alpha_m = n^T u_m > 0, the target
    // camera ray (scaled by alpha_m, leaving scale-invariant projection
    // unchanged) is:
    //   y_m = alpha_m * X_tgt_m = (R_tgt_src + t_tgt_src * n^T) * u_m.
    photometric_internal::VectorMaxM plane_dots(num_samples);
    plane_dots.noalias() = src_rays.transpose().lazyProduct(plane);
    for (int m = 0; m < num_samples; ++m) {
      if (!(plane_dots[m] > 1e-8)) {
        return SetBarrierAndZeroJacobians();
      }
    }
    const Eigen::Matrix3d H_tgt_from_src =
        pose.tgt_cam_from_src_cam_rot +
        pose.tgt_cam_from_src_cam_trans * plane.transpose();
    photometric_internal::Matrix3xMaxM pts_in_tgt_cam(3, num_samples);
    pts_in_tgt_cam.noalias() = H_tgt_from_src.lazyProduct(src_rays);

    // Step 3: Batch target camera projection (+ analytic camera Jacobians).
    photometric_internal::VectorMaxM tgt_xs(num_samples);
    photometric_internal::VectorMaxM tgt_ys(num_samples);
    photometric_internal::MatrixMaxCamJac J_proj_wrt_tgt_cam;
    double* J_tgt_cam_ptr = nullptr;
    if (jac.tgt_cam != nullptr) {
      J_proj_wrt_tgt_cam.resize(2 * num_samples, layout_.tgt_num_cam_params);
      J_tgt_cam_ptr = J_proj_wrt_tgt_cam.data();
    }
    photometric_internal::MatrixMaxMx6RowMajor J_proj_wrt_tgt_pt;
    double* J_uvw_ptr = nullptr;
    if (need_J_uvw) {
      J_proj_wrt_tgt_pt.resize(num_samples, 6);
      J_uvw_ptr = J_proj_wrt_tgt_pt.data();
    }

    if (!photometric_internal::ProjectBatch(layout_.tgt_model_id,
                                            tgt_cam,
                                            pts_in_tgt_cam,
                                            tgt_xs.data(),
                                            tgt_ys.data(),
                                            J_tgt_cam_ptr,
                                            J_uvw_ptr)) {
      return SetBarrierAndZeroJacobians();
    }

    // Step 4: Bilinear interpolation (+ image gradients) and LS-NCC
    // normalization r = Psi(I_tgt) - Psi(I_src).
    photometric_internal::VectorMaxM tgt_dxs(num_samples);
    photometric_internal::VectorMaxM tgt_dys(num_samples);
    photometric_internal::VectorMaxM tgt_norm(num_samples);
    Eigen::Map<Eigen::VectorXd> residuals_vec(residuals, num_samples);
    double inv_sigma = 0.0;

    if (!photometric_internal::InterpolateAndComputeLSNCC(target_level_,
                                                          tgt_xs,
                                                          tgt_ys,
                                                          patch_.src_norm,
                                                          min_target_norm_,
                                                          need_any_jac,
                                                          tgt_dxs,
                                                          tgt_dys,
                                                          tgt_norm,
                                                          residuals_vec,
                                                          inv_sigma)) {
      return SetBarrierAndZeroJacobians();
    }

    if (!need_any_jac) {
      return true;
    }

    // Step 5: Chain rule dI_m / dy_m = [dI/dx, dI/dy] * (d_pi / dy_m)
    // (M x 3 row-major) and LS-NCC projection J_Psi * G into Ceres Jacobians.
    photometric_internal::MatrixMaxMx3RowMajor dI_d_tgt_pt(num_samples, 3);
    if (need_J_uvw) {
      for (int m = 0; m < num_samples; ++m) {
        const Eigen::Map<const photometric_internal::Matrix2x3RowMajor> J_uvw_m(
            J_proj_wrt_tgt_pt.row(m).data());
        dI_d_tgt_pt.row(m).noalias() =
            Eigen::RowVector2d(tgt_dxs[m], tgt_dys[m]) * J_uvw_m;
      }
    }

    // 5a. Plane Jacobian: dy_m / dn = t_tgt_src * u_m^T, so
    //   dI_m / dn = (dI_m / dy_m * t_tgt_src) * u_m^T.
    if (jac.plane != nullptr) {
      auto J_plane =
          photometric_internal::MapFixedColMatrix<3>(jac.plane, num_samples);
      photometric_internal::VectorMaxM dI_d_alpha(num_samples);
      dI_d_alpha.noalias() =
          dI_d_tgt_pt.lazyProduct(pose.tgt_cam_from_src_cam_trans);
      J_plane = src_rays.transpose().array().colwise() * dI_d_alpha.array();
      photometric_internal::ProjectLSNCCJacobian(
          J_plane, tgt_norm, inv_sigma, J_plane, /*accumulate=*/false);
    }

    // 5b. Frame & rig pose Jacobians.
    if (jac.NeedPoses()) {
      ComputePoseJacobians(
          pose, jac, src_rays, plane_dots, dI_d_tgt_pt, tgt_norm, inv_sigma);
    }

    // 5c. Target camera intrinsics Jacobian.
    if (jac.tgt_cam != nullptr) {
      auto J_tgt = photometric_internal::MapRowMajorMatrix(
          jac.tgt_cam, num_samples, layout_.tgt_num_cam_params);
      photometric_internal::ChainRuleAndProjectLSNCC(tgt_dxs,
                                                     tgt_dys,
                                                     J_proj_wrt_tgt_cam.data(),
                                                     tgt_norm,
                                                     inv_sigma,
                                                     J_tgt,
                                                     /*accumulate=*/false);
    }

    // 5d. Source camera intrinsics Jacobian via implicit function theorem:
    //   dI_m / d(u_m, v_m) = (dI_m / dy_m) * H_tgt_from_src[:, 0:2]
    //   dI_m / d(src_cam)  = (dI_m / d(u_m, v_m)) * (d(u_m, v_m) / d(src_cam)).
    // Accumulates into the shared block when source and target share a camera.
    if (jac.src_cam != nullptr) {
      photometric_internal::MatrixMaxMx2 dI_d_src_uv(num_samples, 2);
      dI_d_src_uv.noalias() =
          dI_d_tgt_pt.lazyProduct(H_tgt_from_src.leftCols<2>());

      const bool accumulate = layout_.same_camera && (jac.tgt_cam != nullptr);
      auto J_src = photometric_internal::MapRowMajorMatrix(
          jac.src_cam, num_samples, layout_.src_num_cam_params);
      photometric_internal::ChainRuleAndProjectLSNCC(dI_d_src_uv.col(0),
                                                     dI_d_src_uv.col(1),
                                                     d_uv_d_src_cam.data(),
                                                     tgt_norm,
                                                     inv_sigma,
                                                     J_src,
                                                     accumulate);
    }

    return true;
  }

 private:
  // Differentiate along the kinematic chain of alpha-scaled 3D points:
  //   src_cam (u_m) -> src_rig -> world -> tgt_rig -> tgt_cam (y_m).
  void ComputePoseJacobians(
      const photometric_internal::RelativePoseState& pose,
      const photometric_internal::BlockJacobianPtrs& jac,
      const Eigen::Ref<const Eigen::Matrix3Xd>& src_rays,
      const Eigen::Ref<const Eigen::VectorXd>& plane_dots,
      const Eigen::Ref<const photometric_internal::MatrixMx3RowMajor>&
          dI_d_tgt_pt,
      const Eigen::Ref<const Eigen::VectorXd>& tgt_norm,
      const double inv_sigma) const {
    const int num_samples = static_cast<int>(plane_dots.size());
    const bool need_pts_in_src_rig = jac.tgt_rig_from_world != nullptr ||
                                     jac.src_rig_from_world != nullptr ||
                                     jac.tgt_cam_from_rig != nullptr;
    const bool need_pts_in_world =
        !layout_.same_frame &&
        (jac.tgt_rig_from_world != nullptr || jac.tgt_cam_from_rig != nullptr);

    photometric_internal::Matrix3xMaxM pts_in_src_rig_buf;
    if (need_pts_in_src_rig && !layout_.src_is_ref_sensor) {
      pts_in_src_rig_buf.resize(3, num_samples);
      photometric_internal::TransformScaledPointsInverse(
          pose.src_cam_from_rig, src_rays, plane_dots, pts_in_src_rig_buf);
    }
    const Eigen::Ref<const Eigen::Matrix3Xd> pts_in_src_rig =
        (need_pts_in_src_rig && !layout_.src_is_ref_sensor)
            ? Eigen::Ref<const Eigen::Matrix3Xd>(pts_in_src_rig_buf)
            : src_rays;

    photometric_internal::Matrix3xMaxM pts_in_world_buf;
    if (need_pts_in_world) {
      pts_in_world_buf.resize(3, num_samples);
      photometric_internal::TransformScaledPointsInverse(
          pose.src_rig_from_world,
          pts_in_src_rig,
          plane_dots,
          pts_in_world_buf);
    }
    const Eigen::Ref<const Eigen::Matrix3Xd> pts_in_world =
        need_pts_in_world ? Eigen::Ref<const Eigen::Matrix3Xd>(pts_in_world_buf)
                          : pts_in_src_rig;

    photometric_internal::MatrixMaxMx3RowMajor dI_d_frame_pt(num_samples, 3);

    // Target frame pose: y_m = R_tgt_sfr * (R_tgt_rfw * p_world + alpha*t) + ..
    if (jac.tgt_rig_from_world != nullptr) {
      if (!layout_.tgt_is_ref_sensor) {
        dI_d_frame_pt.noalias() =
            dI_d_tgt_pt.lazyProduct(pose.tgt_cam_from_rig.R);
      }
      const Eigen::Ref<const photometric_internal::MatrixMx3RowMajor>
          dI_d_tgt_rig =
              layout_.tgt_is_ref_sensor
                  ? dI_d_tgt_pt
                  : Eigen::Ref<const photometric_internal::MatrixMx3RowMajor>(
                        dI_d_frame_pt);
      photometric_internal::WriteForwardPoseJacobian(pose.tgt_rig_from_world,
                                                     pts_in_world,
                                                     plane_dots,
                                                     dI_d_tgt_rig,
                                                     tgt_norm,
                                                     inv_sigma,
                                                     jac.tgt_rig_from_world);
    }

    // Source frame pose: p_world = R_src_rfw^T * (p_src_rig - alpha*t_src_rfw).
    if (jac.src_rig_from_world != nullptr) {
      dI_d_frame_pt.noalias() =
          dI_d_tgt_pt.lazyProduct(pose.tgt_cam_from_world_rot);
      photometric_internal::WriteInversePoseJacobian(pose.src_rig_from_world,
                                                     pts_in_src_rig,
                                                     plane_dots,
                                                     dI_d_frame_pt,
                                                     tgt_norm,
                                                     inv_sigma,
                                                     jac.src_rig_from_world,
                                                     /*accumulate=*/false);
    }

    // Target rig sensor pose: y_m = R_tgt_sfr * p_tgt_rig + alpha * t_tgt_sfr.
    if (jac.tgt_cam_from_rig != nullptr) {
      photometric_internal::Matrix3xMaxM pts_in_tgt_rig_buf;
      if (!layout_.same_frame) {
        pts_in_tgt_rig_buf.resize(3, num_samples);
        photometric_internal::TransformScaledPointsForward(
            pose.tgt_rig_from_world,
            pts_in_world,
            plane_dots,
            pts_in_tgt_rig_buf);
      }
      const Eigen::Ref<const Eigen::Matrix3Xd> pts_in_tgt_rig =
          layout_.same_frame
              ? pts_in_src_rig
              : Eigen::Ref<const Eigen::Matrix3Xd>(pts_in_tgt_rig_buf);
      photometric_internal::WriteForwardPoseJacobian(pose.tgt_cam_from_rig,
                                                     pts_in_tgt_rig,
                                                     plane_dots,
                                                     dI_d_tgt_pt,
                                                     tgt_norm,
                                                     inv_sigma,
                                                     jac.tgt_cam_from_rig);
    }

    // Source rig sensor pose: p_src_rig = R_src_sfr^T * (u_m - alpha*t_src_sfr)
    // Accumulates into the shared block when source and target use the same
    // rig sensor across two frames.
    if (jac.src_cam_from_rig != nullptr) {
      if (layout_.same_frame) {
        dI_d_frame_pt.noalias() =
            dI_d_tgt_pt.lazyProduct(pose.tgt_cam_from_rig.R);
      } else {
        dI_d_frame_pt.noalias() =
            dI_d_tgt_pt.lazyProduct(pose.tgt_cam_from_world_rot *
                                    pose.src_rig_from_world.R.transpose());
      }
      const bool accumulate =
          layout_.same_sensor && (jac.tgt_cam_from_rig != nullptr);
      photometric_internal::WriteInversePoseJacobian(pose.src_cam_from_rig,
                                                     src_rays,
                                                     plane_dots,
                                                     dI_d_frame_pt,
                                                     tgt_norm,
                                                     inv_sigma,
                                                     jac.src_cam_from_rig,
                                                     accumulate);
    }
  }

  const PhotometricPatchData& patch_;
  const GrayscaleImageLevel& target_level_;
  const photometric_internal::SourceTargetBlockLayout layout_;
  const double min_target_norm_;
};

}  // namespace colmap
