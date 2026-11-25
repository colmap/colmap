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

// Fixed per-landmark source patch data shared across all target views of the
// landmark at a given pyramid level.
struct PhotometricPatchData {
  // Full-resolution COLMAP pixel coordinates of the M patch samples in the
  // source image.
  std::vector<double> src_xs;
  std::vector<double> src_ys;

  // Precomputed zero-mean unit-norm source intensities Psi(I_src) of length M.
  std::vector<double> src_norm;

  // Initial undistorted source rays (u, v, w) at the start of the current
  // pyramid level (w = 1 for perspective models, unit sphere ray for spherical
  // models). Used directly when source intrinsics are constant or as a
  // thread-safe warm-start for Newton undistortion when variable.
  std::vector<Eigen::Vector3d> src_rays_init;

  // Snapshot of source camera parameters at the start of the current pyramid
  // level so Evaluate() can detect whether undistortion needs refinement.
  std::vector<double> src_cam_params_init;

  // Full-resolution anchor pixel (center of patch) used to reconstruct the
  // updated 3D point along the source ray after plane refinement.
  Eigen::Vector2d anchor_xy = Eigen::Vector2d::Zero();

  int NumSamples() const { return static_cast<int>(src_norm.size()); }
};

namespace photometric_internal {

using MatrixMx3RowMajor =
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatrixMx2ColMajor = Eigen::Matrix<double, Eigen::Dynamic, 2>;
using Matrix3x4RowMajor = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
using Matrix2x3RowMajor = Eigen::Matrix<double, 2, 3, Eigen::RowMajor>;
using RowVectorMaxCamParams = Eigen::Matrix<double,
                                            1,
                                            Eigen::Dynamic,
                                            Eigen::RowMajor,
                                            1,
                                            kPhotometricMaxCamParams>;

// Small stack-backed buffer with heap fallback for large patch sizes.
template <typename T, size_t kStackCapacity>
class InlineBuffer {
 public:
  span<T> Resize(const size_t count) {
    if (count <= kStackCapacity) {
      return span<T>(stack_, count);
    }
    heap_.resize(count);
    return span<T>(heap_.data(), count);
  }

 private:
  T stack_[kStackCapacity];
  std::vector<T> heap_;
};

inline Eigen::Map<Eigen::VectorXd> MapVector(span<double> buf) {
  return Eigen::Map<Eigen::VectorXd>(buf.begin(),
                                     static_cast<Eigen::Index>(buf.size()));
}

template <int kCols>
inline Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, kCols, Eigen::RowMajor>>
MapFixedColMatrix(double* ptr, const int rows) {
  return Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, kCols, Eigen::RowMajor>>(
      ptr, rows, kCols);
}

template <int kCols>
inline Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, kCols, Eigen::RowMajor>>
MapFixedColMatrix(span<double> buf, const int rows) {
  THROW_CHECK_EQ(buf.size(), static_cast<size_t>(rows) * kCols);
  return Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, kCols, Eigen::RowMajor>>(
      buf.begin(), rows, kCols);
}

inline Eigen::Map<Eigen::RowMajorMatrixXd> MapRowMajorMatrix(span<double> buf,
                                                             const int rows,
                                                             const int cols) {
  THROW_CHECK_EQ(buf.size(), static_cast<size_t>(rows) * cols);
  return Eigen::Map<Eigen::RowMajorMatrixXd>(buf.begin(), rows, cols);
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
// and compute the LS-NCC normalized patch `p_norm` and residuals
// `r = p_norm - src_norm`.
inline bool InterpolateAndComputeLSNCC(
    const GrayscaleImageLevel& level,
    const span<const double> xs,
    const span<const double> ys,
    const Eigen::Ref<const Eigen::VectorXd>& src_norm,
    const double min_norm,
    const bool compute_grad,
    Eigen::Ref<Eigen::VectorXd> dxs,
    Eigen::Ref<Eigen::VectorXd> dys,
    Eigen::Ref<Eigen::VectorXd> p_norm,
    Eigen::Ref<Eigen::VectorXd> residuals,
    double& inv_sigma) {
  const int n = static_cast<int>(xs.size());
  if (compute_grad) {
    for (int m = 0; m < n; ++m) {
      const auto val =
          level.InterpolateBilinear(xs[m], ys[m], &dxs[m], &dys[m]);
      if (!val.has_value()) {
        return false;
      }
      p_norm[m] = *val;
    }
  } else {
    for (int m = 0; m < n; ++m) {
      const auto val = level.InterpolateBilinear(xs[m], ys[m]);
      if (!val.has_value()) {
        return false;
      }
      p_norm[m] = *val;
    }
  }

  p_norm.array() -= p_norm.mean();
  const double sq_norm = p_norm.squaredNorm();
  if (!(sq_norm > 0.0 && sq_norm >= min_norm * min_norm)) {
    return false;
  }
  inv_sigma = 1.0 / std::sqrt(sq_norm);
  p_norm *= inv_sigma;
  residuals = p_norm - src_norm;
  return true;
}

// Project a row-major [n x C] raw Jacobian matrix G through the LS-NCC
// normalization Jacobian J_Psi = inv_sigma * (I - 1/n 11^T - p_norm p_norm^T).
// Safe when G and out_jac alias (accumulate == false).
template <typename DerivedG, typename DerivedOut>
inline void ProjectLSNCCJacobian(
    const Eigen::MatrixBase<DerivedG>& G,
    const Eigen::Ref<const Eigen::VectorXd>& p_norm,
    const double inv_sigma,
    Eigen::MatrixBase<DerivedOut>& out_jac,
    const bool accumulate) {
  constexpr int kCols = DerivedG::ColsAtCompileTime;
  using RowVec = std::conditional_t<kCols == Eigen::Dynamic,
                                    RowVectorMaxCamParams,
                                    Eigen::Matrix<double, 1, kCols>>;
  const Eigen::Index rows = G.rows();
  RowVec col_means = RowVec::Zero(G.cols());
  RowVec col_dots = RowVec::Zero(G.cols());
  for (Eigen::Index m = 0; m < rows; ++m) {
    col_means += G.row(m);
    col_dots += p_norm[m] * G.row(m);
  }
  col_means /= static_cast<double>(rows);
  if (accumulate) {
    for (Eigen::Index m = 0; m < rows; ++m) {
      out_jac.row(m) +=
          inv_sigma * (G.row(m) - col_means - p_norm[m] * col_dots);
    }
  } else {
    for (Eigen::Index m = 0; m < rows; ++m) {
      out_jac.row(m) =
          inv_sigma * (G.row(m) - col_means - p_norm[m] * col_dots);
    }
  }
}

// Chain rule G[m, :] = gx[m] * J_raw[m, 0, :] + gy[m] * J_raw[m, 1, :]
// followed by LS-NCC projection J_Psi * G into out_jac.
inline void ChainRuleAndProjectLSNCC(
    const Eigen::Ref<const Eigen::VectorXd>& gx,
    const Eigen::Ref<const Eigen::VectorXd>& gy,
    const span<const double> J_raw,
    const Eigen::Ref<const Eigen::VectorXd>& p_norm,
    const double inv_sigma,
    Eigen::Map<Eigen::RowMajorMatrixXd> g_work,
    Eigen::Map<Eigen::RowMajorMatrixXd> out_jac,
    const bool accumulate) {
  const int n = static_cast<int>(gx.size());
  const int num_cols = static_cast<int>(out_jac.cols());
  for (int m = 0; m < n; ++m) {
    const Eigen::Map<const Eigen::RowMajorMatrixXd> J_m(
        J_raw.begin() + static_cast<size_t>(m) * 2 * num_cols, 2, num_cols);
    g_work.row(m) = gx[m] * J_m.row(0) + gy[m] * J_m.row(1);
  }
  ProjectLSNCCJacobian(g_work, p_norm, inv_sigma, out_jac, accumulate);
}

// Project a batch of M 3D points `pts_in_cam` using
// `CameraModel::ImgFromCamWithJac`. `J_params_out` (optional, size M * 2 *
// num_params) and `J_uvw_out` (optional, size M * 6) are stored contiguously
// per sample m; pass empty spans to skip computing them.
template <typename CameraModel>
bool ProjectBatchImpl(span<const double> params,
                      span<const Eigen::Vector3d> pts_in_cam,
                      span<double> xs_out,
                      span<double> ys_out,
                      span<double> J_params_out,
                      span<double> J_uvw_out) {
  constexpr int kNumParams = static_cast<int>(CameraModel::num_params);
  const size_t n = pts_in_cam.size();
  for (size_t m = 0; m < n; ++m) {
    double* jp = J_params_out.empty()
                     ? nullptr
                     : (J_params_out.begin() + m * 2 * kNumParams);
    double* ju = J_uvw_out.empty() ? nullptr : (J_uvw_out.begin() + m * 6);
    if (!CameraModel::ImgFromCamWithJac(params.begin(),
                                        pts_in_cam[m].x(),
                                        pts_in_cam[m].y(),
                                        pts_in_cam[m].z(),
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
                         span<const double> params,
                         span<const Eigen::Vector3d> pts_in_cam,
                         span<double> xs_out,
                         span<double> ys_out,
                         span<double> J_params_out,
                         span<double> J_uvw_out) {
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

// Unproject a batch of fixed source pixels (src_xs[m], src_ys[m]) to rays
// (u_m, v_m, 1) for perspective models using warm-started Gauss-Newton with
// ImgFromCamWithJac, and optionally compute the implicit-function Jacobian
//   d(u_m, v_m) / d(params) = -J_uv^{-1} * J_params   (2 x num_params per m)
// when `d_uv_d_params_out` is non-empty.
// When `skip_newton` is true (source intrinsics unchanged from init), evaluates
// ImgFromCamWithJac once at `init_rays[m]` without Newton iterations.
// For spherical models (whose params are constant metadata), returns the
// initial unit rays directly with zero param Jacobian.
template <typename CameraModel>
bool UnprojectBatchWithJacImpl(span<const double> params,
                               span<const double> src_xs,
                               span<const double> src_ys,
                               span<const Eigen::Vector3d> init_rays,
                               const bool skip_newton,
                               span<Eigen::Vector3d> rays_out,
                               span<double> d_uv_d_params_out) {
  constexpr int kNumParams = static_cast<int>(CameraModel::num_params);
  const size_t n = init_rays.size();
  const bool compute_jac = !d_uv_d_params_out.empty();

  if constexpr (!std::is_base_of_v<BasePerspectiveCameraModel<CameraModel>,
                                   CameraModel>) {
    std::copy(init_rays.begin(), init_rays.end(), rays_out.begin());
    if (compute_jac) {
      std::fill(d_uv_d_params_out.begin(), d_uv_d_params_out.end(), 0.0);
    }
    return true;
  } else {
    constexpr int kMaxNewtonIters = 15;
    Matrix2x3RowMajor J_uvw;
    Eigen::Matrix<double, 2, kNumParams, Eigen::RowMajor> J_params;
    double* jp_out = compute_jac ? J_params.data() : nullptr;

    for (size_t m = 0; m < n; ++m) {
      Eigen::Vector2d uv = init_rays[m].head<2>();
      const Eigen::Vector2d target(src_xs[m], src_ys[m]);
      // Scale pixel tolerance to 4 ULPs of the pixel coordinates so it stays
      // above double-precision rounding even at >6000 px while remaining tight
      // enough for small distortion perturbations near the image center.
      const double px_scale = std::max(1.0, target.lpNorm<1>());
      const double px_tol =
          4.0 * std::numeric_limits<double>::epsilon() * px_scale;
      const double tol_sq = px_tol * px_tol;

      if (skip_newton) {
        if (compute_jac) {
          Eigen::Vector2d proj;
          if (!CameraModel::ImgFromCamWithJac(params.begin(),
                                              uv.x(),
                                              uv.y(),
                                              1.0,
                                              &proj.x(),
                                              &proj.y(),
                                              jp_out,
                                              J_uvw.data())) {
            return false;
          }
        }
      } else {
        bool converged = false;
        for (int iter = 0; iter < kMaxNewtonIters; ++iter) {
          Eigen::Vector2d proj;
          if (!CameraModel::ImgFromCamWithJac(params.begin(),
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
                  params.begin(), target.x(), target.y(), &uv.x(), &uv.y())) {
            return false;
          }
          Eigen::Vector2d proj;
          if (!CameraModel::ImgFromCamWithJac(params.begin(),
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
      }

      rays_out[m] = uv.homogeneous();

      if (compute_jac) {
        const Eigen::Matrix2d J_uv = J_uvw.leftCols<2>();
        if (!(std::abs(J_uv.determinant()) > 1e-12)) {
          return false;
        }
        Eigen::Map<Eigen::Matrix<double, 2, kNumParams, Eigen::RowMajor>>(
            d_uv_d_params_out.begin() + m * 2 * kNumParams) =
            -J_uv.inverse() * J_params;
      }
    }
    return true;
  }
}

inline bool UnprojectBatchWithJac(const CameraModelId model_id,
                                  span<const double> params,
                                  span<const double> src_xs,
                                  span<const double> src_ys,
                                  span<const Eigen::Vector3d> init_rays,
                                  const bool skip_newton,
                                  span<Eigen::Vector3d> rays_out,
                                  span<double> d_uv_d_params_out) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                         \
  case CameraModel::model_id:                                  \
    return UnprojectBatchWithJacImpl<CameraModel>(params,      \
                                                  src_xs,      \
                                                  src_ys,      \
                                                  init_rays,   \
                                                  skip_newton, \
                                                  rays_out,    \
                                                  d_uv_d_params_out);
    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
  }
  return false;
}

inline bool HasSourceCamChanged(const span<const double> src_cam,
                                const span<const double> src_cam_init) {
  return !std::equal(
      src_cam.begin(), src_cam.end(), src_cam_init.begin(), src_cam_init.end());
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
  int src_rfw_idx = -1;
  int tgt_rfw_idx = -1;
  int src_sfr_idx = -1;
  int tgt_sfr_idx = -1;
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
      layout.src_rfw_idx = static_cast<int>(layout.block_sizes.size());
      layout.block_sizes.push_back(7);
      layout.tgt_rfw_idx = static_cast<int>(layout.block_sizes.size());
      layout.block_sizes.push_back(7);
    }

    if (!layout.src_is_ref_sensor) {
      layout.src_sfr_idx = static_cast<int>(layout.block_sizes.size());
      layout.block_sizes.push_back(7);
    }
    if (!layout.tgt_is_ref_sensor) {
      if (layout.same_sensor) {
        layout.tgt_sfr_idx = layout.src_sfr_idx;
      } else {
        layout.tgt_sfr_idx = static_cast<int>(layout.block_sizes.size());
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
                                              double* src_rfw_ptr,
                                              double* tgt_rfw_ptr,
                                              double* src_sfr_ptr,
                                              double* tgt_sfr_ptr,
                                              double* src_cam_ptr,
                                              double* tgt_cam_ptr) const {
    std::vector<double*> ptrs;
    ptrs.reserve(block_sizes.size());
    ptrs.push_back(plane_ptr);
    if (!same_frame) {
      ptrs.push_back(src_rfw_ptr);
      ptrs.push_back(tgt_rfw_ptr);
    }
    if (!src_is_ref_sensor) {
      ptrs.push_back(src_sfr_ptr);
    }
    if (!tgt_is_ref_sensor && !same_sensor) {
      ptrs.push_back(tgt_sfr_ptr);
    }
    ptrs.push_back(src_cam_ptr);
    if (!same_camera) {
      ptrs.push_back(tgt_cam_ptr);
    }
    THROW_CHECK_EQ(ptrs.size(), block_sizes.size());
    return ptrs;
  }
};

// Unpacked frame & rig poses and composed relative transform
// T_ji = T_j * T_i^-1 for PhotometricNCCCostFunction.
struct RelativePoseState {
  Eigen::Quaterniond q_fi = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d R_fi = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_fi = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_fj = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d R_fj = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_fj = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_si = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d R_si = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_si = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_sj = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d R_sj = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_sj = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_j = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_ji = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_ji = Eigen::Vector3d::Zero();

  static RelativePoseState FromParameters(const SourceTargetBlockLayout& layout,
                                          double const* const* parameters) {
    RelativePoseState s;
    if (!layout.same_frame) {
      const double* src_rfw = parameters[layout.src_rfw_idx];
      const double* tgt_rfw = parameters[layout.tgt_rfw_idx];
      s.q_fi = EigenQuaternionMap<double>(src_rfw);
      s.R_fi = s.q_fi.toRotationMatrix();
      s.t_fi = EigenVector3Map<double>(src_rfw + 4);
      s.q_fj = EigenQuaternionMap<double>(tgt_rfw);
      s.R_fj = s.q_fj.toRotationMatrix();
      s.t_fj = EigenVector3Map<double>(tgt_rfw + 4);
    }
    if (!layout.src_is_ref_sensor) {
      const double* src_sfr = parameters[layout.src_sfr_idx];
      s.q_si = EigenQuaternionMap<double>(src_sfr);
      s.R_si = s.q_si.toRotationMatrix();
      s.t_si = EigenVector3Map<double>(src_sfr + 4);
    }
    if (!layout.tgt_is_ref_sensor) {
      const double* tgt_sfr = parameters[layout.tgt_sfr_idx];
      s.q_sj = EigenQuaternionMap<double>(tgt_sfr);
      s.R_sj = s.q_sj.toRotationMatrix();
      s.t_sj = EigenVector3Map<double>(tgt_sfr + 4);
    }

    const Eigen::Matrix3d R_i = s.R_si * s.R_fi;
    const Eigen::Vector3d t_i = s.R_si * s.t_fi + s.t_si;
    s.R_j = s.R_sj * s.R_fj;
    const Eigen::Vector3d t_j = s.R_sj * s.t_fj + s.t_sj;
    s.R_ji = s.R_j * R_i.transpose();
    s.t_ji = t_j - s.R_ji * t_i;
    return s;
  }
};

struct BlockJacobianPtrs {
  double* plane = nullptr;
  double* src_rfw = nullptr;
  double* tgt_rfw = nullptr;
  double* src_sfr = nullptr;
  double* tgt_sfr = nullptr;
  double* src_cam = nullptr;
  double* tgt_cam = nullptr;

  static BlockJacobianPtrs FromJacobians(const SourceTargetBlockLayout& layout,
                                         double** jacobians) {
    BlockJacobianPtrs j;
    if (jacobians == nullptr) {
      return j;
    }
    j.plane = jacobians[layout.plane_idx];
    if (!layout.same_frame) {
      j.src_rfw = jacobians[layout.src_rfw_idx];
      j.tgt_rfw = jacobians[layout.tgt_rfw_idx];
    }
    if (!layout.src_is_ref_sensor) {
      j.src_sfr = jacobians[layout.src_sfr_idx];
    }
    if (!layout.tgt_is_ref_sensor) {
      j.tgt_sfr = jacobians[layout.tgt_sfr_idx];
    }
    j.src_cam = jacobians[layout.src_cam_idx];
    j.tgt_cam = jacobians[layout.tgt_cam_idx];
    return j;
  }

  bool NeedAny() const {
    return plane || src_rfw || tgt_rfw || src_sfr || tgt_sfr || src_cam ||
           tgt_cam;
  }
  bool NeedJUvw() const {
    return plane || src_rfw || tgt_rfw || src_sfr || tgt_sfr || src_cam;
  }
  bool NeedPoses() const { return src_rfw || tgt_rfw || src_sfr || tgt_sfr; }
};

}  // namespace photometric_internal

// Analytical Least-Squares NCC (Woodford 2018) photometric bundle adjustment
// cost function using LSPBA's (Woodford & Rosten 2020) source-anchored plane
// parameterization n in R^3.
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
    set_num_residuals(num_samples);
    *mutable_parameter_block_sizes() = layout_.block_sizes;
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    const int M = patch_.NumSamples();
    const int num_blocks = static_cast<int>(layout_.block_sizes.size());

    auto SetBarrierAndZeroJacobians = [&]() {
      // Worst-case ZNSSD residual: ||r||^2 = 4.0 (NCC = -1), zero gradient.
      // Prevents Levenberg-Marquardt from being rewarded for pushing patches
      // out of bounds, behind a camera, or into textureless regions.
      const double barrier = 2.0 / std::sqrt(static_cast<double>(M));
      Eigen::Map<Eigen::VectorXd>(residuals, M).setConstant(barrier);
      if (jacobians != nullptr) {
        for (int b = 0; b < num_blocks; ++b) {
          if (jacobians[b] != nullptr) {
            Eigen::Map<Eigen::VectorXd>(
                jacobians[b],
                static_cast<Eigen::Index>(M) * layout_.block_sizes[b])
                .setZero();
          }
        }
      }
      return true;
    };

    const Eigen::Vector3d n =
        EigenVector3Map<double>(parameters[layout_.plane_idx]);
    const span<const double> src_cam(parameters[layout_.src_cam_idx],
                                     layout_.src_num_cam_params);
    const span<const double> tgt_cam(parameters[layout_.tgt_cam_idx],
                                     layout_.tgt_num_cam_params);

    const auto pose = photometric_internal::RelativePoseState::FromParameters(
        layout_, parameters);
    const auto jac = photometric_internal::BlockJacobianPtrs::FromJacobians(
        layout_, jacobians);
    const bool need_any_jac = jac.NeedAny();
    const bool need_J_uvw = jac.NeedJUvw();

    // Step 1: Source unprojection (fast path skips Newton when src_cam is
    // unchanged from level init).
    const span<const double> src_cam_init(patch_.src_cam_params_init.data(),
                                          patch_.src_cam_params_init.size());
    const bool src_cam_changed =
        photometric_internal::HasSourceCamChanged(src_cam, src_cam_init);

    photometric_internal::InlineBuffer<Eigen::Vector3d, kStackMaxM> rays_buf;
    photometric_internal::
        InlineBuffer<double, kStackMaxM * 2 * kPhotometricMaxCamParams>
            d_uv_buf;
    const bool run_unproject = src_cam_changed || jac.src_cam != nullptr;
    const span<const Eigen::Vector3d> init_rays(patch_.src_rays_init.data(), M);
    span<Eigen::Vector3d> rays_out =
        run_unproject ? rays_buf.Resize(M) : span<Eigen::Vector3d>(nullptr, 0);
    span<double> d_uv_d_src_cam =
        jac.src_cam != nullptr ? d_uv_buf.Resize(static_cast<size_t>(M) * 2 *
                                                 layout_.src_num_cam_params)
                               : span<double>(nullptr, 0);

    if (run_unproject) {
      if (!photometric_internal::UnprojectBatchWithJac(
              layout_.src_model_id,
              src_cam,
              span<const double>(patch_.src_xs.data(), M),
              span<const double>(patch_.src_ys.data(), M),
              init_rays,
              !src_cam_changed,
              rays_out,
              d_uv_d_src_cam)) {
        return SetBarrierAndZeroJacobians();
      }
    }
    const span<const Eigen::Vector3d> src_rays_span =
        run_unproject ? span<const Eigen::Vector3d>(rays_out.begin(), M)
                      : init_rays;

    // Step 2: Warp all M patch rays through plane homography
    // H_ji = R_ji + t_ji * n^T.
    photometric_internal::InlineBuffer<Eigen::Vector3d, kStackMaxM> pts_tgt_buf;
    photometric_internal::InlineBuffer<double, kStackMaxM> alphas_buf;
    span<Eigen::Vector3d> pts_tgt = pts_tgt_buf.Resize(M);
    auto alphas_vec = photometric_internal::MapVector(alphas_buf.Resize(M));

    const Eigen::Map<const Eigen::Matrix3Xd> src_rays_mat(
        src_rays_span[0].data(), 3, M);
    alphas_vec.noalias() = src_rays_mat.transpose().lazyProduct(n);
    if (!(alphas_vec.array() > 1e-8).all()) {
      return SetBarrierAndZeroJacobians();
    }
    const Eigen::Matrix3d H_ji = pose.R_ji + pose.t_ji * n.transpose();
    Eigen::Map<Eigen::Matrix3Xd>(pts_tgt[0].data(), 3, M).noalias() =
        H_ji.lazyProduct(src_rays_mat);

    // Step 3: Batch target camera projection (+ analytic camera Jacobians).
    photometric_internal::InlineBuffer<double, kStackMaxM> tgt_xs_buf;
    photometric_internal::InlineBuffer<double, kStackMaxM> tgt_ys_buf;
    photometric_internal::InlineBuffer<double, kStackMaxM> tgt_dxs_buf;
    photometric_internal::InlineBuffer<double, kStackMaxM> tgt_dys_buf;
    photometric_internal::InlineBuffer<double, kStackMaxM> p_norm_buf;
    span<double> tgt_xs = tgt_xs_buf.Resize(M);
    span<double> tgt_ys = tgt_ys_buf.Resize(M);

    photometric_internal::
        InlineBuffer<double, kStackMaxM * 2 * kPhotometricMaxCamParams>
            J_tgt_cam_buf;
    span<double> J_tgt_cam_raw =
        jac.tgt_cam != nullptr
            ? J_tgt_cam_buf.Resize(static_cast<size_t>(M) * 2 *
                                   layout_.tgt_num_cam_params)
            : span<double>(nullptr, 0);

    photometric_internal::InlineBuffer<double, kStackMaxM * 6> J_uvw_buf;
    span<double> J_uvw_raw = need_J_uvw
                                 ? J_uvw_buf.Resize(static_cast<size_t>(M) * 6)
                                 : span<double>(nullptr, 0);

    if (!photometric_internal::ProjectBatch(
            layout_.tgt_model_id,
            tgt_cam,
            span<const Eigen::Vector3d>(pts_tgt.begin(), M),
            tgt_xs,
            tgt_ys,
            J_tgt_cam_raw,
            J_uvw_raw)) {
      return SetBarrierAndZeroJacobians();
    }

    // Step 4 & 5: Bilinear interpolation (+ image gradients) and Least-Squares
    // NCC normalization r = Psi(I_tgt) - s_k.
    auto tgt_dxs = photometric_internal::MapVector(tgt_dxs_buf.Resize(M));
    auto tgt_dys = photometric_internal::MapVector(tgt_dys_buf.Resize(M));
    auto p_norm = photometric_internal::MapVector(p_norm_buf.Resize(M));
    const Eigen::Map<const Eigen::VectorXd> src_norm(patch_.src_norm.data(), M);
    Eigen::Map<Eigen::VectorXd> residuals_vec(residuals, M);
    double inv_sigma = 0.0;

    if (!photometric_internal::InterpolateAndComputeLSNCC(
            target_level_,
            span<const double>(tgt_xs.begin(), M),
            span<const double>(tgt_ys.begin(), M),
            src_norm,
            min_target_norm_,
            need_any_jac,
            tgt_dxs,
            tgt_dys,
            p_norm,
            residuals_vec,
            inv_sigma)) {
      return SetBarrierAndZeroJacobians();
    }

    if (!need_any_jac) {
      return true;
    }

    // Step 6: Chain rule g_y[m] = [dI/dx, dI/dy] * J_uvw[m] (M x 3 row-major)
    // and LS-NCC projection J_Psi * G directly into Ceres Jacobians.
    photometric_internal::InlineBuffer<double, kStackMaxM * 3> gy_buf;
    auto gy_mat = photometric_internal::MapFixedColMatrix<3>(
        gy_buf.Resize(static_cast<size_t>(M) * 3), M);
    if (need_J_uvw) {
      for (int m = 0; m < M; ++m) {
        const Eigen::Map<const photometric_internal::Matrix2x3RowMajor> J_uvw_m(
            J_uvw_raw.begin() + static_cast<size_t>(m) * 6);
        gy_mat.row(m).noalias() =
            Eigen::RowVector2d(tgt_dxs[m], tgt_dys[m]) * J_uvw_m;
      }
    }

    photometric_internal::InlineBuffer<double,
                                       kStackMaxM * kPhotometricMaxCamParams>
        scratch_buf;

    // 6a. Plane Jacobian: d(y_m)/dn = t_ji * src_ray_m^T  (M x 3, in-place).
    if (jac.plane != nullptr) {
      auto J_plane = photometric_internal::MapFixedColMatrix<3>(jac.plane, M);
      auto s_vec = photometric_internal::MapVector(scratch_buf.Resize(M));
      s_vec.noalias() = gy_mat.lazyProduct(pose.t_ji);
      J_plane = src_rays_mat.transpose().array().colwise() * s_vec.array();
      photometric_internal::ProjectLSNCCJacobian(
          J_plane, p_norm, inv_sigma, J_plane, /*accumulate=*/false);
    }

    // 6b-6e. Frame & rig pose Jacobians.
    if (jac.NeedPoses()) {
      ComputePoseJacobians(pose,
                           jac,
                           src_rays_mat,
                           alphas_vec,
                           gy_mat,
                           p_norm,
                           inv_sigma,
                           scratch_buf);
    }

    // 6f. Target camera intrinsics Jacobian.
    if (jac.tgt_cam != nullptr) {
      const int P_tgt = layout_.tgt_num_cam_params;
      auto J_tgt =
          photometric_internal::MapRowMajorMatrix(jac.tgt_cam, M, P_tgt);
      photometric_internal::ChainRuleAndProjectLSNCC(
          tgt_dxs,
          tgt_dys,
          span<const double>(J_tgt_cam_raw.begin(), J_tgt_cam_raw.size()),
          p_norm,
          inv_sigma,
          J_tgt,
          J_tgt,
          /*accumulate=*/false);
    }

    // 6g. Source camera intrinsics Jacobian (via implicit function theorem).
    // Accumulates into the shared block when same_camera is true.
    if (jac.src_cam != nullptr) {
      const int P_src = layout_.src_num_cam_params;
      photometric_internal::InlineBuffer<double, kStackMaxM * 2> guv_buf;
      Eigen::Map<photometric_internal::MatrixMx2ColMajor> G_uv(
          guv_buf.Resize(static_cast<size_t>(M) * 2).begin(), M, 2);
      G_uv.noalias() = gy_mat.lazyProduct(H_ji.leftCols<2>());

      const bool accumulate = layout_.same_camera && (jac.tgt_cam != nullptr);
      auto J_src =
          photometric_internal::MapRowMajorMatrix(jac.src_cam, M, P_src);
      auto g_work =
          accumulate ? photometric_internal::MapRowMajorMatrix(
                           scratch_buf.Resize(static_cast<size_t>(M) * P_src),
                           M,
                           P_src)
                     : J_src;
      photometric_internal::ChainRuleAndProjectLSNCC(
          G_uv.col(0),
          G_uv.col(1),
          span<const double>(d_uv_d_src_cam.begin(), d_uv_d_src_cam.size()),
          p_norm,
          inv_sigma,
          g_work,
          J_src,
          accumulate);
    }

    return true;
  }

 private:
  static constexpr size_t kStackMaxM = 64;

  void ComputePoseJacobians(
      const photometric_internal::RelativePoseState& pose,
      const photometric_internal::BlockJacobianPtrs& jac,
      const Eigen::Ref<const Eigen::Matrix3Xd>& src_rays_mat,
      const Eigen::Ref<const Eigen::VectorXd>& alphas_vec,
      const Eigen::Ref<const photometric_internal::MatrixMx3RowMajor>& gy_mat,
      const Eigen::Ref<const Eigen::VectorXd>& p_norm,
      const double inv_sigma,
      photometric_internal::InlineBuffer<double,
                                         kStackMaxM * kPhotometricMaxCamParams>&
          scratch_buf) const {
    const int M = static_cast<int>(alphas_vec.size());
    const bool need_p_fi = jac.tgt_rfw != nullptr || jac.src_rfw != nullptr ||
                           jac.tgt_sfr != nullptr;
    const bool need_p_w = !layout_.same_frame &&
                          (jac.tgt_rfw != nullptr || jac.tgt_sfr != nullptr);

    photometric_internal::InlineBuffer<double, kStackMaxM * 3> p_fi_buf;
    photometric_internal::InlineBuffer<double, kStackMaxM * 3> p_w_buf;
    photometric_internal::InlineBuffer<double, kStackMaxM * 3> g_pose_buf;
    auto g_pose_mat = photometric_internal::MapFixedColMatrix<3>(
        g_pose_buf.Resize(static_cast<size_t>(M) * 3), M);

    const Eigen::Matrix3d R_si_T = pose.R_si.transpose();
    const Eigen::Matrix3d R_fi_T = pose.R_fi.transpose();

    const Eigen::Ref<const Eigen::Matrix3Xd> p_fi_mat =
        [&]() -> Eigen::Ref<const Eigen::Matrix3Xd> {
      if (!need_p_fi || layout_.src_is_ref_sensor) {
        return src_rays_mat;
      }
      Eigen::Map<Eigen::Matrix3Xd> mat(
          p_fi_buf.Resize(static_cast<size_t>(M) * 3).begin(), 3, M);
      const Eigen::Vector3d t_si_rot = R_si_T * pose.t_si;
      mat = R_si_T.lazyProduct(src_rays_mat) -
            t_si_rot.lazyProduct(alphas_vec.transpose());
      return mat;
    }();

    const Eigen::Ref<const Eigen::Matrix3Xd> p_w_mat =
        [&]() -> Eigen::Ref<const Eigen::Matrix3Xd> {
      if (!need_p_w) {
        return p_fi_mat;
      }
      Eigen::Map<Eigen::Matrix3Xd> mat(
          p_w_buf.Resize(static_cast<size_t>(M) * 3).begin(), 3, M);
      const Eigen::Vector3d t_fi_rot = R_fi_T * pose.t_fi;
      mat = R_fi_T.lazyProduct(p_fi_mat) -
            t_fi_rot.lazyProduct(alphas_vec.transpose());
      return mat;
    }();

    if (jac.tgt_rfw != nullptr) {
      auto J_tgt = photometric_internal::MapFixedColMatrix<7>(jac.tgt_rfw, M);
      const Eigen::Ref<const photometric_internal::MatrixMx3RowMajor> g_tgt =
          [&]() -> Eigen::Ref<const photometric_internal::MatrixMx3RowMajor> {
        if (layout_.tgt_is_ref_sensor) {
          return gy_mat;
        }
        g_pose_mat.noalias() = gy_mat.lazyProduct(pose.R_sj);
        return g_pose_mat;
      }();
      for (int m = 0; m < M; ++m) {
        J_tgt.row(m).head<4>().noalias() =
            g_tgt.row(m) * photometric_internal::QuaternionRotatePointJac(
                               pose.q_fj, p_w_mat.col(m));
        J_tgt.row(m).tail<3>() = alphas_vec[m] * g_tgt.row(m);
      }
      photometric_internal::ProjectLSNCCJacobian(
          J_tgt, p_norm, inv_sigma, J_tgt, /*accumulate=*/false);
    }

    if (jac.src_rfw != nullptr) {
      auto J_src = photometric_internal::MapFixedColMatrix<7>(jac.src_rfw, M);
      g_pose_mat.noalias() = gy_mat.lazyProduct(pose.R_j);
      for (int m = 0; m < M; ++m) {
        const Eigen::Vector3d p_shift =
            p_fi_mat.col(m) - alphas_vec[m] * pose.t_fi;
        J_src.row(m).head<4>().noalias() =
            g_pose_mat.row(m) *
            photometric_internal::QuaternionInverseRotatePointJac(pose.q_fi,
                                                                  p_shift);
        J_src.row(m).tail<3>().noalias() =
            (-alphas_vec[m]) * (g_pose_mat.row(m) * R_fi_T);
      }
      photometric_internal::ProjectLSNCCJacobian(
          J_src, p_norm, inv_sigma, J_src, /*accumulate=*/false);
    }

    if (jac.tgt_sfr != nullptr) {
      auto J_tgt = photometric_internal::MapFixedColMatrix<7>(jac.tgt_sfr, M);
      for (int m = 0; m < M; ++m) {
        const Eigen::Vector3d p_fj =
            layout_.same_frame ? Eigen::Vector3d(p_fi_mat.col(m))
                               : Eigen::Vector3d(pose.R_fj * p_w_mat.col(m) +
                                                 alphas_vec[m] * pose.t_fj);
        J_tgt.row(m).head<4>().noalias() =
            gy_mat.row(m) *
            photometric_internal::QuaternionRotatePointJac(pose.q_sj, p_fj);
        J_tgt.row(m).tail<3>() = alphas_vec[m] * gy_mat.row(m);
      }
      photometric_internal::ProjectLSNCCJacobian(
          J_tgt, p_norm, inv_sigma, J_tgt, /*accumulate=*/false);
    }

    if (jac.src_sfr != nullptr) {
      const bool accumulate = layout_.same_sensor && (jac.tgt_sfr != nullptr);
      auto J_src = photometric_internal::MapFixedColMatrix<7>(jac.src_sfr, M);
      auto G_dst = accumulate
                       ? photometric_internal::MapFixedColMatrix<7>(
                             scratch_buf.Resize(static_cast<size_t>(M) * 7), M)
                       : J_src;
      if (layout_.same_frame) {
        g_pose_mat.noalias() = gy_mat.lazyProduct(pose.R_sj);
      } else {
        g_pose_mat.noalias() = gy_mat.lazyProduct(pose.R_j * R_fi_T);
      }
      for (int m = 0; m < M; ++m) {
        const Eigen::Vector3d p_shift =
            src_rays_mat.col(m) - alphas_vec[m] * pose.t_si;
        G_dst.row(m).head<4>().noalias() =
            g_pose_mat.row(m) *
            photometric_internal::QuaternionInverseRotatePointJac(pose.q_si,
                                                                  p_shift);
        G_dst.row(m).tail<3>().noalias() =
            (-alphas_vec[m]) * (g_pose_mat.row(m) * R_si_T);
      }
      photometric_internal::ProjectLSNCCJacobian(
          G_dst, p_norm, inv_sigma, J_src, accumulate);
    }
  }

  const PhotometricPatchData& patch_;
  const GrayscaleImageLevel& target_level_;
  const photometric_internal::SourceTargetBlockLayout layout_;
  const double min_target_norm_;
};

}  // namespace colmap
