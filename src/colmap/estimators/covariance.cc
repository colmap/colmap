// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/covariance.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/reprojection_error.h"
#include "colmap/util/hash_containers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>

#include <ceres/crs_matrix.h>

namespace colmap {
namespace {

// Point Jacobian singular values below this fraction of the largest singular
// value are truncated during Schur elimination so ill-conditioned or
// rank-deficient points preserve the gauge invariance of the Schur complement.
constexpr double kMinPointSingularValueRatio = 1e-7;

bool ComputeSchurComplement(bool estimate_point_covs,
                            bool compute_schur,
                            double damping,
                            int point_num_params,
                            const std::vector<internal::PointParam>& points,
                            const std::vector<internal::PoseParam>& poses,
                            const std::vector<const double*>& others,
                            ceres::Problem& problem,
                            FlatHashMap<point3D_t, Eigen::MatrixXd>& point_covs,
                            Eigen::SparseMatrix<double>& S,
                            double* total_cost) {
  VLOG(2) << "Evaluating the Jacobian for Schur elimination";

  ceres::Problem::EvaluateOptions eval_options;
  eval_options.parameter_blocks.reserve(
      (compute_schur ? poses.size() + others.size() : 0) + points.size());
  if (compute_schur) {
    for (const auto& pose : poses) {
      eval_options.parameter_blocks.push_back(
          const_cast<double*>(pose.cam_from_world));
    }
    for (const double* other : others) {
      eval_options.parameter_blocks.push_back(const_cast<double*>(other));
    }
  }
  for (const auto& point : points) {
    eval_options.parameter_blocks.push_back(const_cast<double*>(point.xyz));
  }

  if (eval_options.parameter_blocks.empty()) {
    if (total_cost != nullptr) {
      if (!problem.Evaluate(ceres::Problem::EvaluateOptions(),
                            total_cost,
                            nullptr,
                            nullptr,
                            nullptr)) {
        LOG(WARNING) << "Failed to evaluate cost";
        return false;
      }
    }
    S.resize(0, 0);
    return true;
  }

  ceres::CRSMatrix J_full_crs;
  if (!problem.Evaluate(
          eval_options, total_cost, nullptr, nullptr, &J_full_crs)) {
    LOG(WARNING) << "Failed to evaluate Jacobian";
    return false;
  }

  const Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor>> J_full(
      J_full_crs.num_rows,
      J_full_crs.num_cols,
      J_full_crs.values.size(),
      J_full_crs.rows.data(),
      J_full_crs.cols.data(),
      J_full_crs.values.data());

  if (points.empty()) {
    S = J_full.transpose() * J_full;
    return true;
  }

  if (estimate_point_covs) {
    point_covs.reserve(points.size());
  }

  VLOG(2) << "Schur elimination of point parameters (n = " << point_num_params
          << ")";

  // Notice that here "a" refers to pose/other and "p" to point parameters.
  const int a_num_params = J_full.cols() - point_num_params;
  const Eigen::SparseMatrix<double> J_p =
      J_full.block(0, a_num_params, J_full.rows(), point_num_params);
  // Orthogonal left singular vectors U_p of each point's Jacobian block,
  // preserving the sparsity pattern of J_p. Forming S = J_a^T J_a - W^T W with
  // W = U_p^T J_a avoids squaring cond(J_p) via normal equations.
  std::vector<Eigen::Triplet<double>> U_p_triplets;
  if (compute_schur) {
    U_p_triplets.reserve(J_p.nonZeros());
  }

  std::vector<int> rows;
  int point_param_idx = 0;
  for (const internal::PointParam& point : points) {
    const int tangent_size = ParameterBlockTangentSize(problem, point.xyz);
    rows.clear();
    for (Eigen::SparseMatrix<double>::InnerIterator it(J_p, point_param_idx);
         it;
         ++it) {
      rows.push_back(it.row());
    }

    if (rows.empty()) {
      if (estimate_point_covs) {
        point_covs.emplace(point.point3D_id,
                           (1.0 / damping) * Eigen::MatrixXd::Identity(
                                                 tangent_size, tangent_size));
      }
      point_param_idx += tangent_size;
      continue;
    }

    Eigen::MatrixXd J_pi(rows.size(), tangent_size);
    for (int c = 0; c < tangent_size; ++c) {
      size_t row_pos = 0;
      for (Eigen::SparseMatrix<double>::InnerIterator it(J_p,
                                                         point_param_idx + c);
           it;
           ++it, ++row_pos) {
        THROW_CHECK_LT(row_pos, rows.size());
        THROW_CHECK_EQ(it.row(), rows[row_pos]);
        J_pi(row_pos, c) = it.value();
      }
      THROW_CHECK_EQ(row_pos, rows.size());
    }

    const unsigned int svd_options =
        (compute_schur ? Eigen::ComputeThinU : 0) |
        (estimate_point_covs ? Eigen::ComputeFullV : 0);
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(J_pi, svd_options);
    const Eigen::VectorXd& singular_values = svd.singularValues();

    if (compute_schur && singular_values(0) > 0) {
      const Eigen::MatrixXd& U = svd.matrixU();
      const double min_singular_value =
          kMinPointSingularValueRatio * singular_values(0);
      for (int r = 0; r < singular_values.size(); ++r) {
        if (singular_values(r) <= min_singular_value) {
          break;
        }
        for (size_t j = 0; j < rows.size(); ++j) {
          U_p_triplets.emplace_back(rows[j], point_param_idx + r, U(j, r));
        }
      }
    }

    if (estimate_point_covs) {
      // Point covariance conditioned on fixed pose/other parameters.
      const Eigen::MatrixXd& V = svd.matrixV();
      Eigen::VectorXd inv_eigs(tangent_size);
      for (int r = 0; r < tangent_size; ++r) {
        const double s = r < singular_values.size() ? singular_values(r) : 0.0;
        inv_eigs(r) = 1.0 / (s * s + damping);
      }
      point_covs.emplace(point.point3D_id,
                         V * inv_eigs.asDiagonal() * V.transpose());
    }
    point_param_idx += tangent_size;
  }

  if (!compute_schur) {
    return true;
  }

  Eigen::SparseMatrix<double> U_p(J_full.rows(), point_num_params);
  U_p.setFromTriplets(U_p_triplets.begin(), U_p_triplets.end());
  const Eigen::SparseMatrix<double> J_a =
      J_full.block(0, 0, J_full.rows(), a_num_params);
  const Eigen::SparseMatrix<double> W = U_p.transpose() * J_a;
  S = J_a.transpose() * J_a - W.transpose() * W;

  return true;
}

bool SchurEliminateOtherParams(double damping,
                               int pose_num_params,
                               int other_num_params,
                               Eigen::SparseMatrix<double>& S) {
  VLOG(2) << "Schur elimination of other parameters (n = " << other_num_params
          << ")";

  // Notice that here "c" refers to pose and "o" to other parameters.
  const Eigen::SparseMatrix<double> S_cc =
      S.block(0, 0, pose_num_params, pose_num_params);
  const Eigen::SparseMatrix<double> S_co =
      S.block(0, pose_num_params, pose_num_params, other_num_params);
  const Eigen::SparseMatrix<double> S_oc = S_co.transpose();
  Eigen::SparseMatrix<double> S_oo = S.block(
      pose_num_params, pose_num_params, other_num_params, other_num_params);
  for (int i = 0; i < other_num_params; ++i) {
    if (S_oo.coeff(i, i) == 0.0) {
      S_oo.coeffRef(i, i) = damping;
    } else {
      S_oo.coeffRef(i, i) += damping;
    }
  }

  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt_S_oo(S_oo);
  if (llt_S_oo.info() != Eigen::Success) {
    LOG(WARNING)
        << "Simplicial LLT for Schur elimination of other parameters failed";
    return false;
  }

  S = S_cc - S_co * llt_S_oo.solve(S_oc);

  return true;
}

// Eigenvalues of the Jacobi-scaled Schur complement (unit diagonal) below this
// threshold are considered zero. Gauge eigenvalues are at round-off level,
// since the points are eliminated without damping.
constexpr double kMaxNullEigenvalue = 1e-12;
// Minimum spectral gap above kMaxNullEigenvalue required for the smallest
// non-null eigenvalue when a gauge null space is present.
constexpr double kMinNullSpectralGap = 1e2;
// Regularization of the Jacobi-scaled Schur complement for the inverse
// iteration and the pseudo-inverse, above the round-off of its null space and
// well below the null space threshold. It biases the covariance by a relative
// O(kScaledRegularization / eigenvalue).
constexpr double kScaledRegularization = 1e-13;
// Gauge null vectors of the Jacobi-scaled Schur complement (unit norm) must
// have components below this threshold on the other parameters; gauge-dependent
// other parameters (e.g., rig extrinsics) are not supported and are rejected.
constexpr double kMaxGaugeOtherParamsNorm = 1e-6;
// Maximum estimated relative error from mapping the scaled gauge null space
// back to unscaled coordinates and projecting out the W = I minimum-trace
// gauge (Davis-Kahan bound: (||S_s * N_s|| / gap) * diag_ratio^2). The bound is
// worst-case conservative by several orders of magnitude, so 1e-1 accepts real
// scenes with unnormalized coordinates or weakly constrained cameras while
// still rejecting large world offsets and extreme scene scales.
constexpr double kMaxGaugeProjectionError = 1e-1;
// A bundle adjustment problem without any constraints on the gauge has a
// 7-DoF gauge freedom (Sim3).
constexpr int kMaxGaugeDim = 7;

// Grid dimensions and minimum observations per tile for the 3x2 spatial-tile
// design effect estimator.
constexpr int kSpatialGridLong = 3;
constexpr int kSpatialGridShort = 2;
constexpr int kNumSpatialTiles = kSpatialGridLong * kSpatialGridShort;
constexpr int kMinObsPerSpatialTile = 5;

struct ImageTileSpec {
  image_t image_id = kInvalidImageId;
  int pose_tangent_size = 0;
  // False for external images that only entered the problem via
  // config->VariablePoints() / config->ConstantPoints().
  bool all_image_observations = true;
};

std::vector<ImageTileSpec> CollectSpatialDeffImageSpecs(
    const Reconstruction& reconstruction,
    const ceres::Problem& problem,
    const std::vector<internal::PoseParam>& poses,
    const BundleAdjustmentConfig* config,
    bool has_custom_poses) {
  std::vector<ImageTileSpec> image_specs;
  if (config != nullptr) {
    FlatHashMap<image_t, int> variable_pose_dofs;
    variable_pose_dofs.reserve(poses.size());
    for (const auto& pose : poses) {
      variable_pose_dofs.emplace(
          pose.image_id,
          ParameterBlockTangentSize(problem, pose.cam_from_world));
    }

    FlatHashSet<image_t> added_images;
    added_images.reserve(config->NumImages());
    image_specs.reserve(config->NumImages());
    for (const image_t image_id : config->Images()) {
      if (!reconstruction.ExistsImage(image_id)) {
        continue;
      }
      const auto it = variable_pose_dofs.find(image_id);
      const int pose_dof = it != variable_pose_dofs.end() ? it->second : 0;
      image_specs.push_back({image_id, pose_dof, true});
      added_images.insert(image_id);
    }

    if (config->NumPoints() > 0) {
      auto add_point_track_images =
          [&](const FlatHashSet<point3D_t>& point_ids) {
            for (const point3D_t point3D_id : point_ids) {
              if (config->IsIgnoredPoint(point3D_id) ||
                  !reconstruction.ExistsPoint3D(point3D_id)) {
                continue;
              }
              const Point3D& point3D = reconstruction.Point3D(point3D_id);
              if (!problem.HasParameterBlock(point3D.xyz.data())) {
                continue;
              }
              for (const TrackElement& track_el : point3D.track.Elements()) {
                if (added_images.insert(track_el.image_id).second &&
                    reconstruction.ExistsImage(track_el.image_id)) {
                  image_specs.push_back({track_el.image_id, 0, false});
                }
              }
            }
          };
      add_point_track_images(config->VariablePoints());
      add_point_track_images(config->ConstantPoints());
    }
    return image_specs;
  }

  if (has_custom_poses) {
    image_specs.reserve(poses.size());
    for (const auto& pose : poses) {
      if (!reconstruction.ExistsImage(pose.image_id)) {
        continue;
      }
      image_specs.push_back(
          {pose.image_id,
           ParameterBlockTangentSize(problem, pose.cam_from_world),
           true});
    }
    return image_specs;
  }

  const std::vector<image_t> reg_image_ids = reconstruction.RegImageIds();
  image_specs.reserve(reg_image_ids.size());
  for (const image_t image_id : reg_image_ids) {
    const Image& image = reconstruction.Image(image_id);
    if (!image.HasPose()) {
      continue;
    }
    const Rigid3d& rig_from_world = image.FramePtr()->RigFromWorld();
    if (problem.HasParameterBlock(rig_from_world.params.data())) {
      const int pose_dof = problem.IsParameterBlockConstant(const_cast<double*>(
                               rig_from_world.params.data()))
                               ? 0
                               : ParameterBlockTangentSize(
                                     problem, rig_from_world.params.data());
      image_specs.push_back({image_id, pose_dof, true});
    }
  }
  if (image_specs.empty()) {
    // All poses were baked into constant-pose cost functors, so no pose
    // parameter blocks exist in the problem.
    for (const image_t image_id : reg_image_ids) {
      const Image& image = reconstruction.Image(image_id);
      if (problem.HasParameterBlock(image.CameraPtr()->params.data())) {
        image_specs.push_back({image_id, 0, true});
      }
    }
  }
  return image_specs;
}

// Estimates the within-image spatial design effect (DEFF >= 1.0) from 3x2
// spatial tile means of reprojection residuals. Under i.i.d. noise, 3D point
// elimination attenuates each residual's variance equally to raw_res_var, and
// each tile mean of n_t distinct-point observations has variance raw_res_var /
// n_t (minus 0.5 * pose_tangent_size tile-pair DoFs absorbed by the image's
// pose). Positively correlated spatial errors inflate the tile-mean quadratic
// form by DEFF = 1 + (n_tile - 1) * rho_bar, preventing covariance collapse
// with growing observation density.
std::optional<double> ComputeSpatialDesignEffect(
    const Reconstruction& reconstruction,
    const ceres::Problem& problem,
    const std::vector<internal::PoseParam>& poses,
    const BundleAdjustmentConfig* config,
    bool has_custom_poses) {
  const std::vector<ImageTileSpec> image_specs = CollectSpatialDeffImageSpecs(
      reconstruction, problem, poses, config, has_custom_poses);

  double total_res_sq = 0.0;
  int64_t total_obs = 0;
  double total_tile_quad = 0.0;
  double total_tile_dof = 0.0;

  for (const ImageTileSpec& spec : image_specs) {
    const Image& image = reconstruction.Image(spec.image_id);
    if (!image.HasPose()) {
      continue;
    }
    const Camera& camera = *image.CameraPtr();
    if (camera.width == 0 || camera.height == 0) {
      continue;
    }

    const int num_tiles_x =
        camera.width >= camera.height ? kSpatialGridLong : kSpatialGridShort;
    const int num_tiles_y =
        camera.width >= camera.height ? kSpatialGridShort : kSpatialGridLong;

    std::array<Eigen::Vector2d, kNumSpatialTiles> tile_res_sum;
    tile_res_sum.fill(Eigen::Vector2d::Zero());
    std::array<double, kNumSpatialTiles> tile_res_sq;
    tile_res_sq.fill(0.0);
    std::array<int, kNumSpatialTiles> tile_obs_count;
    tile_obs_count.fill(0);

    const Rigid3d cam_from_world = image.CamFromWorld();
    const double inv_width = static_cast<double>(num_tiles_x) / camera.width;
    const double inv_height = static_cast<double>(num_tiles_y) / camera.height;

    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D()) {
        continue;
      }
      if (config != nullptr) {
        if (config->IsIgnoredPoint(point2D.point3D_id)) {
          continue;
        }
        if (!spec.all_image_observations &&
            !config->HasPoint(point2D.point3D_id)) {
          continue;
        }
      }
      if (!std::isfinite(point2D.xy.x()) || !std::isfinite(point2D.xy.y())) {
        continue;
      }
      const Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
      if (!problem.HasParameterBlock(point3D.xyz.data())) {
        continue;
      }
      const std::optional<Eigen::Vector2d> proj =
          camera.ImgFromCam(cam_from_world * point3D.xyz);
      if (!proj.has_value()) {
        continue;
      }
      Eigen::Vector2d residual = *proj - point2D.xy;
      if (camera.model_id == CameraModelId::kEquirectangular) {
        WrapEquirectangularHorizontalSeam<EquirectangularCameraModel>(
            camera.params.data(), residual.data());
      }
      if (!residual.allFinite()) {
        continue;
      }

      const int gx =
          static_cast<int>(std::clamp(std::floor(point2D.xy.x() * inv_width),
                                      0.0,
                                      static_cast<double>(num_tiles_x - 1)));
      const int gy =
          static_cast<int>(std::clamp(std::floor(point2D.xy.y() * inv_height),
                                      0.0,
                                      static_cast<double>(num_tiles_y - 1)));
      const int tile_idx = gy * num_tiles_x + gx;
      tile_res_sum[tile_idx] += residual;
      tile_res_sq[tile_idx] += residual.squaredNorm();
      tile_obs_count[tile_idx] += 1;
    }

    int num_populated_tiles = 0;
    double image_tile_quad = 0.0;
    double image_res_sq = 0.0;
    int64_t image_obs = 0;
    for (int t = 0; t < kNumSpatialTiles; ++t) {
      if (tile_obs_count[t] >= kMinObsPerSpatialTile) {
        num_populated_tiles += 1;
        image_tile_quad +=
            0.5 * tile_res_sum[t].squaredNorm() / tile_obs_count[t];
        image_res_sq += tile_res_sq[t];
        image_obs += tile_obs_count[t];
      }
    }

    const double image_tile_dof =
        num_populated_tiles - 0.5 * static_cast<double>(spec.pose_tangent_size);
    if (image_tile_dof >= 1.0) {
      total_tile_quad += image_tile_quad;
      total_tile_dof += image_tile_dof;
      total_res_sq += image_res_sq;
      total_obs += image_obs;
    }
  }

  if (total_obs == 0 || total_tile_dof <= 0.0) {
    LOG(WARNING) << "Unable to estimate spatial design effect. Insufficient "
                    "observations per 3x2 image tile.";
    return std::nullopt;
  }

  const double raw_res_var = total_res_sq / (2.0 * total_obs);
  if (!std::isfinite(raw_res_var) || raw_res_var <= 0.0) {
    LOG(WARNING) << StringPrintf(
        "Unable to estimate spatial design effect. Non-positive residual "
        "variance (%e).",
        raw_res_var);
    return std::nullopt;
  }

  const double raw_deff = total_tile_quad / (raw_res_var * total_tile_dof);
  if (!std::isfinite(raw_deff)) {
    LOG(WARNING) << "Unable to estimate spatial design effect (non-finite).";
    return std::nullopt;
  }
  return std::max(1.0, raw_deff);
}

// Returns L_inv such that L_inv^T * L_inv = S^-1, given S = P^T L D L^T P.
Eigen::MatrixXd LInverseFromLDLT(
    const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& ldlt_S) {
  const Eigen::SparseMatrix<double> L_sparse = ldlt_S.matrixL();
  Eigen::MatrixXd L_inv =
      Eigen::MatrixXd::Identity(L_sparse.rows(), L_sparse.cols());
  L_sparse.triangularView<Eigen::Lower>().solveInPlace(L_inv);
  L_inv = ldlt_S.vectorD().cwiseSqrt().cwiseInverse().asDiagonal() * L_inv;
  L_inv *= ldlt_S.permutationP();
  return L_inv;
}

// Computes L_inv such that L_inv^T * L_inv = S^-1 for a positive definite S.
bool ComputeLInverseFullRank(const Eigen::SparseMatrix<double>& S,
                             Eigen::MatrixXd& L_inv) {
  VLOG(2) << "Start sparse Cholesky decomposition (n = " << S.rows() << ")";
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_S(S);
  if (ldlt_S.info() != Eigen::Success ||
      !(ldlt_S.vectorD().array() > 0).all()) {
    LOG(WARNING) << "Simplicial LDLT for computing L_inv failed";
    return false;
  }
  VLOG(2) << "Finish sparse Cholesky decomposition.";
  L_inv = LInverseFromLDLT(ldlt_S);
  VLOG(2) << "Finish factorization.";
  return true;
}

// Computes L_inv such that L_inv^T * L_inv is the inverse of S or, if the
// problem leaves the gauge free, its pseudo-inverse, i.e., the covariance in
// the minimum-trace gauge (Förstner & Wrobel 2016, Eq. 4.224 with W = I).
// The null space is detected on the Jacobi-scaled S_s = D^-1/2 S D^-1/2,
// which is independent of the parameter units and the scene scale.
bool ComputeLInverse(const Eigen::SparseMatrix<double>& S,
                     int pose_num_params,
                     Eigen::MatrixXd& L_inv,
                     int* num_null_out) {
  *num_null_out = 0;
  const int n = S.rows();
  const Eigen::VectorXd S_diag = S.diagonal();
  if ((S_diag.array() <= 0).any()) {
    LOG(WARNING) << "Unable to compute covariance. Some pose/other parameters "
                    "are not constrained by any residual.";
    return false;
  }
  const Eigen::VectorXd D_inv_sqrt = S_diag.cwiseSqrt().cwiseInverse();
  const Eigen::SparseMatrix<double> S_s =
      D_inv_sqrt.asDiagonal() * S * D_inv_sqrt.asDiagonal();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_S_s_damped;
  ldlt_S_s_damped.setShift(kScaledRegularization).compute(S_s);
  if (ldlt_S_s_damped.info() != Eigen::Success ||
      !(ldlt_S_s_damped.vectorD().array() > 0).all()) {
    LOG(WARNING) << "Simplicial LDLT of the damped Schur complement failed";
    return false;
  }

  // Block inverse iteration for the smallest eigenvalues of S_s, with one more
  // vector than the gauge freedom to detect larger null spaces, followed by
  // Rayleigh-Ritz.
  const int block_size = std::min(kMaxGaugeDim + 1, n);
  std::mt19937 prng(0);
  std::normal_distribution<double> normal;
  Eigen::MatrixXd X = Eigen::MatrixXd::NullaryExpr(
      n, block_size, [&]() { return normal(prng); });
  for (int i = 0; i < 3; ++i) {
    X = Eigen::HouseholderQR<Eigen::MatrixXd>(ldlt_S_s_damped.solve(X))
            .householderQ() *
        Eigen::MatrixXd::Identity(n, block_size);
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ritz(X.transpose() *
                                                            (S_s * X));
  VLOG(2) << "Smallest eigenvalues of the scaled Schur complement: "
          << ritz.eigenvalues().transpose();
  const int num_null =
      (ritz.eigenvalues().array() <= kMaxNullEigenvalue).count();
  if (num_null == 0) {
    return ComputeLInverseFullRank(S, L_inv);
  }
  if (num_null > kMaxGaugeDim || num_null == n) {
    LOG(WARNING) << StringPrintf(
        "Unable to compute covariance. The Schur complement on pose/other "
        "parameters has a null space of dimension %d or more, which exceeds "
        "the gauge freedom. This is likely due to the pose/other parameters "
        "being underconstrained or other degeneracies.",
        num_null);
    return false;
  }
  if (ritz.eigenvalues()(num_null) < kMinNullSpectralGap * kMaxNullEigenvalue) {
    LOG(WARNING) << StringPrintf(
        "Unable to compute covariance. The Schur complement on pose/other "
        "parameters has an ambiguous near-zero eigenvalue (%e) that is not "
        "clearly separated from the gauge null space.",
        ritz.eigenvalues()(num_null));
    return false;
  }

  // Eigenvalues are sorted in increasing order. Only the null-space dimension
  // (<= kMaxGaugeDim) and absence of other-parameter support are checked so
  // ComputeLInverse stays independent of pose parameterizations (pose-only
  // degeneracies of dimension <= 7 are therefore treated as gauge freedoms).
  const Eigen::MatrixXd N_s = X * ritz.eigenvectors().leftCols(num_null);
  if (N_s.bottomRows(n - pose_num_params).norm() > kMaxGaugeOtherParamsNorm) {
    LOG(WARNING) << StringPrintf(
        "Unable to compute covariance. The Schur complement on pose/other "
        "parameters has a null space of dimension %d that involves the other "
        "parameters and is thus not a gauge freedom.",
        num_null);
    return false;
  }
  VLOG(2) << "Gauge freedom of dimension " << num_null;

  // Mapping N_s to unscaled pose coordinates amplifies its subspace error by
  // the pose diagonal scale ratio, and unscaled projection I - N * N^T can
  // leak the larger pose block into the smaller one by another factor of that
  // ratio. Reject ill-conditioned projections (e.g., extreme scene scales or
  // large world offsets) instead of returning corrupted covariances.
  const Eigen::VectorXd pose_D_inv_sqrt = D_inv_sqrt.head(pose_num_params);
  const double diag_ratio =
      pose_D_inv_sqrt.maxCoeff() / pose_D_inv_sqrt.minCoeff();
  const double null_residual =
      std::max(std::numeric_limits<double>::epsilon(), (S_s * N_s).norm());
  const double est_projection_error =
      (null_residual / ritz.eigenvalues()(num_null)) * diag_ratio * diag_ratio;
  if (est_projection_error > kMaxGaugeProjectionError) {
    LOG(WARNING) << StringPrintf(
        "Unable to compute covariance. Unscaled gauge projection is too "
        "ill-conditioned (estimated relative error %.2e, pose diagonal ratio "
        "%.2e, smallest non-null eigenvalue %.2e). Consider normalizing the "
        "reconstruction scale and origin, or fixing the gauge.",
        est_projection_error,
        diag_ratio,
        ritz.eigenvalues()(num_null));
    return false;
  }

  // Project out N_s in scaled coordinates (two passes) before unscaling,
  // eliminating the O(1 / sqrt(reg)) regularized null-space spike on both pose
  // and other parameters.
  L_inv = LInverseFromLDLT(ldlt_S_s_damped);
  for (int i = 0; i < 2; ++i) {
    const Eigen::MatrixXd L_N = L_inv * N_s;
    L_inv.noalias() -= L_N * N_s.transpose();
  }

  // Null space of S on the pose block, orthonormal in unscaled tangent
  // coordinates.
  const Eigen::MatrixXd N_pose =
      Eigen::JacobiSVD<Eigen::MatrixXd>(
          pose_D_inv_sqrt.asDiagonal() * N_s.topRows(pose_num_params),
          Eigen::ComputeThinU)
          .matrixU();

  // P * D^-1/2 * S_s^+ * D^-1/2 * P with P = I - N * N^T is the pseudo-inverse
  // of S up to O(reg): D^-1/2 * S_s^+ * D^-1/2 is a generalized inverse of S.
  L_inv *= D_inv_sqrt.asDiagonal();
  const Eigen::MatrixXd L_pose_N = L_inv.leftCols(pose_num_params) * N_pose;
  L_inv.leftCols(pose_num_params).noalias() -= L_pose_N * N_pose.transpose();
  *num_null_out = num_null;
  return true;
}

Eigen::MatrixXd ExtractCovFromLInverse(const Eigen::MatrixXd& L_inv,
                                       int row_start,
                                       int col_start,
                                       int row_block_size,
                                       int col_block_size) {
  return L_inv.block(0, row_start, L_inv.rows(), row_block_size).transpose() *
         L_inv.block(0, col_start, L_inv.cols(), col_block_size);
}

}  // namespace

BACovariance::BACovariance(
    FlatHashMap<point3D_t, Eigen::MatrixXd> point_covs,
    NodeHashMap<image_t, std::pair<int, int>> pose_L_start_size,
    NodeHashMap<const double*, std::pair<int, int>> other_L_start_size,
    Eigen::MatrixXd L_inv,
    double posterior_variance_factor,
    double spatial_design_effect)
    : point_covs_(std::move(point_covs)),
      pose_L_start_size_(std::move(pose_L_start_size)),
      other_L_start_size_(std::move(other_L_start_size)),
      L_inv_(std::move(L_inv)),
      posterior_variance_factor_(posterior_variance_factor),
      spatial_design_effect_(spatial_design_effect) {}

double BACovariance::PosteriorVarianceFactor() const {
  return posterior_variance_factor_;
}

double BACovariance::SpatialDesignEffect() const {
  return spatial_design_effect_;
}

double BACovariance::VarianceFactor() const {
  return posterior_variance_factor_ * spatial_design_effect_;
}

std::optional<Eigen::MatrixXd> BACovariance::GetPointCov(
    point3D_t point3D_id) const {
  const auto it = point_covs_.find(point3D_id);
  if (it == point_covs_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<Eigen::MatrixXd> BACovariance::GetCamCovFromWorld(
    image_t image_id) const {
  const auto it = pose_L_start_size_.find(image_id);
  if (it == pose_L_start_size_.end()) {
    return std::nullopt;
  }
  const auto [start, size] = it->second;
  return ExtractCovFromLInverse(L_inv_, start, start, size, size);
}

std::optional<Eigen::MatrixXd> BACovariance::GetCamCrossCovFromWorld(
    image_t image_id1, image_t image_id2) const {
  const auto it1 = pose_L_start_size_.find(image_id1);
  const auto it2 = pose_L_start_size_.find(image_id2);
  if (it1 == pose_L_start_size_.end() || it2 == pose_L_start_size_.end()) {
    return std::nullopt;
  }
  const auto [start1, size1] = it1->second;
  const auto [start2, size2] = it2->second;
  return ExtractCovFromLInverse(L_inv_, start1, start2, size1, size2);
}

std::optional<Eigen::MatrixXd> BACovariance::GetCam2CovFromCam1(
    image_t image_id1,
    const Rigid3d& cam1_from_world,
    image_t image_id2,
    const Rigid3d& cam2_from_world) const {
  auto cov_11 = GetCamCovFromWorld(image_id1);
  if (!cov_11.has_value()) return std::nullopt;
  if (cov_11->rows() != 6) {
    LOG(WARNING) << "cam1_from_world is not fully in the problem. This is "
                    "likely due to the pose being set (partially) constant. ";
    return std::nullopt;
  }
  auto cov_22 = GetCamCovFromWorld(image_id2);
  if (!cov_22.has_value()) return std::nullopt;
  if (cov_22->rows() != 6) {
    LOG(WARNING) << "cam2_from_world is not fully in the problem. This is "
                    "likely due to the pose being set (partially) constant. ";
    return std::nullopt;
  }
  auto cov_12 = GetCamCrossCovFromWorld(image_id1, image_id2);
  THROW_CHECK(cov_12.has_value());
  Eigen::Matrix<double, 12, 12> cov;
  cov.block<6, 6>(0, 0) = *cov_11;
  cov.block<6, 6>(6, 6) = *cov_22;
  cov.block<6, 6>(0, 6) = *cov_12;
  cov.block<6, 6>(6, 0) = cov_12->transpose();
  return GetCovarianceForRelativeRigid3d(cam1_from_world, cam2_from_world, cov);
}

std::optional<Eigen::MatrixXd> BACovariance::GetOtherParamsCov(
    const double* params) const {
  const auto it = other_L_start_size_.find(params);
  if (it == other_L_start_size_.end()) {
    return std::nullopt;
  }
  const auto [start, size] = it->second;
  return ExtractCovFromLInverse(L_inv_, start, start, size, size);
}

namespace {

std::optional<BACovariance> EstimateBACovarianceImpl(
    const BACovarianceOptions& options,
    const Reconstruction& reconstruction,
    ceres::Problem& problem,
    const BundleAdjustmentConfig* config) {
  const bool estimate_point_covs =
      options.params == BACovarianceOptions::Params::POINTS ||
      options.params == BACovarianceOptions::Params::POSES_AND_POINTS ||
      options.params == BACovarianceOptions::Params::ALL;
  const bool estimate_pose_covs =
      options.params == BACovarianceOptions::Params::POSES ||
      options.params == BACovarianceOptions::Params::POSES_AND_POINTS ||
      options.params == BACovarianceOptions::Params::ALL;
  const bool estimate_other_covs =
      options.params == BACovarianceOptions::Params::ALL;

  const std::vector<internal::PointParam> points =
      internal::GetPointParams(reconstruction, problem);
  const bool has_custom_poses = !options.experimental_custom_poses.empty();
  const std::vector<internal::PoseParam>& poses =
      has_custom_poses ? options.experimental_custom_poses
                       : internal::GetPoseParams(reconstruction, problem);
  const std::vector<const double*> others =
      GetOtherParams(problem, poses, points);

  int point_num_params = 0;
  int pose_num_params = 0;
  int other_num_params = 0;
  NodeHashMap<image_t, std::pair<int, int>> pose_L_start_size;
  NodeHashMap<const double*, std::pair<int, int>> other_L_start_size;
  for (const auto& point : points) {
    point_num_params += ParameterBlockTangentSize(problem, point.xyz);
  }
  if (estimate_pose_covs) {
    pose_L_start_size.reserve(poses.size());
  }
  if (estimate_other_covs) {
    other_L_start_size.reserve(others.size());
  }
  for (const auto& pose : poses) {
    const int num_params =
        ParameterBlockTangentSize(problem, pose.cam_from_world);
    if (estimate_pose_covs) {
      pose_L_start_size.emplace(pose.image_id,
                                std::make_pair(pose_num_params, num_params));
    }
    pose_num_params += num_params;
  }
  for (const double* other : others) {
    const int num_params = ParameterBlockTangentSize(problem, other);
    if (estimate_other_covs) {
      other_L_start_size.emplace(
          other,
          std::make_pair(pose_num_params + other_num_params, num_params));
    }
    other_num_params += num_params;
  }

  const bool scale_variance =
      options.variance_scaling != BACovarianceOptions::VarianceScaling::NONE;
  double total_cost = 0.0;
  const bool compute_pose_or_other_covs =
      (estimate_pose_covs && pose_num_params > 0) ||
      (estimate_other_covs && other_num_params > 0);
  FlatHashMap<point3D_t, Eigen::MatrixXd> point_covs;
  Eigen::SparseMatrix<double> S;
  if (!compute_pose_or_other_covs && (!estimate_point_covs || points.empty())) {
    if (scale_variance && !problem.Evaluate(ceres::Problem::EvaluateOptions(),
                                            &total_cost,
                                            nullptr,
                                            nullptr,
                                            nullptr)) {
      LOG(WARNING) << "Failed to evaluate cost";
      return std::nullopt;
    }
  } else if (!ComputeSchurComplement(estimate_point_covs,
                                     compute_pose_or_other_covs,
                                     options.damping,
                                     point_num_params,
                                     points,
                                     poses,
                                     others,
                                     problem,
                                     point_covs,
                                     S,
                                     scale_variance ? &total_cost : nullptr)) {
    return std::nullopt;
  }

  Eigen::MatrixXd L_inv;
  int num_null = 0;
  if (compute_pose_or_other_covs) {
    if (!estimate_other_covs && other_num_params > 0) {
      if (!SchurEliminateOtherParams(
              options.damping, pose_num_params, other_num_params, S)) {
        return std::nullopt;
      }
    }

    VLOG(2) << "Computing L inverse";

    if (!ComputeLInverse(S, pose_num_params, L_inv, &num_null)) {
      return std::nullopt;
    }
  }

  double posterior_variance_factor = 1.0;
  double spatial_design_effect = 1.0;
  if (scale_variance) {
    const int total_num_params =
        point_num_params + pose_num_params + other_num_params;
    const int redundancy =
        problem.NumResiduals() - (total_num_params - num_null);
    if (redundancy <= 0 || !std::isfinite(total_cost) || total_cost < 0.0) {
      LOG(WARNING) << StringPrintf(
          "Unable to compute posterior variance factor (cost=%e, "
          "num_residuals=%d, num_params=%d, num_null=%d).",
          total_cost,
          problem.NumResiduals(),
          total_num_params,
          num_null);
      return std::nullopt;
    }
    posterior_variance_factor = (2.0 / redundancy) * total_cost;
    if (!std::isfinite(posterior_variance_factor)) {
      LOG(WARNING) << StringPrintf(
          "Unable to compute posterior variance factor (overflow: cost=%e, "
          "redundancy=%d).",
          total_cost,
          redundancy);
      return std::nullopt;
    }

    if (options.variance_scaling ==
        BACovarianceOptions::VarianceScaling::SPATIAL_DEFF) {
      const std::optional<double> deff = ComputeSpatialDesignEffect(
          reconstruction, problem, poses, config, has_custom_poses);
      if (!deff.has_value()) {
        return std::nullopt;
      }
      spatial_design_effect = *deff;
    }

    const double pose_variance_factor =
        posterior_variance_factor * spatial_design_effect;
    if (!std::isfinite(pose_variance_factor)) {
      LOG(WARNING) << StringPrintf(
          "Unable to scale covariance (non-finite variance factor: "
          "sigma0_sq=%e, deff=%e).",
          posterior_variance_factor,
          spatial_design_effect);
      return std::nullopt;
    }

    if (posterior_variance_factor != 1.0) {
      for (auto& [point3D_id, cov] : point_covs) {
        cov *= posterior_variance_factor;
      }
    }
    if (pose_variance_factor != 1.0 && L_inv.size() > 0) {
      L_inv *= std::sqrt(pose_variance_factor);
    }
  }

  return BACovariance(std::move(point_covs),
                      compute_pose_or_other_covs
                          ? std::move(pose_L_start_size)
                          : NodeHashMap<image_t, std::pair<int, int>>{},
                      compute_pose_or_other_covs
                          ? std::move(other_L_start_size)
                          : NodeHashMap<const double*, std::pair<int, int>>{},
                      std::move(L_inv),
                      posterior_variance_factor,
                      spatial_design_effect);
}

}  // namespace

std::optional<BACovariance> EstimateBACovariance(
    const BACovarianceOptions& options,
    const Reconstruction& reconstruction,
    CeresBundleAdjuster& bundle_adjuster) {
  ceres::Problem& problem = *THROW_CHECK_NOTNULL(bundle_adjuster.Problem());
  if (options.variance_scaling != BACovarianceOptions::VarianceScaling::NONE &&
      bundle_adjuster.Options().ceres != nullptr &&
      bundle_adjuster.Options().ceres->loss_function_type !=
          CeresLossFunctionType::TRIVIAL) {
    LOG(WARNING)
        << "Variance scaling assumes quadratic (TRIVIAL) residuals; results "
           "may be distorted under non-trivial robust loss functions.";
  }
  return EstimateBACovarianceImpl(
      options, reconstruction, problem, &bundle_adjuster.Config());
}

std::optional<BACovariance> EstimateBACovarianceFromProblem(
    const BACovarianceOptions& options,
    const Reconstruction& reconstruction,
    ceres::Problem& problem) {
  return EstimateBACovarianceImpl(
      options, reconstruction, problem, /*config=*/nullptr);
}

namespace internal {

std::vector<PoseParam> GetPoseParams(const Reconstruction& reconstruction,
                                     const ceres::Problem& problem) {
  std::vector<PoseParam> params;
  params.reserve(reconstruction.NumImages());
  for (const auto& [image_id, image] : reconstruction.Images()) {
    // TODO(jsch): Add support for non-trivial frames.
    THROW_CHECK(image.IsRefInFrame());
    const Rigid3d& cam_from_world = image.FramePtr()->RigFromWorld();
    if (problem.HasParameterBlock(cam_from_world.params.data()) &&
        !problem.IsParameterBlockConstant(
            const_cast<double*>(cam_from_world.params.data()))) {
      params.push_back({image_id, cam_from_world.params.data()});
    }
  }
  return params;
}

std::vector<PointParam> GetPointParams(const Reconstruction& reconstruction,
                                       const ceres::Problem& problem) {
  std::vector<PointParam> params;
  params.reserve(reconstruction.NumPoints3D());
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    const double* xyz = point3D.xyz.data();
    if (problem.HasParameterBlock(xyz) &&
        !problem.IsParameterBlockConstant(const_cast<double*>(xyz))) {
      params.push_back({point3D_id, xyz});
    }
  }
  return params;
}

std::vector<const double*> GetOtherParams(
    const ceres::Problem& problem,
    const std::vector<PoseParam>& poses,
    const std::vector<PointParam>& points) {
  FlatHashSet<const double*> pose_and_point_params;
  pose_and_point_params.reserve(poses.size() + points.size());
  for (const auto& pose : poses) {
    pose_and_point_params.insert(pose.cam_from_world);
  }
  for (const auto& point : points) {
    pose_and_point_params.insert(point.xyz);
  }

  std::vector<const double*> params;
  std::vector<double*> all_params;
  problem.GetParameterBlocks(&all_params);
  for (const double* param : all_params) {
    if (!problem.IsParameterBlockConstant(const_cast<double*>(param)) &&
        pose_and_point_params.count(param) == 0) {
      params.push_back(param);
    }
  }
  return params;
}

}  // namespace internal
}  // namespace colmap
