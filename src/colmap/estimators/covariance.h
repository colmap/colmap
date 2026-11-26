// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/hash_containers.h"

#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ceres/problem.h>

namespace colmap {
namespace internal {
struct PoseParam;
}

struct BACovariance {
  explicit BACovariance(
      FlatHashMap<point3D_t, Eigen::MatrixXd> point_covs,
      NodeHashMap<image_t, std::pair<int, int>> pose_L_start_size,
      NodeHashMap<const double*, std::pair<int, int>> other_L_start_size,
      Eigen::MatrixXd L_inv,
      double posterior_variance_factor = 1.0,
      double spatial_design_effect = 1.0);

  // Estimated a posteriori observation variance factor sigma_0^2 (1.0 when
  // VarianceScaling::NONE). Scales all covariances (points, poses, and others).
  double PosteriorVarianceFactor() const;

  // Estimated 3x2 spatial-tile design effect DEFF >= 1.0 accounting for
  // within-image spatially correlated reprojection residuals (1.0 unless
  // VarianceScaling::SPATIAL_DEFF). Scales pose and other covariances.
  double SpatialDesignEffect() const;

  // Combined variance factor applied to pose and other covariances:
  // PosteriorVarianceFactor() * SpatialDesignEffect().
  double VarianceFactor() const;

  // Covariance for 3D points, conditioned on all other variables set constant.
  // If some dimensions are kept constant, the respective rows/columns are
  // omitted. Returns null if 3D point not a variable in the problem.
  std::optional<Eigen::MatrixXd> GetPointCov(point3D_t point3D_id) const;

  // Tangent space covariance in the order [rotation, translation]. If some
  // dimensions are kept constant, the respective rows/columns are omitted.
  // Returns null if image is not a variable in the problem.
  std::optional<Eigen::MatrixXd> GetCamCovFromWorld(image_t image_id) const;
  std::optional<Eigen::MatrixXd> GetCamCrossCovFromWorld(
      image_t image_id1, image_t image_id2) const;
  // Get relative pose covariance in the order [rotation, translation]. This
  // function returns null if some dimensions are kept constant for either of
  // the two poses. This does not mean that one cannot get relative pose
  // covariance for such case, but requires custom logic to fill in zero block
  // in the covariance matrix.
  std::optional<Eigen::MatrixXd> GetCam2CovFromCam1(
      image_t image_id1,
      const Rigid3d& cam1_from_world,
      image_t image_id2,
      const Rigid3d& cam2_from_world) const;

  // Tangent space covariance for any other variable parameter block in the
  // problem. If some dimensions are kept constant, the respective rows/columns
  // are omitted. Returns null if parameter block is not a variable in the
  // problem.
  std::optional<Eigen::MatrixXd> GetOtherParamsCov(const double* params) const;

 private:
  const FlatHashMap<point3D_t, Eigen::MatrixXd> point_covs_;
  const NodeHashMap<image_t, std::pair<int, int>> pose_L_start_size_;
  const NodeHashMap<const double*, std::pair<int, int>> other_L_start_size_;
  const Eigen::MatrixXd L_inv_;
  const double posterior_variance_factor_;
  const double spatial_design_effect_;
};

struct BACovarianceOptions {
  enum class Params {
    POSES,
    POINTS,
    POSES_AND_POINTS,
    ALL,  // + Others
  };

  enum class VarianceScaling {
    // Return unscaled covariances assuming unit observation variance.
    NONE,
    // Scale all covariances (points, poses, and others) by the unbiased a
    // posteriori variance factor
    //   sigma_0^2 = 2 * cost / (num_residuals - (num_params - num_null)).
    // Assumes quadratic (TRIVIAL) residuals. When params == POINTS with
    // variable poses, the Schur complement is not factored so num_null is not
    // subtracted from num_params.
    POSTERIOR,
    // Scale point covariances by sigma_0^2 and pose/other covariances by
    // sigma_0^2 * DEFF, where DEFF >= 1.0 is the 3x2 spatial-tile design effect
    // estimated from within-image reprojection residuals to account for
    // spatially correlated errors. Assumes quadratic (TRIVIAL) residuals and is
    // sensitive to outliers and robust loss functions. Prefer
    // EstimateBACovariance over EstimateBACovarianceFromProblem so the exact
    // BundleAdjustmentConfig image and point selection is used. Fails loudly
    // (returns nullopt) if residual redundancy <= 0, variance scaling
    // overflows,
    // or insufficient observations populate the 3x2 image tiles.
    SPATIAL_DEFF,
  };

  // For which parameters to compute the covariance.
  Params params = Params::ALL;

  // How to scale the unscaled inverse-Hessian covariances from residuals.
  VarianceScaling variance_scaling = VarianceScaling::NONE;

  // Damping for the point covariances and the Schur elimination of the other
  // parameters. Enables to robustly deal with poorly conditioned parameters.
  double damping = 1e-8;

  // WARNING: This option will be removed in a future release, use at your own
  // risk. For custom bundle adjustment problems, this enables to specify a
  // custom set of pose parameter blocks to consider. Note that these pose
  // blocks must not necessarily be part of the reconstruction but they must
  // follow the standard requirement for applying the Schur complement trick.
  // TODO: This is a temporary option to enable extraction of pose covariances
  // for custom rig bundle adjustment problems. To be removed when proper rig
  // support is enabled in colmap natively.
  std::vector<internal::PoseParam> experimental_custom_poses;
};

// Computes covariances for the parameters in a bundle adjustment problem. It is
// important that the problem has a structure suitable for solving using the
// Schur complement trick. This is the case for the standard configuration of
// bundle adjustment problems, but be careful if you modify the underlying
// problem with custom residuals. If the problem leaves the gauge free (up to
// the 7 DoF of a similarity transform), the pose and other covariances are
// returned in the minimum-trace gauge, i.e., the pseudo-inverse of the Schur
// complement. Note that any rank deficiency of dimension <= 7 supported solely
// on pose parameters is treated as a gauge freedom (underconstrained poses in
// otherwise fixed-gauge problems are not distinguished from a free gauge).
// Point covariances are conditioned on the pose and other parameters and thus
// independent of the gauge.
// Returns null if the estimation was not successful.
std::optional<BACovariance> EstimateBACovariance(
    const BACovarianceOptions& options,
    const Reconstruction& reconstruction,
    CeresBundleAdjuster& bundle_adjuster);
// Same as EstimateBACovariance, operating directly on a ceres::Problem without
// a BundleAdjustmentConfig. With VarianceScaling::SPATIAL_DEFF (and empty
// experimental_custom_poses), participating images are inferred from pose
// parameter blocks in the problem (falling back to camera parameter blocks
// only when no pose parameter blocks exist). Problems with baked constant-pose
// cost functors (e.g., mixed constant/variable poses or constant-pose subsets
// sharing a camera) should use EstimateBACovariance instead.
std::optional<BACovariance> EstimateBACovarianceFromProblem(
    const BACovarianceOptions& options,
    const Reconstruction& reconstruction,
    ceres::Problem& problem);

namespace internal {

struct PoseParam {
  image_t image_id = kInvalidImageId;
  const double* cam_from_world = nullptr;
};

std::vector<PoseParam> GetPoseParams(const Reconstruction& reconstruction,
                                     const ceres::Problem& problem);

struct PointParam {
  point3D_t point3D_id = kInvalidPoint3DId;
  const double* xyz = nullptr;
};

std::vector<PointParam> GetPointParams(const Reconstruction& reconstruction,
                                       const ceres::Problem& problem);

std::vector<const double*> GetOtherParams(
    const ceres::Problem& problem,
    const std::vector<PoseParam>& poses,
    const std::vector<PointParam>& points);

}  // namespace internal
}  // namespace colmap
