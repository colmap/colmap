// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/camera.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <cmath>
#include <optional>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// Triangulation estimator to estimate 3D point from multiple observations.
// The triangulation must satisfy the following constraints:
//    - Sufficient triangulation angle between observation pairs.
//    - All observations must satisfy cheirality constraint.
//
// An observation is composed of an image measurement and the corresponding
// camera pose and calibration.
class TriangulationEstimator {
 public:
  enum class ResidualType {
    ANGULAR_ERROR,
    REPROJECTION_ERROR,
  };

  struct PointData {
    PointData() = default;
    PointData(const Eigen::Vector2d& img_point, const Eigen::Vector3d& cam_ray)
        : img_point(img_point), cam_ray(cam_ray) {}
    // Image observation in pixels. Only needs to be set for REPROJECTION_ERROR.
    Eigen::Vector2d img_point = Eigen::Vector2d::Zero();
    // Unit bearing vector in the camera frame (Camera::CamRayFromImg). The
    // canonical observation representation for all camera models, including
    // omnidirectional (EQUIRECTANGULAR) back-hemisphere rays that the 2D
    // normalized representation cannot encode.
    Eigen::Vector3d cam_ray = Eigen::Vector3d::Zero();
  };

  struct PoseData {
    PoseData() : camera(nullptr) {}
    PoseData(const Eigen::Matrix3x4d& cam_from_world,
             const Eigen::Vector3d& proj_center,
             const Camera* camera)
        : cam_from_world(cam_from_world),
          proj_center(proj_center),
          camera(camera) {}
    // The projection matrix for the image of the observation.
    Eigen::Matrix3x4d cam_from_world;
    // The projection center for the image of the observation.
    Eigen::Vector3d proj_center;
    // The camera for the image of the observation.
    const Camera* camera;
  };

  using X_t = PointData;
  using Y_t = PoseData;
  using M_t = Eigen::Vector3d;

  TriangulationEstimator(double min_tri_angle, ResidualType residual_type);

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 2;

  // Estimate a 3D point from a two-view observation.
  //
  // @param point_data        Image measurements.
  // @param pose_data         Camera poses.
  //
  // @return                  Triangulated point if successful, otherwise none.
  void Estimate(const std::vector<X_t>& point_data,
                const std::vector<Y_t>& pose_data,
                std::vector<M_t>* models) const;

  // Calculate residuals in terms of squared reprojection or angular error.
  //
  // @param point_data        Image measurements.
  // @param pose_data         Camera poses.
  // @param xyz               3D point.
  //
  // @return                  Residual for each observation.
  void Residuals(const std::vector<X_t>& point_data,
                 const std::vector<Y_t>& pose_data,
                 const M_t& xyz,
                 std::vector<double>* residuals) const;

 private:
  const double min_tri_angle_;
  const ResidualType residual_type_;
};

struct EstimateTriangulationOptions {
  // Minimum triangulation angle in radians.
  double min_tri_angle = 0.0;

  // The employed residual type.
  TriangulationEstimator::ResidualType residual_type =
      TriangulationEstimator::ResidualType::ANGULAR_ERROR;

  // RANSAC options for TriangulationEstimator.
  RANSACOptions ransac_options;

  EstimateTriangulationOptions() {
    ransac_options.max_error = DegToRad(2.0);
    ransac_options.confidence = 0.9999;
    ransac_options.min_inlier_ratio = 0.02;
    ransac_options.max_num_trials = 10000;
  }

  void Check() const {
    THROW_CHECK_GE(min_tri_angle, 0.0);
    ransac_options.Check();
  }
};

// Robustly estimate 3D point from observations in multiple views using RANSAC
// and a subsequent non-linear refinement using all inliers. Returns true
// if the estimated number of inliers has more than two views.
bool EstimateTriangulation(const EstimateTriangulationOptions& options,
                           const std::vector<Eigen::Vector2d>& points,
                           const std::vector<Rigid3d>& cams_from_world,
                           const std::vector<Camera const*>& cameras,
                           std::vector<char>* inlier_mask,
                           Eigen::Vector3d* xyz);

// Propagate a 6x6 pose covariance to image coordinates using first-order
// covariance propagation:
//
//      S_x = J_pose * S_pose * J_pose^T
//          with J_pose = J_proj * [-2 * skew(R * X_world), I_3]
//
// where R is the world-to-camera rotation, X_world the 3D point, and J_proj
// the 2x3 projection Jacobian d(img)/d(cam_point). The pose covariance follows
// the Ceres EigenQuaternionManifold tangent convention: a left-multiplied
// quaternion delta whose vector is half the physical rotation angle, followed
// by additive translation coordinates.
Eigen::Matrix2d PropagatePoseCovarianceToImage(
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& point3D_in_world,
    const Eigen::Matrix<double, 2, 3>& J_proj,
    const Eigen::Matrix6d& pose_cov);

// Propagate a 3D point covariance from the world frame to image coordinates:
//
//      S_x = J * S_X * J^T with J = J_proj * R
//
// where R is the world-to-camera rotation and J_proj the 2x3 projection
// Jacobian d(img)/d(cam_point) for any camera model. This overload takes an
// explicit Jacobian, unlike the normalized-coordinates variant in
// solvers/absolute_pose.h which derives it from the pinhole model.
Eigen::Matrix2d PropagatePointCovarianceToImage(
    const Eigen::Matrix3d& rotation,
    const Eigen::Matrix<double, 2, 3>& J_proj,
    const Eigen::Matrix3d& point3D_cov);

// Wrap an x-residual in pixels into [-width / 2, width / 2) for
// equirectangular cameras, so that the residual stays continuous across the
// +-pi seam. Returns the residual unchanged for all other camera models.
double WrapEquirectangularXResidual(const Camera& camera, double dx);

// Triangulation estimator that considers 2D measurement and pose covariances:
// residuals are squared Mahalanobis distances of the pixel reprojection error
// under the joint measurement + propagated pose covariance. Intended for use
// in (LO-)RANSAC, where Refine serves as the local estimator.
class CovariantTriangulationEstimator {
 public:
  struct PointData {
    PointData() = default;
    // Image observation in pixels.
    Eigen::Vector2d img_point = Eigen::Vector2d::Zero();
    // Unit bearing vector in the camera frame, for the DLT seed.
    Eigen::Vector3d cam_ray = Eigen::Vector3d::Zero();
    // 2D measurement covariance in pixels. Identity by default.
    Eigen::Matrix2d img_cov = Eigen::Matrix2d::Identity();
  };

  struct PoseData {
    PoseData() : camera(nullptr) {}
    // The projection matrix for the image of the observation.
    Eigen::Matrix3x4d cam_from_world;
    // The camera for the image of the observation.
    const Camera* camera = nullptr;
    // 6x6 pose covariance in the Ceres tangent convention (see
    // PropagatePoseCovarianceToImage). Zero means an exactly known pose.
    Eigen::Matrix6d pose_cov = Eigen::Matrix6d::Zero();
  };

  using X_t = PointData;
  using Y_t = PoseData;
  using M_t = Eigen::Vector3d;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 2;

  // Estimate a 3D point from a two-view observation with the DLT seed,
  // ignoring covariances. Enforces cheirality but no triangulation angle;
  // degeneracy is instead gated on the estimated uncertainty downstream.
  static void Estimate(const std::vector<X_t>& point_data,
                       const std::vector<Y_t>& pose_data,
                       std::vector<M_t>* models);

  // Squared Mahalanobis distance of the pixel reprojection error under the
  // joint measurement + propagated pose covariance. Observations behind the
  // camera, failing projection, or with non-positive-definite joint
  // covariance receive maximum residual.
  static void Residuals(const std::vector<X_t>& point_data,
                        const std::vector<Y_t>& pose_data,
                        const M_t& xyz,
                        std::vector<double>* residuals);

  // Nonlinear local optimization of the 3D point over the given observations,
  // starting from *xyz. Minimizes covariance-whitened pixel reprojection
  // errors with Levenberg-Marquardt (colmap::TinySolver) and an analytical
  // Jacobian. Covariance weights are held fixed per solve and updated from
  // the current estimate over a small number of reweighting rounds.
  //
  // Returns true and overwrites *xyz with the refined point on success.
  // Returns false and leaves *xyz unchanged if fewer than kMinNumSamples
  // observations are given or the solve fails.
  static bool Refine(const std::vector<X_t>& point_data,
                     const std::vector<Y_t>& pose_data,
                     M_t* xyz);

  // Linearized 3x3 point covariance at xyz over the given observations,
  // (J^T J)^-1 of the whitened Jacobian. Returns nullopt if the information
  // matrix is singular or fewer than two observations project successfully.
  static std::optional<Eigen::Matrix3d> PointCovariance(
      const std::vector<X_t>& point_data,
      const std::vector<Y_t>& pose_data,
      const M_t& xyz);
};

// Squared Mahalanobis distance of the pixel reprojection error of a 3D point
// under the joint measurement + propagated point and pose covariance. Returns
// nullopt if the observation cannot be scored (point behind the camera,
// projection failure, non-finite inputs, or non-positive-definite joint
// covariance). The observed bearing point_data.cam_ray is only required for
// spherical cameras.
std::optional<double> CovariantSquaredReprojectionError(
    const CovariantTriangulationEstimator::PointData& point_data,
    const CovariantTriangulationEstimator::PoseData& pose_data,
    const Eigen::Vector3d& xyz,
    const Eigen::Matrix3d& xyz_cov = Eigen::Matrix3d::Zero());

// Maximum relative depth uncertainty of a 3D point over the given cameras,
// i.e., the standard deviation along the viewing ray over the distance.
// Returns infinity if the point coincides with a camera center.
double MaxRelativeDepthUncertainty(
    const Eigen::Vector3d& xyz,
    const Eigen::Matrix3d& xyz_cov,
    const std::vector<Rigid3d>& cams_from_world);

struct CovariantTriangulationOptions {
  // Chi-squared threshold for inlier gating of the squared Mahalanobis
  // residuals (2 DoF). Overrides ransac_options.max_error.
  double inlier_chi2_threshold = kChiSquare99TwoDof;

  // Maximum relative depth uncertainty (ray-direction std over distance,
  // worst inlier view) for accepting a triangulation. Scale-free
  // replacement for a minimum triangulation angle.
  double max_relative_depth_uncertainty = 0.05;

  // RANSAC options for CovariantTriangulationEstimator. max_error is
  // overridden from inlier_chi2_threshold.
  RANSACOptions ransac_options;

  CovariantTriangulationOptions() {
    ransac_options.max_error = std::sqrt(kChiSquare99TwoDof);
    ransac_options.confidence = 0.9999;
    ransac_options.min_inlier_ratio = 0.02;
    ransac_options.max_num_trials = 10000;
  }

  void Check() const {
    THROW_CHECK_GT(inlier_chi2_threshold, 0.0);
    THROW_CHECK_GT(max_relative_depth_uncertainty, 0.0);
    ransac_options.Check();
  }
};

// Robustly estimate a 3D point and its covariance from observations in
// multiple views using LORANSAC with whitened refinement, and a subsequent
// uncertainty-degeneracy gate. Returns false if RANSAC fails, the information
// matrix is singular, or the relative depth uncertainty is too large.
//
// @param options            Covariant triangulation options.
// @param points2D           Corresponding 2D points in pixels.
// @param points2D_cov       Corresponding 2D measurement covariances. Empty
//                           for identity covariances.
// @param cams_from_world    Corresponding camera poses.
// @param pose_covs          Corresponding 6x6 pose covariances in the Ceres
//                           tangent convention. Empty for exactly known poses.
// @param cameras            Corresponding cameras.
// @param inlier_mask        Inlier mask for the observations.
// @param xyz                Estimated 3D point.
// @param xyz_cov            Estimated 3D point covariance.
//
// @return                   Whether triangulation is accepted.
bool EstimateCovariantTriangulation(
    const CovariantTriangulationOptions& options,
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Matrix2d>& points2D_cov,
    const std::vector<Rigid3d>& cams_from_world,
    const std::vector<Eigen::Matrix6d>& pose_covs,
    const std::vector<Camera const*>& cameras,
    std::vector<char>* inlier_mask,
    Eigen::Vector3d* xyz,
    Eigen::Matrix3d* xyz_cov);

}  // namespace colmap
