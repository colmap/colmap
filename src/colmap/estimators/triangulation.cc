// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/triangulation.h"

#include "colmap/geometry/triangulation.h"
#include "colmap/optim/combination_sampler.h"
#include "colmap/optim/loransac.h"
#include "colmap/optim/tiny_solver.h"
#include "colmap/scene/projection.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace colmap {
namespace {

struct ProjectedObservation {
  Eigen::Vector2d img_point;
  Eigen::Matrix2d joint_cov;
  Eigen::Matrix2x3d J_proj;
};

// Project xyz into the observation's image and form the joint 2x2 image
// covariance (measurement + propagated pose covariance). Returns nullopt if
// the point is behind the camera, projection fails, or a covariance is
// non-finite.
std::optional<ProjectedObservation> ProjectWithJointCovariance(
    const CovariantTriangulationEstimator::PointData& point_data,
    const CovariantTriangulationEstimator::PoseData& pose_data,
    const Eigen::Vector3d& xyz) {
  if (!point_data.img_cov.allFinite() || !pose_data.pose_cov.allFinite()) {
    return std::nullopt;
  }
  const Camera* camera = THROW_CHECK_NOTNULL(pose_data.camera);
  const Eigen::Vector3d point3D_in_cam =
      pose_data.cam_from_world * xyz.homogeneous();
  // Perspective cameras require positive depth. Spherical cameras instead
  // accept both hemispheres, but the projected point must face the observed
  // bearing. The inverted comparisons also reject NaN coordinates.
  if (camera->IsPerspective()) {
    if (!(point3D_in_cam.z() > std::numeric_limits<double>::epsilon())) {
      return std::nullopt;
    }
  } else if (!(point3D_in_cam.dot(point_data.cam_ray) > 0.0)) {
    return std::nullopt;
  }
  Eigen::Matrix2x3d J_proj;
  const std::optional<Eigen::Vector2d> proj =
      camera->ImgFromCamWithJac(point3D_in_cam, &J_proj);
  if (!proj.has_value() || !proj->allFinite() || !J_proj.allFinite()) {
    return std::nullopt;
  }
  Eigen::Matrix2d joint_cov = point_data.img_cov;
  if (pose_data.pose_cov.squaredNorm() > 0) {
    joint_cov +=
        PropagatePoseCovarianceToImage(pose_data.cam_from_world.leftCols<3>(),
                                       xyz,
                                       J_proj,
                                       pose_data.pose_cov);
  }
  if (!joint_cov.allFinite()) {
    return std::nullopt;
  }
  return ProjectedObservation{*proj, joint_cov, J_proj};
}

// Cost functor for colmap::TinySolver refinement of a 3D point over
// covariance-whitened pixel reprojection errors (two residuals per
// observation):
//
//      r_i = S_i^-1/2 * (x_i - Proj_i(X))
//
// with an analytical Jacobian and fixed per-solve weights. Observations that
// fail projection contribute a zero residual with zero Jacobian rows.
class TinyTriangulationCostFunctor {
 public:
  using Scalar = double;
  static constexpr int NUM_RESIDUALS = Eigen::Dynamic;
  static constexpr int NUM_PARAMETERS = 3;

  TinyTriangulationCostFunctor(
      const std::vector<CovariantTriangulationEstimator::PointData>& point_data,
      const std::vector<CovariantTriangulationEstimator::PoseData>& pose_data,
      const std::vector<Eigen::Matrix2d>& sqrt_infos)
      : point_data_(point_data),
        pose_data_(pose_data),
        sqrt_infos_(sqrt_infos) {}

  int NumResiduals() const { return 2 * static_cast<int>(point_data_.size()); }

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    const Eigen::Map<const Eigen::Vector3d> xyz(parameters);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::ColMajor>>
        jacobian_mat(jacobian, NumResiduals(), 3);
    for (size_t i = 0; i < point_data_.size(); ++i) {
      Eigen::Map<Eigen::Vector2d> residual_vec(residuals + 2 * i);
      const Camera* camera = THROW_CHECK_NOTNULL(pose_data_[i].camera);
      const Eigen::Vector3d point3D_in_cam =
          pose_data_[i].cam_from_world * xyz.homogeneous();
      Eigen::Matrix2x3d J_proj;
      const bool has_valid_depth =
          camera->IsPerspective()
              ? point3D_in_cam.z() > std::numeric_limits<double>::epsilon()
              : point3D_in_cam.dot(point_data_[i].cam_ray) > 0.0;
      const std::optional<Eigen::Vector2d> proj =
          has_valid_depth ? camera->ImgFromCamWithJac(point3D_in_cam, &J_proj)
                          : std::nullopt;
      if (!proj.has_value() || !proj->allFinite() || !J_proj.allFinite()) {
        residual_vec.setZero();
        if (jacobian != nullptr) {
          jacobian_mat.block<2, 3>(2 * i, 0).setZero();
        }
        continue;
      }
      Eigen::Vector2d error = point_data_[i].img_point - *proj;
      error.x() = WrapEquirectangularXResidual(*camera, error.x());
      residual_vec = sqrt_infos_[i] * error;
      if (jacobian != nullptr) {
        jacobian_mat.block<2, 3>(2 * i, 0) =
            -sqrt_infos_[i] * J_proj *
            pose_data_[i].cam_from_world.leftCols<3>();
      }
    }
    return true;
  }

 private:
  const std::vector<CovariantTriangulationEstimator::PointData>& point_data_;
  const std::vector<CovariantTriangulationEstimator::PoseData>& pose_data_;
  const std::vector<Eigen::Matrix2d>& sqrt_infos_;
};

}  // namespace

TriangulationEstimator::TriangulationEstimator(double min_tri_angle,
                                               ResidualType residual_type)
    : min_tri_angle_(min_tri_angle), residual_type_(residual_type) {
  THROW_CHECK_GE(min_tri_angle, 0);
}

void TriangulationEstimator::Estimate(const std::vector<X_t>& point_data,
                                      const std::vector<Y_t>& pose_data,
                                      std::vector<M_t>* models) const {
  THROW_CHECK_GE(point_data.size(), 2);
  THROW_CHECK_EQ(point_data.size(), pose_data.size());
  THROW_CHECK(models != nullptr);

  models->clear();

  M_t xyz;
  if (point_data.size() == 2) {
    // More efficient closed-form solution for the two-view case.
    const bool all_cams_perspective =
        std::all_of(pose_data.begin(), pose_data.end(), [](const Y_t& pose) {
          return THROW_CHECK_NOTNULL(pose.camera)->IsPerspective();
        });
    if (all_cams_perspective) {
      if (!TriangulatePoint(
              pose_data[0].cam_from_world,
              pose_data[1].cam_from_world,
              Eigen::Vector2d(point_data[0].cam_ray.hnormalized()),
              Eigen::Vector2d(point_data[1].cam_ray.hnormalized()),
              &xyz)) {
        return;
      }
    } else {
      if (!TriangulatePoint(pose_data[0].cam_from_world,
                            pose_data[1].cam_from_world,
                            point_data[0].cam_ray,
                            point_data[1].cam_ray,
                            &xyz)) {
        return;
      }
    }

  } else {
    std::vector<Eigen::Matrix3x4d> cams_from_world(point_data.size());
    std::vector<Eigen::Vector3d> cam_rays(point_data.size());
    for (size_t i = 0; i < point_data.size(); ++i) {
      cams_from_world[i] = pose_data[i].cam_from_world;
      cam_rays[i] = point_data[i].cam_ray;
    }
    if (!TriangulateMultiViewPoint(
            span<const Eigen::Matrix3x4d>(cams_from_world.data(),
                                          cams_from_world.size()),
            span<const Eigen::Vector3d>(cam_rays.data(), cam_rays.size()),
            &xyz)) {
      return;
    }
  }

  // Cheirality. Perspective cameras require positive depth (the point in front
  // of the local +Z axis). Omnidirectional cameras (e.g. EQUIRECTANGULAR) have
  // no single front, but the point must still lie in the half-space the
  // observed bearing points toward.
  for (size_t i = 0; i < pose_data.size(); ++i) {
    if (pose_data[i].camera->IsPerspective()) {
      if (!HasPointPositiveDepth(pose_data[i].cam_from_world, xyz)) {
        return;
      }
    } else if ((pose_data[i].cam_from_world * xyz.homogeneous())
                   .dot(point_data[i].cam_ray) <= 0.0) {
      return;
    }
  }

  // Require a sufficient triangulation angle for at least one pair of views.
  for (size_t i = 0; i < pose_data.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (CalculateTriangulationAngle(pose_data[i].proj_center,
                                      pose_data[j].proj_center,
                                      xyz) >= min_tri_angle_) {
        models->resize(1);
        (*models)[0] = xyz;
        return;
      }
    }
  }
}

void TriangulationEstimator::Residuals(const std::vector<X_t>& point_data,
                                       const std::vector<Y_t>& pose_data,
                                       const M_t& xyz,
                                       std::vector<double>* residuals) const {
  THROW_CHECK_EQ(point_data.size(), pose_data.size());

  residuals->resize(point_data.size());

  for (size_t i = 0; i < point_data.size(); ++i) {
    if (residual_type_ == ResidualType::REPROJECTION_ERROR) {
      (*residuals)[i] =
          CalculateSquaredReprojectionError(point_data[i].img_point,
                                            xyz,
                                            pose_data[i].cam_from_world,
                                            *pose_data[i].camera);
    } else if (residual_type_ == ResidualType::ANGULAR_ERROR) {
      const double angular_error = CalculateAngularReprojectionError(
          point_data[i].cam_ray, xyz, pose_data[i].cam_from_world);
      (*residuals)[i] = angular_error * angular_error;
    }
  }
}

bool EstimateTriangulation(const EstimateTriangulationOptions& options,
                           const std::vector<Eigen::Vector2d>& points,
                           const std::vector<Rigid3d>& cams_from_world,
                           const std::vector<Camera const*>& cameras,
                           std::vector<char>* inlier_mask,
                           Eigen::Vector3d* xyz) {
  THROW_CHECK_NOTNULL(inlier_mask);
  THROW_CHECK_NOTNULL(xyz);
  THROW_CHECK_GE(points.size(), 2);
  THROW_CHECK_EQ(points.size(), cams_from_world.size());
  THROW_CHECK_EQ(points.size(), cameras.size());
  options.Check();

  std::vector<TriangulationEstimator::PointData> point_data;
  point_data.resize(points.size());
  std::vector<TriangulationEstimator::PoseData> pose_data;
  pose_data.resize(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    point_data[i].img_point = points[i];
    // Unit bearing in the camera frame. CamRayFromImg yields a valid ray for
    // any camera model, including omnidirectional (EQUIRECTANGULAR)
    // back-hemisphere observations that CamFromImg cannot represent. Fall back
    // to a defined forward bearing (+Z) if unprojection fails, so downstream
    // normalize() in the DLT never sees a zero vector (which would produce
    // NaNs).
    point_data[i].cam_ray =
        cameras[i]->CamRayFromImg(points[i]).value_or(Eigen::Vector3d::UnitZ());
    pose_data[i].cam_from_world = cams_from_world[i].ToMatrix();
    pose_data[i].proj_center = cams_from_world[i].TgtOriginInSrc();
    pose_data[i].camera = cameras[i];
  }

  // Robustly estimate track using LORANSAC.
  LORANSAC<TriangulationEstimator,
           TriangulationEstimator,
           InlierSupportMeasurer,
           CombinationSampler>
      ransac(
          options.ransac_options,
          TriangulationEstimator(options.min_tri_angle, options.residual_type),
          TriangulationEstimator(options.min_tri_angle, options.residual_type));
  auto report = ransac.Estimate(point_data, pose_data);
  if (!report.success) {
    return false;
  }

  *inlier_mask = std::move(report.inlier_mask);
  *xyz = report.model;

  return report.success;
}

double WrapEquirectangularXResidual(const Camera& camera, const double dx) {
  if (camera.model_id == CameraModelId::kEquirectangular && camera.width > 0) {
    // Locally constant offset, hence Jacobians are unaffected.
    const double width = static_cast<double>(camera.width);
    return dx - width * std::floor(dx / width + 0.5);
  }
  return dx;
}

Eigen::Matrix2d PropagatePoseCovarianceToImage(
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& point3D_in_world,
    const Eigen::Matrix<double, 2, 3>& J_proj,
    const Eigen::Matrix6d& pose_cov) {
  // Ceres' quaternion manifold uses the left-multiplicative retraction
  // Exp(delta) * q, where delta is the vector part of the delta quaternion.
  // Thus the physical rotation angle is 2 * |delta| and
  // d(cam_point)/d(delta) = -2 * skew(R * X_world).
  const Eigen::Vector3d rotated_point = rotation * point3D_in_world;
  Eigen::Matrix<double, 3, 3> skew;
  skew << 0, -rotated_point.z(), rotated_point.y(), rotated_point.z(), 0,
      -rotated_point.x(), -rotated_point.y(), rotated_point.x(), 0;
  Eigen::Matrix<double, 2, 6> J_pose;
  J_pose.leftCols<3>() = -2.0 * J_proj * skew;
  J_pose.rightCols<3>() = J_proj;
  return J_pose * pose_cov * J_pose.transpose();
}

Eigen::Matrix2d PropagatePointCovarianceToImage(
    const Eigen::Matrix3d& rotation,
    const Eigen::Matrix<double, 2, 3>& J_proj,
    const Eigen::Matrix3d& point3D_cov) {
  const Eigen::Matrix<double, 2, 3> J = J_proj * rotation;
  return J * point3D_cov * J.transpose();
}

void CovariantTriangulationEstimator::Estimate(
    const std::vector<X_t>& point_data,
    const std::vector<Y_t>& pose_data,
    std::vector<M_t>* models) {
  THROW_CHECK_GE(point_data.size(), 2);
  THROW_CHECK_EQ(point_data.size(), pose_data.size());
  THROW_CHECK(models != nullptr);

  models->clear();

  M_t xyz;
  if (point_data.size() == 2) {
    // More efficient closed-form solution for the two-view case.
    const bool all_cams_perspective =
        std::all_of(pose_data.begin(), pose_data.end(), [](const Y_t& pose) {
          return THROW_CHECK_NOTNULL(pose.camera)->IsPerspective();
        });
    if (all_cams_perspective) {
      if (!TriangulatePoint(
              pose_data[0].cam_from_world,
              pose_data[1].cam_from_world,
              Eigen::Vector2d(point_data[0].cam_ray.hnormalized()),
              Eigen::Vector2d(point_data[1].cam_ray.hnormalized()),
              &xyz)) {
        return;
      }
    } else {
      if (!TriangulatePoint(pose_data[0].cam_from_world,
                            pose_data[1].cam_from_world,
                            point_data[0].cam_ray,
                            point_data[1].cam_ray,
                            &xyz)) {
        return;
      }
    }
  } else {
    std::vector<Eigen::Matrix3x4d> cams_from_world(point_data.size());
    std::vector<Eigen::Vector3d> cam_rays(point_data.size());
    for (size_t i = 0; i < point_data.size(); ++i) {
      cams_from_world[i] = pose_data[i].cam_from_world;
      cam_rays[i] = point_data[i].cam_ray;
    }
    if (!TriangulateMultiViewPoint(
            span<const Eigen::Matrix3x4d>(cams_from_world.data(),
                                          cams_from_world.size()),
            span<const Eigen::Vector3d>(cam_rays.data(), cam_rays.size()),
            &xyz)) {
      return;
    }
  }

  if (!xyz.allFinite()) {
    return;
  }

  // Cheirality, as in TriangulationEstimator::Estimate.
  for (size_t i = 0; i < pose_data.size(); ++i) {
    if (pose_data[i].camera->IsPerspective()) {
      if (!HasPointPositiveDepth(pose_data[i].cam_from_world, xyz)) {
        return;
      }
    } else if ((pose_data[i].cam_from_world * xyz.homogeneous())
                   .dot(point_data[i].cam_ray) <= 0.0) {
      return;
    }
  }

  models->resize(1);
  (*models)[0] = xyz;
}

void CovariantTriangulationEstimator::Residuals(
    const std::vector<X_t>& point_data,
    const std::vector<Y_t>& pose_data,
    const M_t& xyz,
    std::vector<double>* residuals) {
  THROW_CHECK_EQ(point_data.size(), pose_data.size());

  residuals->resize(point_data.size());

  for (size_t i = 0; i < point_data.size(); ++i) {
    (*residuals)[i] =
        CovariantSquaredReprojectionError(point_data[i], pose_data[i], xyz)
            .value_or(std::numeric_limits<double>::max());
  }
}

bool CovariantTriangulationEstimator::Refine(const std::vector<X_t>& point_data,
                                             const std::vector<Y_t>& pose_data,
                                             M_t* xyz) {
  THROW_CHECK_EQ(point_data.size(), pose_data.size());
  THROW_CHECK_NOTNULL(xyz);
  if (point_data.size() < kMinNumSamples) {
    return false;
  }

  // Iteratively reweighted refinement: covariance weights are held fixed per
  // solve and updated from the current estimate.
  constexpr int kNumReweightingRounds = 2;
  M_t refined_xyz = *xyz;
  std::vector<Eigen::Matrix2d> sqrt_infos(point_data.size());
  for (int round = 0; round < kNumReweightingRounds; ++round) {
    for (size_t i = 0; i < point_data.size(); ++i) {
      const std::optional<ProjectedObservation> proj =
          ProjectWithJointCovariance(point_data[i], pose_data[i], refined_xyz);
      Eigen::LLT<Eigen::Matrix2d> llt(
          proj.has_value() ? proj->joint_cov : Eigen::Matrix2d::Identity());
      sqrt_infos[i] = llt.matrixL().solve(Eigen::Matrix2d::Identity());
      if (llt.info() != Eigen::Success || !sqrt_infos[i].allFinite()) {
        // Fall back to unweighted residuals for degenerate covariances.
        sqrt_infos[i] = Eigen::Matrix2d::Identity();
      }
    }

    TinyTriangulationCostFunctor functor(point_data, pose_data, sqrt_infos);
    using Solver = TinySolver<TinyTriangulationCostFunctor>;
    Solver solver;
    typename Solver::Options options;
    options.max_num_iterations = 25;

    Eigen::Vector3d x = refined_xyz;
    if (solver.Solve(functor, &x, options).status ==
        Solver::NUMERICAL_FAILURE) {
      return false;
    }
    refined_xyz = x;
    if (!refined_xyz.allFinite()) {
      return false;
    }
  }

  *xyz = refined_xyz;
  return true;
}

std::optional<Eigen::Matrix3d> CovariantTriangulationEstimator::PointCovariance(
    const std::vector<X_t>& point_data,
    const std::vector<Y_t>& pose_data,
    const M_t& xyz) {
  THROW_CHECK_EQ(point_data.size(), pose_data.size());

  Eigen::MatrixXd whitened_jacobian =
      Eigen::MatrixXd::Zero(2 * point_data.size(), 3);
  size_t num_valid = 0;
  for (size_t i = 0; i < point_data.size(); ++i) {
    const std::optional<ProjectedObservation> proj =
        ProjectWithJointCovariance(point_data[i], pose_data[i], xyz);
    if (!proj.has_value()) {
      continue;
    }
    Eigen::LLT<Eigen::Matrix2d> llt(proj->joint_cov);
    if (llt.info() != Eigen::Success) {
      continue;
    }
    const Eigen::Matrix2d sqrt_info =
        llt.matrixL().solve(Eigen::Matrix2d::Identity());
    if (!sqrt_info.allFinite()) {
      continue;
    }
    const Eigen::Vector3d point3D_in_cam =
        pose_data[i].cam_from_world * xyz.homogeneous();
    Eigen::Matrix2x3d J_proj;
    if (!pose_data[i]
             .camera->ImgFromCamWithJac(point3D_in_cam, &J_proj)
             .has_value() ||
        !J_proj.allFinite()) {
      continue;
    }
    whitened_jacobian.block<2, 3>(2 * i, 0) =
        sqrt_info * J_proj * pose_data[i].cam_from_world.leftCols<3>();
    num_valid += 1;
  }

  if (num_valid < 2) {
    return std::nullopt;
  }

  Eigen::LDLT<Eigen::Matrix3d> ldlt(whitened_jacobian.transpose() *
                                    whitened_jacobian);
  if (ldlt.info() != Eigen::Success || (ldlt.vectorD().array() <= 0).any()) {
    return std::nullopt;
  }
  const Eigen::Matrix3d cov = ldlt.solve(Eigen::Matrix3d::Identity());
  if (!cov.allFinite()) {
    return std::nullopt;
  }
  return cov;
}

std::optional<double> CovariantSquaredReprojectionError(
    const CovariantTriangulationEstimator::PointData& point_data,
    const CovariantTriangulationEstimator::PoseData& pose_data,
    const Eigen::Vector3d& xyz,
    const Eigen::Matrix3d& xyz_cov) {
  if (!xyz_cov.allFinite()) {
    return std::nullopt;
  }
  const std::optional<ProjectedObservation> proj =
      ProjectWithJointCovariance(point_data, pose_data, xyz);
  if (!proj.has_value()) {
    return std::nullopt;
  }
  Eigen::Matrix2d joint_cov = proj->joint_cov;
  if (xyz_cov.squaredNorm() > 0) {
    joint_cov += PropagatePointCovarianceToImage(
        pose_data.cam_from_world.leftCols<3>(), proj->J_proj, xyz_cov);
    if (!joint_cov.allFinite()) {
      return std::nullopt;
    }
  }
  Eigen::Vector2d error = point_data.img_point - proj->img_point;
  error.x() = WrapEquirectangularXResidual(*pose_data.camera, error.x());
  Eigen::LDLT<Eigen::Matrix2d> ldlt(joint_cov);
  if (ldlt.info() != Eigen::Success || (ldlt.vectorD().array() <= 0).any()) {
    return std::nullopt;
  }
  const double mahalanobis_dist_sqr = error.transpose() * ldlt.solve(error);
  if (!std::isfinite(mahalanobis_dist_sqr)) {
    return std::nullopt;
  }
  return mahalanobis_dist_sqr;
}

double MaxRelativeDepthUncertainty(
    const Eigen::Vector3d& xyz,
    const Eigen::Matrix3d& xyz_cov,
    const std::vector<Rigid3d>& cams_from_world) {
  double max_relative_depth_uncertainty = 0;
  for (const Rigid3d& cam_from_world : cams_from_world) {
    const Eigen::Vector3d ray_in_world = xyz - cam_from_world.TgtOriginInSrc();
    const double depth = ray_in_world.norm();
    if (!(depth > std::numeric_limits<double>::epsilon())) {
      return std::numeric_limits<double>::infinity();
    }
    const Eigen::Vector3d ray_dir = ray_in_world / depth;
    const double var_along_ray = (ray_dir.transpose() * xyz_cov * ray_dir)(0, 0);
    const double sigma_along_ray = std::sqrt(std::max(var_along_ray, 0.0));
    max_relative_depth_uncertainty =
        std::max(max_relative_depth_uncertainty, sigma_along_ray / depth);
  }
  return max_relative_depth_uncertainty;
}

bool EstimateCovariantTriangulation(
    const CovariantTriangulationOptions& options,
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Matrix2d>& points2D_cov,
    const std::vector<Rigid3d>& cams_from_world,
    const std::vector<Eigen::Matrix6d>& pose_covs,
    const std::vector<Camera const*>& cameras,
    std::vector<char>* inlier_mask,
    Eigen::Vector3d* xyz,
    Eigen::Matrix3d* xyz_cov) {
  THROW_CHECK_NOTNULL(inlier_mask);
  THROW_CHECK_NOTNULL(xyz);
  THROW_CHECK_NOTNULL(xyz_cov);
  THROW_CHECK_GE(points2D.size(), 2);
  THROW_CHECK_EQ(points2D.size(), cams_from_world.size());
  THROW_CHECK_EQ(points2D.size(), cameras.size());
  THROW_CHECK(points2D_cov.empty() || points2D_cov.size() == points2D.size());
  THROW_CHECK(pose_covs.empty() || pose_covs.size() == points2D.size());
  options.Check();

  std::vector<CovariantTriangulationEstimator::PointData> point_data(
      points2D.size());
  std::vector<CovariantTriangulationEstimator::PoseData> pose_data(
      points2D.size());
  for (size_t i = 0; i < points2D.size(); ++i) {
    point_data[i].img_point = points2D[i];
    // Unit bearing in the camera frame, as in EstimateTriangulation.
    point_data[i].cam_ray = cameras[i]
                                ->CamRayFromImg(points2D[i])
                                .value_or(Eigen::Vector3d::UnitZ());
    if (!points2D_cov.empty()) {
      point_data[i].img_cov = points2D_cov[i];
    }
    pose_data[i].cam_from_world = cams_from_world[i].ToMatrix();
    pose_data[i].camera = cameras[i];
    if (!pose_covs.empty()) {
      pose_data[i].pose_cov = pose_covs[i];
    }
  }

  // Robustly estimate track using LORANSAC with whitened refinement.
  CovariantTriangulationOptions options_(options);
  options_.ransac_options.max_error = std::sqrt(options.inlier_chi2_threshold);
  LORANSAC<CovariantTriangulationEstimator,
           CovariantTriangulationEstimator,
           InlierSupportMeasurer,
           CombinationSampler>
      ransac(options_.ransac_options,
             CovariantTriangulationEstimator(),
             CovariantTriangulationEstimator());
  auto report = ransac.Estimate(point_data, pose_data);
  if (!report.success) {
    return false;
  }

  std::vector<CovariantTriangulationEstimator::PointData> inlier_point_data;
  std::vector<CovariantTriangulationEstimator::PoseData> inlier_pose_data;
  std::vector<Rigid3d> inlier_cams_from_world;
  const auto collect_inliers = [&]() {
    inlier_point_data.clear();
    inlier_pose_data.clear();
    inlier_cams_from_world.clear();
    for (size_t i = 0; i < points2D.size(); ++i) {
      if (report.inlier_mask[i]) {
        inlier_point_data.push_back(point_data[i]);
        inlier_pose_data.push_back(pose_data[i]);
        inlier_cams_from_world.push_back(cams_from_world[i]);
      }
    }
  };
  collect_inliers();

  // LORANSAC only locally optimizes models with more than kMinNumSamples
  // inliers, so e.g. two-view tracks would otherwise keep the
  // covariance-agnostic DLT seed. Refine such models over their inliers and
  // keep the result, unless it loses support.
  Eigen::Vector3d refined_xyz = report.model;
  if (report.support.num_inliers <=
          CovariantTriangulationEstimator::kMinNumSamples &&
      CovariantTriangulationEstimator::Refine(
          inlier_point_data, inlier_pose_data, &refined_xyz)) {
    std::vector<double> residuals;
    CovariantTriangulationEstimator::Residuals(
        point_data, pose_data, refined_xyz, &residuals);
    std::vector<char> refined_inlier_mask(points2D.size(), false);
    size_t num_refined_inliers = 0;
    for (size_t i = 0; i < residuals.size(); ++i) {
      if (residuals[i] <= options.inlier_chi2_threshold) {
        refined_inlier_mask[i] = true;
        ++num_refined_inliers;
      }
    }
    if (num_refined_inliers >= report.support.num_inliers) {
      report.model = refined_xyz;
      report.inlier_mask = std::move(refined_inlier_mask);
      report.support.num_inliers = num_refined_inliers;
      collect_inliers();
    }
  }

  // Estimate the point covariance over the inliers and gate on the relative
  // depth uncertainty (scale-free degeneracy check).
  const std::optional<Eigen::Matrix3d> cov =
      CovariantTriangulationEstimator::PointCovariance(
          inlier_point_data, inlier_pose_data, report.model);
  if (!cov.has_value()) {
    return false;
  }
  if (!(MaxRelativeDepthUncertainty(report.model, *cov, inlier_cams_from_world) <=
        options.max_relative_depth_uncertainty)) {
    return false;
  }

  *inlier_mask = std::move(report.inlier_mask);
  *xyz = report.model;
  *xyz_cov = *cov;

  return true;
}

}  // namespace colmap
