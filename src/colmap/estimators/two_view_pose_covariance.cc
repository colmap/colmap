// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/two_view_pose_covariance.h"

#include "colmap/estimators/cost_functions/tiny_manifold.h"
#include "colmap/estimators/cost_functions/tiny_sampson_error.h"
#include "colmap/geometry/pose.h"
#include "colmap/math/math.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/string.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"
#include "colmap/util/types.h"

#include <algorithm>
#include <atomic>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace colmap {

namespace {

// Relative tolerance below which an eigenvalue of an information matrix is
// considered numerically zero.
constexpr double kMinRelativeEigenvalue = 1e-12;

// Camera rays, with their Jacobians w.r.t. the image points, of the inlier
// correspondences.
struct InlierCamRaysWithJac {
  std::vector<CamRayWithJac> rays1;
  std::vector<CamRayWithJac> rays2;
};

InlierCamRaysWithJac ExtractInlierCamRaysWithJac(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches) {
  InlierCamRaysWithJac rays;
  rays.rays1.reserve(matches.size());
  rays.rays2.reserve(matches.size());
  for (const FeatureMatch& match : matches) {
    THROW_CHECK_LT(match.point2D_idx1, points1.size());
    THROW_CHECK_LT(match.point2D_idx2, points2.size());
    const auto ray1 = camera1.CamRayFromImgWithJac(points1[match.point2D_idx1]);
    const auto ray2 = camera2.CamRayFromImgWithJac(points2[match.point2D_idx2]);
    if (ray1.has_value() && ray2.has_value() && !ray1->ray.isZero() &&
        !ray2->ray.isZero()) {
      rays.rays1.push_back(*ray1);
      rays.rays2.push_back(*ray2);
    }
  }
  return rays;
}

// Robustly estimates the standard deviation of the observation noise from the
// absolute residuals of a model with num_params parameters.
double EstimateObservationNoise(std::vector<double> abs_residuals,
                                int num_params,
                                double min_sigma) {
  // Ratio between the standard deviation and the median absolute value of a
  // zero-mean normal distribution, i.e., 1 / Phi^-1(3/4), where Phi is its
  // cumulative distribution function.
  constexpr double kNormalMadScale = 1.482602218505602;
  // Correct for the degrees of freedom absorbed by the estimated parameters.
  const double num_residuals = static_cast<double>(abs_residuals.size());
  const double dof_scale =
      std::sqrt(num_residuals / std::max(1.0, num_residuals - num_params));
  const double sigma = kNormalMadScale * dof_scale * Median(abs_residuals);
  return std::max(sigma, min_sigma);
}

// Information matrix of the relative rotation under a pure-rotation
// (zero-baseline) model, in which each correspondence constrains the 2D
// tangent plane orthogonal to ray2 via c_i = B2^T (R21 * ray1).
std::optional<Eigen::Matrix3d> EstimatePanoramicRotationInformation(
    const InlierCamRaysWithJac& rays,
    const Eigen::Quaterniond& cam2_from_cam1_rotation,
    double min_sigma_obs_px) {
  const Eigen::Matrix3d R21 = cam2_from_cam1_rotation.toRotationMatrix();
  Eigen::Matrix3d unscaled_information = Eigen::Matrix3d::Zero();
  std::vector<double> abs_residuals;
  abs_residuals.reserve(2 * rays.rays1.size());

  for (size_t i = 0; i < rays.rays1.size(); ++i) {
    const Eigen::Vector3d& x1 = rays.rays1[i].ray;
    const Eigen::Matrix3x2d& J1 = rays.rays1[i].jacobian;
    const Eigen::Vector3d& x2 = rays.rays2[i].ray;
    const Eigen::Matrix3x2d& J2 = rays.rays2[i].jacobian;

    Eigen::Matrix<double, 3, 2, Eigen::RowMajor> B2;
    SphereManifold<3>().PlusJacobian(x2.data(), B2.data());

    // Whiten the residual with its covariance propagated from the image noise.
    const Eigen::Vector2d c_i = B2.transpose() * (R21 * x1);
    const Eigen::Matrix3x2d R21_J1 = R21 * J1;
    const Eigen::Matrix2d C_i =
        B2.transpose() * (R21_J1 * R21_J1.transpose() + J2 * J2.transpose()) *
        B2;
    const Eigen::LLT<Eigen::Matrix2d> llt(C_i);
    if (llt.info() != Eigen::Success) {
      continue;
    }

    // d(R21 * Exp(delta) * x1)/d(delta) = -R21 * [x1]_x.
    Eigen::Matrix3d x1_skew;
    x1_skew << 0.0, -x1.z(), x1.y(), x1.z(), 0.0, -x1.x(), -x1.y(), x1.x(), 0.0;
    const Eigen::Matrix<double, 2, 3> J_c = -B2.transpose() * R21 * x1_skew;

    const Eigen::Vector2d e_i = llt.matrixL().solve(c_i);
    const Eigen::Matrix<double, 2, 3> J_R_i = llt.matrixL().solve(J_c);

    abs_residuals.push_back(std::abs(e_i.x()));
    abs_residuals.push_back(std::abs(e_i.y()));
    unscaled_information.noalias() += J_R_i.transpose() * J_R_i;
  }

  if (abs_residuals.size() < 6) {
    return std::nullopt;
  }
  const double sigma_obs_px = EstimateObservationNoise(
      std::move(abs_residuals), /*num_params=*/3, min_sigma_obs_px);
  return unscaled_information / (sigma_obs_px * sigma_obs_px);
}

// Information matrix of the relative pose from the tangent Sampson errors,
// w.r.t. the right perturbation of the rotation and the perturbation of the
// translation direction on the tangent space of S^2.
Eigen::Matrix<double, 5, 5> EstimateRelativePoseInformation(
    const InlierCamRaysWithJac& rays,
    const Rigid3d& cam2_from_cam1,
    double min_sigma_obs_px) {
  using RelativePoseManifold =
      ProductManifold<EigenQuaternionManifold, SphereManifold<3>>;

  const RelPoseParams params = RelPoseParamsFromRigid3d(cam2_from_cam1);

  const size_t num_inliers = rays.rays1.size();
  const TinyTangentSampsonErrorCostFunctor cost_fn(rays.rays1, rays.rays2);
  Eigen::VectorXd residuals(num_inliers);
  Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::ColMajor> J_amb(num_inliers,
                                                                  7);
  cost_fn(params.data(), residuals.data(), J_amb.data());

  Eigen::Matrix<double, 7, 5, Eigen::RowMajor> plus_jac;
  RelativePoseManifold().PlusJacobian(params.data(), plus_jac.data());
  const Eigen::Matrix<double, Eigen::Dynamic, 5> J_tan = J_amb * plus_jac;

  std::vector<double> abs_residuals(num_inliers);
  for (size_t i = 0; i < num_inliers; ++i) {
    abs_residuals[i] = std::abs(residuals(i));
  }
  const double sigma_obs_px = EstimateObservationNoise(
      std::move(abs_residuals), /*num_params=*/5, min_sigma_obs_px);
  return (J_tan.transpose() * J_tan) / (sigma_obs_px * sigma_obs_px);
}

// Marginalizes the translation direction out of the relative pose information
// with the Schur complement, excluding only numerically null directions of its
// information.
std::optional<Eigen::Matrix3d> MarginalizeTranslationDirection(
    const Eigen::Matrix<double, 5, 5>& information) {
  const Eigen::Matrix3d Lambda_RR = information.topLeftCorner<3, 3>();
  const Eigen::Matrix<double, 3, 2> Lambda_Rt =
      information.topRightCorner<3, 2>();
  const Eigen::Matrix2d Lambda_tt = information.bottomRightCorner<2, 2>();

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> trans_eig(Lambda_tt);
  if (trans_eig.info() != Eigen::Success) {
    return std::nullopt;
  }
  const Eigen::Vector2d& trans_evals = trans_eig.eigenvalues();
  const Eigen::Matrix2d& trans_evecs = trans_eig.eigenvectors();
  const double trans_thresh =
      kMinRelativeEigenvalue * std::max(0.0, trans_evals(1));

  Eigen::Matrix2d Lambda_tt_pinv = Eigen::Matrix2d::Zero();
  for (int j = 0; j < 2; ++j) {
    if (trans_evals(j) > trans_thresh) {
      Lambda_tt_pinv.noalias() +=
          (1.0 / trans_evals(j)) *
          (trans_evecs.col(j) * trans_evecs.col(j).transpose());
    }
  }

  return Lambda_RR - Lambda_Rt * Lambda_tt_pinv * Lambda_Rt.transpose();
}

// Inverts the rotation information into a covariance. Returns nullopt if the
// rotation is degenerate, i.e., if its standard deviation along the least
// constrained axis exceeds max_sigma_deg.
std::optional<Eigen::Matrix3d> RotationCovarianceFromInformation(
    const Eigen::Matrix3d& information, double max_sigma_deg) {
  const double max_sigma = DegToRad(max_sigma_deg);
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(
      0.5 * (information + information.transpose()));
  if (eig.info() != Eigen::Success ||
      eig.eigenvalues()(0) * max_sigma * max_sigma <= 1.0) {
    return std::nullopt;
  }

  // Clamp the anisotropy of the covariance to avoid near-singular whitening.
  constexpr double kMaxConditionNumber = 1e4;
  Eigen::Vector3d variances = eig.eigenvalues().cwiseInverse();
  variances = variances.cwiseMax(variances.maxCoeff() / kMaxConditionNumber);

  const Eigen::Matrix3d cov = eig.eigenvectors() * variances.asDiagonal() *
                              eig.eigenvectors().transpose();
  return 0.5 * (cov + cov.transpose());
}

}  // namespace

std::optional<Eigen::Matrix3d> EstimateTwoViewPoseCovariance(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const TwoViewGeometry& geometry,
    const TwoViewPoseCovarianceOptions& options) {
  if (!geometry.cam2_from_cam1.has_value()) {
    return std::nullopt;
  }
  const Rigid3d& cam2_from_cam1 = *geometry.cam2_from_cam1;

  // Use the intrinsics estimated with the two-view geometry, if any.
  const InlierCamRaysWithJac rays = ExtractInlierCamRaysWithJac(
      geometry.camera1.has_value() ? *geometry.camera1 : camera1,
      points1,
      geometry.camera2.has_value() ? *geometry.camera2 : camera2,
      points2,
      geometry.inlier_matches);

  const bool is_panoramic =
      geometry.config == TwoViewGeometry::ConfigurationType::PANORAMIC ||
      cam2_from_cam1.translation().squaredNorm() < 1e-12;
  const size_t num_inliers = rays.rays1.size();
  if (num_inliers < (is_panoramic ? 3 : 5)) {
    return std::nullopt;
  }

  std::optional<Eigen::Matrix3d> rotation_information;
  if (is_panoramic) {
    rotation_information = EstimatePanoramicRotationInformation(
        rays, cam2_from_cam1.rotation(), options.min_sigma_obs_px);
  } else {
    rotation_information =
        MarginalizeTranslationDirection(EstimateRelativePoseInformation(
            rays, cam2_from_cam1, options.min_sigma_obs_px));
  }
  if (!rotation_information.has_value()) {
    return std::nullopt;
  }

  return RotationCovarianceFromInformation(*rotation_information,
                                           options.max_rotation_sigma_deg);
}

namespace {

NodeHashMap<image_t, std::vector<Eigen::Vector2d>> ExtractImagePoints(
    const DatabaseCache& database_cache,
    const FlatHashSet<image_t>& image_ids) {
  NodeHashMap<image_t, std::vector<Eigen::Vector2d>> image_points;
  image_points.reserve(image_ids.size());
  for (const image_t image_id : image_ids) {
    const Image& image = database_cache.Image(image_id);
    std::vector<Eigen::Vector2d> points;
    points.reserve(image.NumPoints2D());
    for (const Point2D& point : image.Points2D()) {
      points.push_back(point.xy);
    }
    image_points.emplace(image_id, std::move(points));
  }
  return image_points;
}

// Estimates the rotation covariance of a pose graph edge, evaluated at its
// relative pose rather than at the one stored in the correspondence graph.
std::optional<Eigen::Matrix3d> EstimateEdgeRotationCovariance(
    const DatabaseCache& database_cache,
    const CorrespondenceGraph& correspondence_graph,
    const NodeHashMap<image_t, std::vector<Eigen::Vector2d>>& image_points,
    image_pair_t pair_id,
    const Rigid3d& cam2_from_cam1,
    const TwoViewPoseCovarianceOptions& options) {
  const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
  TwoViewGeometry two_view_geometry =
      correspondence_graph.ExtractTwoViewGeometry(
          image_id1, image_id2, /*extract_inlier_matches=*/true);
  two_view_geometry.cam2_from_cam1 = cam2_from_cam1;

  return EstimateTwoViewPoseCovariance(
      database_cache.Camera(database_cache.Image(image_id1).CameraId()),
      image_points.at(image_id1),
      database_cache.Camera(database_cache.Image(image_id2).CameraId()),
      image_points.at(image_id2),
      two_view_geometry,
      options);
}

}  // namespace

void EstimatePoseGraphCovariances(const DatabaseCache& database_cache,
                                  PoseGraph& pose_graph,
                                  const TwoViewPoseCovarianceOptions& options) {
  Timer timer;
  timer.Start();

  std::vector<std::pair<image_pair_t, PoseGraph::Edge*>> edges;
  edges.reserve(pose_graph.NumEdges());
  FlatHashSet<image_t> image_ids;
  for (auto& [pair_id, edge] : pose_graph.Edges()) {
    if (!edge.valid || edge.cam2_from_cam1_rotation_cov.has_value()) {
      continue;
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (!database_cache.ExistsImage(image_id1) ||
        !database_cache.ExistsImage(image_id2)) {
      continue;
    }
    image_ids.insert(image_id1);
    image_ids.insert(image_id2);
    edges.emplace_back(pair_id, &edge);
  }

  if (edges.empty()) {
    return;
  }

  const auto image_points = ExtractImagePoints(database_cache, image_ids);
  const auto correspondence_graph = database_cache.CorrespondenceGraph();

  std::atomic<size_t> num_degenerate(0);
  ThreadPool thread_pool(GetEffectiveNumThreads(options.num_threads));
  std::vector<std::shared_future<void>> futures;
  futures.reserve(edges.size());
  for (const auto& pair_id_and_edge : edges) {
    const image_pair_t pair_id = pair_id_and_edge.first;
    PoseGraph::Edge* edge = pair_id_and_edge.second;
    futures.push_back(thread_pool.AddTask([&, pair_id, edge]() {
      edge->cam2_from_cam1_rotation_cov =
          EstimateEdgeRotationCovariance(database_cache,
                                         *correspondence_graph,
                                         image_points,
                                         pair_id,
                                         edge->cam2_from_cam1,
                                         options);
      if (!edge->cam2_from_cam1_rotation_cov.has_value()) {
        num_degenerate.fetch_add(1, std::memory_order_relaxed);
      }
    }));
  }
  for (auto& future : futures) {
    future.get();
  }

  LOG(INFO) << StringPrintf(
      "Estimated %zu two-view rotation covariances (%zu degenerate) in %.3fs",
      edges.size(),
      num_degenerate.load(),
      timer.ElapsedSeconds());
}

}  // namespace colmap
