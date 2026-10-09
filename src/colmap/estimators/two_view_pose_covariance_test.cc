// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/two_view_pose_covariance.h"

#include "colmap/estimators/solvers/essential_matrix.h"
#include "colmap/estimators/solvers/utils.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/testing.h"

#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

struct TwoViewPoseCovarianceTestData {
  Camera camera1;
  Camera camera2;
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  TwoViewGeometry geometry;
};

// Noise-free calibrated two-view geometry with ground-truth relative pose.
TwoViewPoseCovarianceTestData CreateTwoViewPoseCovarianceTestData() {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const Image& image1 = reconstruction.Image(1);
  const Image& image2 = reconstruction.Image(2);

  TwoViewPoseCovarianceTestData data;
  data.camera1 = reconstruction.Camera(image1.CameraId());
  data.camera2 = reconstruction.Camera(image2.CameraId());
  data.geometry.config = TwoViewGeometry::ConfigurationType::CALIBRATED;
  data.geometry.cam2_from_cam1 =
      image2.CamFromWorld() * Inverse(image1.CamFromWorld());

  for (const Point2D& point2D : image1.Points2D()) {
    data.points1.push_back(point2D.xy);
  }
  for (const Point2D& point2D : image2.Points2D()) {
    data.points2.push_back(point2D.xy);
  }
  for (const auto& [_, point3D] : reconstruction.Points3D()) {
    const Track& track = point3D.track;
    CHECK_EQ(track.Length(), 2);
    const TrackElement& elem1 = track.Element(0);
    const TrackElement& elem2 = track.Element(1);
    if (elem1.image_id == image1.ImageId()) {
      data.geometry.inlier_matches.emplace_back(elem1.point2D_idx,
                                                elem2.point2D_idx);
    } else {
      data.geometry.inlier_matches.emplace_back(elem2.point2D_idx,
                                                elem1.point2D_idx);
    }
  }

  return data;
}

struct TwoViewPoseCovarianceTestCase {
  std::string name;
  // Relative rotation angle and baseline (scene depth is in [4, 8]). A zero
  // baseline denotes a pure rotation.
  double rotation_deg = 10.0;
  Eigen::Vector3d translation = Eigen::Vector3d(1, 0, 0);
  // Whether points lie on a single plane.
  bool planar = false;
  // Whether points only cover the left half of the first image.
  bool one_sided = false;
};

class ParameterizedTwoViewPoseCovarianceTests
    : public ::testing::TestWithParam<TwoViewPoseCovarianceTestCase> {};

// Maximum-likelihood rotation-only estimate from noisy bearings via
// Gauss-Newton on the whitened tangent-plane residuals of the second bearing.
Eigen::Quaterniond RefineRotationOnlyMLE(
    const std::vector<CamRayWithJac>& rays1,
    const std::vector<CamRayWithJac>& rays2,
    const Eigen::Quaterniond& cam2_from_cam1_init) {
  Eigen::Quaterniond cam2_from_cam1 = cam2_from_cam1_init;
  for (int iter = 0; iter < 20; ++iter) {
    const Eigen::Matrix3d R = cam2_from_cam1.toRotationMatrix();
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < rays1.size(); ++i) {
      const Eigen::Vector3d& x1 = rays1[i].ray;
      const Eigen::Vector3d& x2 = rays2[i].ray;
      // Orthonormal basis of the tangent plane at x2.
      const Eigen::Vector3d a = x2.unitOrthogonal();
      Eigen::Matrix<double, 3, 2> B2;
      B2 << a, x2.cross(a);
      const Eigen::Matrix<double, 3, 2> R_J1 = R * rays1[i].jacobian;
      const Eigen::Matrix2d C =
          B2.transpose() *
          (R_J1 * R_J1.transpose() +
           rays2[i].jacobian * rays2[i].jacobian.transpose()) *
          B2;
      const Eigen::Matrix2d W = C.inverse();
      const Eigen::Vector2d e = B2.transpose() * (R * x1);
      Eigen::Matrix3d x1_skew;
      x1_skew << 0, -x1.z(), x1.y(), x1.z(), 0, -x1.x(), -x1.y(), x1.x(), 0;
      const Eigen::Matrix<double, 2, 3> J = -B2.transpose() * R * x1_skew;
      H += J.transpose() * W * J;
      g += J.transpose() * W * e;
    }
    const Eigen::Vector3d delta = -H.ldlt().solve(g);
    if (delta.norm() < 1e-12) {
      break;
    }
    cam2_from_cam1 = cam2_from_cam1 * Eigen::Quaterniond(Eigen::AngleAxisd(
                                          delta.norm(), delta.normalized()));
    cam2_from_cam1.normalize();
  }
  return cam2_from_cam1;
}

// Checks that the estimated rotation covariance is calibrated, i.e., that the
// normalized estimation error squared (NEES) of the maximum-likelihood rotation
// follows a chi-squared distribution with 3 degrees of freedom.
TEST_P(ParameterizedTwoViewPoseCovarianceTests, MonteCarloCalibration) {
  const TwoViewPoseCovarianceTestCase& test_case = GetParam();
  SetPRNGSeed(0);

  Camera camera = Camera::CreateFromModelId(
      /*camera_id=*/1,
      SimplePinholeCameraModel::model_id,
      /*focal_length=*/500.0,
      /*width=*/640,
      /*height=*/480);
  camera.has_prior_focal_length = true;

  const bool is_panoramic = test_case.translation.isZero();
  const Rigid3d cam2_from_cam1(Eigen::Quaterniond(Eigen::AngleAxisd(
                                   DegToRad(test_case.rotation_deg),
                                   Eigen::Vector3d(0.2, 1, 0.1).normalized())),
                               test_case.translation);

  constexpr int kNumPoints = 100;
  constexpr int kNumTrials = 2000;
  constexpr double kPoint2DStddev = 1.0;

  std::vector<Eigen::Vector3d> points3D;
  while (points3D.size() < kNumPoints) {
    const double x = RandomUniformReal(test_case.one_sided ? 0.0 : 40.0, 600.0);
    const double y = RandomUniformReal(40.0, 440.0);
    const double depth = test_case.planar
                             ? 6.0 + 0.3 * (x - 320.0) / 500.0 * 6.0
                             : RandomUniformReal(4.0, 8.0);
    const Eigen::Vector3d point3D =
        depth * camera.CamFromImg(Eigen::Vector2d(x, y))->homogeneous();
    const Eigen::Vector3d point_in_cam2 = cam2_from_cam1 * point3D;
    const std::optional<Eigen::Vector2d> xy2 = camera.ImgFromCam(point_in_cam2);
    if (point_in_cam2.z() < 0.5 || !xy2.has_value() || xy2->x() < 0 ||
        xy2->y() < 0 || xy2->x() > camera.width || xy2->y() > camera.height) {
      continue;
    }
    points3D.push_back(point3D);
  }

  TwoViewGeometry geometry;
  geometry.config = is_panoramic
                        ? TwoViewGeometry::ConfigurationType::PANORAMIC
                        : TwoViewGeometry::ConfigurationType::CALIBRATED;
  for (int i = 0; i < kNumPoints; ++i) {
    geometry.inlier_matches.emplace_back(i, i);
  }

  TwoViewPoseCovarianceOptions options;
  options.min_sigma_obs_px = 0.01;

  std::vector<double> nees;
  nees.reserve(kNumTrials);
  double sum_sigma_obs = 0;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    std::vector<Eigen::Vector2d> points1(kNumPoints);
    std::vector<Eigen::Vector2d> points2(kNumPoints);
    std::vector<CamRayWithJac> rays1(kNumPoints);
    std::vector<CamRayWithJac> rays2(kNumPoints);
    for (int i = 0; i < kNumPoints; ++i) {
      points1[i] = *camera.ImgFromCam(points3D[i]) +
                   kPoint2DStddev * Eigen::Vector2d(RandomGaussian(0.0, 1.0),
                                                    RandomGaussian(0.0, 1.0));
      points2[i] = *camera.ImgFromCam(cam2_from_cam1 * points3D[i]) +
                   kPoint2DStddev * Eigen::Vector2d(RandomGaussian(0.0, 1.0),
                                                    RandomGaussian(0.0, 1.0));
      rays1[i] = *camera.CamRayFromImgWithJac(points1[i]);
      rays2[i] = *camera.CamRayFromImgWithJac(points2[i]);
    }

    // Estimate the relative pose from the noisy observations, starting from
    // the ground-truth to isolate the uncertainty from local minima.
    Rigid3d estimated_cam2_from_cam1;
    if (is_panoramic) {
      estimated_cam2_from_cam1.rotation() =
          RefineRotationOnlyMLE(rays1, rays2, cam2_from_cam1.rotation());
    } else {
      Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);
      ASSERT_TRUE(
          EssentialMatrixTangentSampsonEstimator::Refine(rays1, rays2, &E));
      std::vector<int> valid_indices;
      PoseFromEssentialMatrix(E,
                              RaysFromCamRaysWithJac(rays1),
                              RaysFromCamRaysWithJac(rays2),
                              &estimated_cam2_from_cam1,
                              &valid_indices);
    }
    geometry.cam2_from_cam1 = estimated_cam2_from_cam1;

    const std::optional<TwoViewPoseCovariance> cov =
        EstimateTwoViewPoseCovariance(
            camera, points1, camera, points2, geometry, options);
    ASSERT_TRUE(cov.has_value());
    ASSERT_TRUE(cov->cov_rot.has_value());
    EXPECT_EQ(cov->num_inliers, kNumPoints);
    EXPECT_EQ(cov->cov_trans_tangent.has_value(), !is_panoramic);

    // Right perturbation: estimated = true * Exp(delta).
    const Eigen::AngleAxisd delta_angle_axis(
        cam2_from_cam1.rotation().inverse() *
        estimated_cam2_from_cam1.rotation());
    const Eigen::Vector3d delta =
        delta_angle_axis.angle() * delta_angle_axis.axis();
    nees.push_back(delta.dot(cov->cov_rot->ldlt().solve(delta)));
    sum_sigma_obs += cov->sigma_obs_px;
  }

  // The first-order covariance is exact only asymptotically, so a few trials
  // fall into a slightly heavier tail. Use robust statistics: the median of
  // chi2(3) is 2.366 (the sample median of 2000 trials has a standard deviation
  // of ~0.06, so the bounds are robust to platform-dependent random draws) and
  // only 1% of chi2(3) samples exceed 11.34.
  const double median_nees = Median(nees);
  const double tail_fraction =
      std::count_if(
          nees.begin(), nees.end(), [](double v) { return v > 11.34; }) /
      static_cast<double>(kNumTrials);
  const double mean_sigma_obs = sum_sigma_obs / kNumTrials;
  LOG(INFO) << test_case.name << ": median NEES=" << median_nees
            << ", tail fraction=" << tail_fraction
            << ", mean sigma_obs=" << mean_sigma_obs;
  EXPECT_GT(median_nees, 1.9);
  EXPECT_LT(median_nees, 2.9);
  EXPECT_LT(tail_fraction, 0.05);
  EXPECT_NEAR(mean_sigma_obs, kPoint2DStddev, 0.15);
}

INSTANTIATE_TEST_SUITE_P(
    TwoViewPoseCovarianceTests,
    ParameterizedTwoViewPoseCovarianceTests,
    ::testing::Values(
        TwoViewPoseCovarianceTestCase{"WideBaseline"},
        // ~16px median parallax. Below ~10px, the translation direction is
        // so uncertain that the first-order approximation becomes
        // overconfident for the rotation (heavier NEES tail).
        TwoViewPoseCovarianceTestCase{
            "NarrowBaseline", 5.0, Eigen::Vector3d(0.2, 0.04, 0)},
        TwoViewPoseCovarianceTestCase{
            "ForwardMotion", 5.0, Eigen::Vector3d(0.05, 0, -1)},
        TwoViewPoseCovarianceTestCase{
            "Planar", 10.0, Eigen::Vector3d(1, 0, 0), /*planar=*/true},
        TwoViewPoseCovarianceTestCase{"OneSided",
                                      10.0,
                                      Eigen::Vector3d(1, 0, 0),
                                      /*planar=*/false,
                                      /*one_sided=*/true},
        TwoViewPoseCovarianceTestCase{
            "PureRotation", 10.0, Eigen::Vector3d::Zero()}),
    [](const ::testing::TestParamInfo<TwoViewPoseCovarianceTestCase>& info) {
      return info.param.name;
    });

TEST(EstimateTwoViewPoseCovariance, Nominal) {
  SetPRNGSeed(0);
  const TwoViewPoseCovarianceTestData data =
      CreateTwoViewPoseCovarianceTestData();
  const TwoViewGeometry& geometry = data.geometry;
  ASSERT_TRUE(geometry.cam2_from_cam1.has_value());

  TwoViewPoseCovarianceOptions options;
  const std::optional<TwoViewPoseCovariance> cov =
      EstimateTwoViewPoseCovariance(data.camera1,
                                    data.points1,
                                    data.camera2,
                                    data.points2,
                                    geometry,
                                    options);
  ASSERT_TRUE(cov.has_value());
  ASSERT_TRUE(cov->cov_rot.has_value());
  EXPECT_TRUE(cov->cov_rot->isApprox(cov->cov_rot->transpose()));
  EXPECT_GT(cov->cov_rot->ldlt().vectorD().minCoeff(), 0);
  ASSERT_TRUE(cov->cov_trans_tangent.has_value());
  EXPECT_TRUE(
      cov->cov_trans_tangent->isApprox(cov->cov_trans_tangent->transpose()));
  EXPECT_GT(cov->cov_trans_tangent->ldlt().vectorD().minCoeff(), 0);
  // Noise-free observations are clamped to the minimum observation noise.
  EXPECT_EQ(cov->sigma_obs_px, options.min_sigma_obs_px);

  // The rotation is degenerate if less certain than the maximum sigma.
  const double max_sigma_deg =
      RadToDeg(std::sqrt(cov->cov_rot->eigenvalues().real().maxCoeff()));
  options.max_rotation_sigma_deg = 0.5 * max_sigma_deg;
  const std::optional<TwoViewPoseCovariance> cov_degenerate =
      EstimateTwoViewPoseCovariance(data.camera1,
                                    data.points1,
                                    data.camera2,
                                    data.points2,
                                    geometry,
                                    options);
  ASSERT_TRUE(cov_degenerate.has_value());
  EXPECT_FALSE(cov_degenerate->cov_rot.has_value());
  EXPECT_EQ(cov_degenerate->cov_trans_tangent, cov->cov_trans_tangent);
}

TEST(EstimateTwoViewPoseCovariance, MissingPoseOrTooFewInliers) {
  const TwoViewPoseCovarianceTestData data =
      CreateTwoViewPoseCovarianceTestData();
  TwoViewGeometry geometry = data.geometry;
  geometry.cam2_from_cam1.reset();
  EXPECT_FALSE(
      EstimateTwoViewPoseCovariance(
          data.camera1, data.points1, data.camera2, data.points2, geometry)
          .has_value());
  geometry = data.geometry;
  geometry.inlier_matches.resize(4);
  EXPECT_FALSE(
      EstimateTwoViewPoseCovariance(
          data.camera1, data.points1, data.camera2, data.points2, geometry)
          .has_value());
  geometry.config = TwoViewGeometry::PANORAMIC;
  geometry.inlier_matches.resize(2);
  EXPECT_FALSE(
      EstimateTwoViewPoseCovariance(
          data.camera1, data.points1, data.camera2, data.points2, geometry)
          .has_value());
}

TEST(EstimatePoseGraphCovariances, Nominal) {
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());
  PoseGraph pose_graph;
  pose_graph.Load(*cache->CorrespondenceGraph());
  ASSERT_EQ(pose_graph.NumEdges(), 6);

  // Pre-populated covariances are preserved.
  const image_pair_t preset_pair_id = pose_graph.Edges().begin()->first;
  const Eigen::Matrix3d preset_cov = 2 * Eigen::Matrix3d::Identity();
  pose_graph.Edges().begin()->second.cam2_from_cam1_rotation_cov = preset_cov;

  // Degenerate rotations are left unset.
  PoseGraph degenerate_pose_graph = pose_graph;
  TwoViewPoseCovarianceOptions options;
  options.max_rotation_sigma_deg = 1e-6;
  EstimatePoseGraphCovariances(*cache, degenerate_pose_graph, options);
  for (const auto& [pair_id, edge] : degenerate_pose_graph.Edges()) {
    EXPECT_EQ(edge.cam2_from_cam1_rotation_cov.has_value(),
              pair_id == preset_pair_id);
  }

  options = TwoViewPoseCovarianceOptions();
  EstimatePoseGraphCovariances(*cache, pose_graph, options);
  for (const auto& [pair_id, edge] : pose_graph.Edges()) {
    ASSERT_TRUE(edge.cam2_from_cam1_rotation_cov.has_value());
    if (pair_id == preset_pair_id) {
      EXPECT_EQ(*edge.cam2_from_cam1_rotation_cov, preset_cov);
      continue;
    }
    EXPECT_TRUE(edge.cam2_from_cam1_rotation_cov->isApprox(
        edge.cam2_from_cam1_rotation_cov->transpose()));
    const Eigen::Vector3d eigvals =
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(
            *edge.cam2_from_cam1_rotation_cov)
            .eigenvalues();
    EXPECT_GT(eigvals(0), 0);
    // Well-constrained synthetic pairs are certain to below a degree.
    EXPECT_LT(eigvals(2), DegToRad(1.0) * DegToRad(1.0));
  }
}

}  // namespace
}  // namespace colmap
