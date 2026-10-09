// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/two_view_pose_salvage.h"

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/sensor/models.h"

#include <random>

#include <gtest/gtest.h>

namespace colmap {
namespace {

Camera CreateTestCamera() {
  Camera camera = Camera::CreateFromModelId(
      /*camera_id=*/1,
      SimplePinholeCameraModel::model_id,
      /*focal_length=*/800.0,
      /*width=*/1000,
      /*height=*/800);
  camera.has_prior_focal_length = true;
  return camera;
}

void AppendSyntheticCorrespondences(const Camera& camera1,
                                    const Camera& camera2,
                                    const Rigid3d& cam2_from_cam1,
                                    const int num_points,
                                    const double noise_px,
                                    std::mt19937& rng,
                                    std::vector<Eigen::Vector2d>& points1,
                                    std::vector<Eigen::Vector2d>& points2,
                                    FeatureMatches& matches) {
  std::uniform_real_distribution<double> xy_dist(-2.5, 2.5);
  std::uniform_real_distribution<double> z_dist(4.0, 12.0);
  std::normal_distribution<double> noise_dist(0.0, noise_px);

  int added = 0;
  while (added < num_points) {
    const Eigen::Vector3d p_cam1(xy_dist(rng), xy_dist(rng), z_dist(rng));
    const Eigen::Vector3d p_cam2 = cam2_from_cam1 * p_cam1;
    if (p_cam1.z() <= 0.5 || p_cam2.z() <= 0.5) {
      continue;
    }
    const std::optional<Eigen::Vector2d> uv1 = camera1.ImgFromCam(p_cam1);
    const std::optional<Eigen::Vector2d> uv2 = camera2.ImgFromCam(p_cam2);
    if (!uv1.has_value() || !uv2.has_value()) {
      continue;
    }
    const point2D_t idx1 = static_cast<point2D_t>(points1.size());
    const point2D_t idx2 = static_cast<point2D_t>(points2.size());
    points1.emplace_back(
        *uv1 + Eigen::Vector2d(noise_dist(rng), noise_dist(rng)));
    points2.emplace_back(
        *uv2 + Eigen::Vector2d(noise_dist(rng), noise_dist(rng)));
    matches.emplace_back(idx1, idx2);
    ++added;
  }
}

TEST(TwoViewPoseSalvage, DominantOutlierMotionMode) {
  SetPRNGSeed(42);
  std::mt19937 rng(42);

  const Camera camera1 = CreateTestCamera();
  const Camera camera2 = CreateTestCamera();

  // True static background motion (30% of correspondences = 60 points).
  const Rigid3d true_cam2_from_cam1(
      Eigen::Quaterniond(
          Eigen::AngleAxisd(DegToRad(12.0), Eigen::Vector3d(0.2, 1.0, -0.1)
                                                .normalized())),
      Eigen::Vector3d(1.0, 0.2, -0.15).normalized());

  // Coherent dominant outlier motion (70% of correspondences = 140 points),
  // e.g. a moving object or repeated structure rotated by 45 deg.
  const Rigid3d wrong_cam2_from_cam1(
      Eigen::Quaterniond(
          Eigen::AngleAxisd(DegToRad(-40.0), Eigen::Vector3d(0.0, 1.0, 0.3)
                                                 .normalized())),
      Eigen::Vector3d(-0.6, 0.8, 0.1).normalized());

  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  FeatureMatches matches;
  const int num_static = 60;
  const int num_wrong = 140;
  AppendSyntheticCorrespondences(camera1,
                                 camera2,
                                 true_cam2_from_cam1,
                                 num_static,
                                 /*noise_px=*/0.4,
                                 rng,
                                 points1,
                                 points2,
                                 matches);
  AppendSyntheticCorrespondences(camera1,
                                 camera2,
                                 wrong_cam2_from_cam1,
                                 num_wrong,
                                 /*noise_px=*/0.4,
                                 rng,
                                 points1,
                                 points2,
                                 matches);

  // Confirm that standard ungated two-view geometry estimation locks onto the
  // 70% wrong motion mode.
  TwoViewGeometryOptions tvg_options;
  tvg_options.compute_relative_pose = true;
  tvg_options.ransac_options.random_seed = 42;
  const TwoViewGeometry ungated = EstimateCalibratedTwoViewGeometry(
      camera1, points1, camera2, points2, matches, tvg_options);
  ASSERT_TRUE(ungated.cam2_from_cam1.has_value());
  const double ungated_rot_err_deg = RadToDeg(
      ungated.cam2_from_cam1->rotation().angularDistance(
          true_cam2_from_cam1.rotation()));
  EXPECT_GT(ungated_rot_err_deg, 30.0);

  // Now run SalvageTwoViewPose with a leave-one-out prior around the true
  // rotation.
  const Eigen::Quaterniond prior_rot =
      true_cam2_from_cam1.rotation() *
      Eigen::Quaterniond(Eigen::AngleAxisd(
          DegToRad(0.5), Eigen::Vector3d(1.0, -0.5, 0.2).normalized()));
  const Eigen::Matrix3d prior_cov =
      std::pow(DegToRad(1.0), 2) * Eigen::Matrix3d::Identity();

  TwoViewPoseSalvageOptions salvage_options;
  salvage_options.random_seed = 42;
  salvage_options.num_threads = 1;
  salvage_options.min_num_inliers = 30;

  const std::optional<SalvagedTwoViewPose> salvaged =
      SalvageTwoViewPose(camera1,
                         points1,
                         camera2,
                         points2,
                         matches,
                         prior_rot,
                         prior_cov,
                         salvage_options);
  ASSERT_TRUE(salvaged.has_value());
  ASSERT_TRUE(salvaged->geometry.cam2_from_cam1.has_value());

  const double rot_err_deg = RadToDeg(
      salvaged->geometry.cam2_from_cam1->rotation().angularDistance(
          true_cam2_from_cam1.rotation()));
  const double trans_cos =
      salvaged->geometry.cam2_from_cam1->translation().normalized().dot(
          true_cam2_from_cam1.translation().normalized());
  const double trans_err_deg = RadToDeg(std::acos(std::clamp(trans_cos, -1.0, 1.0)));

  EXPECT_LT(rot_err_deg, 0.5);
  EXPECT_LT(trans_err_deg, 1.5);
  EXPECT_GE(salvaged->geometry.inlier_matches.size(), 50u);
  EXPECT_GT(salvaged->posterior_p_value, salvage_options.posterior_significance);

  // Verify that the salvaged inliers come overwhelmingly from the static set
  // [0, num_static).
  int static_inliers = 0;
  for (const auto& m : salvaged->geometry.inlier_matches) {
    if (m.point2D_idx1 < static_cast<point2D_t>(num_static)) {
      ++static_inliers;
    }
  }
  EXPECT_GE(static_inliers, 50);
}

TEST(TwoViewPoseSalvage, NoisyRotationPriorTolerance) {
  SetPRNGSeed(123);
  std::mt19937 rng(123);

  const Camera camera1 = CreateTestCamera();
  const Camera camera2 = CreateTestCamera();

  const Rigid3d true_cam2_from_cam1(
      Eigen::Quaterniond(
          Eigen::AngleAxisd(DegToRad(15.0), Eigen::Vector3d(-0.3, 1.0, 0.2)
                                                .normalized())),
      Eigen::Vector3d(0.8, -0.3, 0.2).normalized());

  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  FeatureMatches matches;
  AppendSyntheticCorrespondences(camera1,
                                 camera2,
                                 true_cam2_from_cam1,
                                 /*num_points=*/80,
                                 /*noise_px=*/0.5,
                                 rng,
                                 points1,
                                 points2,
                                 matches);

  // Add 40 gross outlier matches.
  std::uniform_real_distribution<double> u_dist(50.0, 950.0);
  std::uniform_real_distribution<double> v_dist(50.0, 750.0);
  for (int i = 0; i < 40; ++i) {
    const point2D_t idx1 = static_cast<point2D_t>(points1.size());
    const point2D_t idx2 = static_cast<point2D_t>(points2.size());
    points1.emplace_back(u_dist(rng), v_dist(rng));
    points2.emplace_back(u_dist(rng), v_dist(rng));
    matches.emplace_back(idx1, idx2);
  }

  // Inject a 3-degree error into the leave-one-out rotation prior, consistent
  // with prior covariance (3 deg)^2 * I_3.
  const Eigen::Quaterniond noisy_prior_rot =
      true_cam2_from_cam1.rotation() *
      Eigen::Quaterniond(Eigen::AngleAxisd(
          DegToRad(3.0), Eigen::Vector3d(0.6, -0.7, 0.4).normalized()));
  const Eigen::Matrix3d prior_cov =
      std::pow(DegToRad(3.0), 2) * Eigen::Matrix3d::Identity();

  TwoViewPoseSalvageOptions salvage_options;
  salvage_options.random_seed = 123;
  salvage_options.num_threads = 1;

  const std::optional<SalvagedTwoViewPose> salvaged =
      SalvageTwoViewPose(camera1,
                         points1,
                         camera2,
                         points2,
                         matches,
                         noisy_prior_rot,
                         prior_cov,
                         salvage_options);
  ASSERT_TRUE(salvaged.has_value());
  ASSERT_TRUE(salvaged->geometry.cam2_from_cam1.has_value());

  // Unweighted 5-DoF tangent Sampson refinement should converge to the true
  // relative rotation (much closer than the 3-deg prior).
  const double rot_err_deg = RadToDeg(
      salvaged->geometry.cam2_from_cam1->rotation().angularDistance(
          true_cam2_from_cam1.rotation()));
  EXPECT_LT(rot_err_deg, 0.3);
  EXPECT_GT(salvaged->posterior_p_value, 0.05);
}

TEST(TwoViewPoseSalvage, RejectsPureRotationAndIncompatiblePrior) {
  SetPRNGSeed(7);
  std::mt19937 rng(7);

  const Camera camera1 = CreateTestCamera();
  const Camera camera2 = CreateTestCamera();

  // Case 1: Pure rotation (zero baseline -> triangulation angle ~ 0).
  const Rigid3d pure_rot_cam2_from_cam1(
      Eigen::Quaterniond(
          Eigen::AngleAxisd(DegToRad(10.0), Eigen::Vector3d::UnitY())),
      Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  FeatureMatches matches;
  AppendSyntheticCorrespondences(camera1,
                                 camera2,
                                 pure_rot_cam2_from_cam1,
                                 /*num_points=*/60,
                                 /*noise_px=*/0.2,
                                 rng,
                                 points1,
                                 points2,
                                 matches);

  TwoViewPoseSalvageOptions salvage_options;
  salvage_options.random_seed = 7;
  salvage_options.num_threads = 1;
  const Eigen::Matrix3d prior_cov =
      std::pow(DegToRad(1.0), 2) * Eigen::Matrix3d::Identity();

  EXPECT_FALSE(SalvageTwoViewPose(camera1,
                                  points1,
                                  camera2,
                                  points2,
                                  matches,
                                  pure_rot_cam2_from_cam1.rotation(),
                                  prior_cov,
                                  salvage_options)
                   .has_value());

  // Case 2: Valid baseline, but prior rotation is 25 degrees away with tight
  // covariance.
  points1.clear();
  points2.clear();
  matches.clear();
  const Rigid3d valid_cam2_from_cam1(
      Eigen::Quaterniond(
          Eigen::AngleAxisd(DegToRad(10.0), Eigen::Vector3d::UnitY())),
      Eigen::Vector3d(1.0, 0.0, 0.0));
  AppendSyntheticCorrespondences(camera1,
                                 camera2,
                                 valid_cam2_from_cam1,
                                 /*num_points=*/60,
                                 /*noise_px=*/0.2,
                                 rng,
                                 points1,
                                 points2,
                                 matches);
  const Eigen::Quaterniond incompatible_prior =
      valid_cam2_from_cam1.rotation() *
      Eigen::Quaterniond(
          Eigen::AngleAxisd(DegToRad(25.0), Eigen::Vector3d::UnitX()));
  const Eigen::Matrix3d tight_cov =
      std::pow(DegToRad(0.5), 2) * Eigen::Matrix3d::Identity();
  EXPECT_FALSE(SalvageTwoViewPose(camera1,
                                  points1,
                                  camera2,
                                  points2,
                                  matches,
                                  incompatible_prior,
                                  tight_cov,
                                  salvage_options)
                   .has_value());
}

TEST(TwoViewPoseSalvage, SalvageTwoViewPosesUpdatesGraphAndCorrespondences) {
  SetPRNGSeed(99);
  std::mt19937 rng(99);

  DatabaseCache cache;
  Camera camera = CreateTestCamera();
  cache.AddCamera(camera);

  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(camera.SensorId());
  cache.AddRig(rig);

  const int num_images = 4;
  std::vector<Rigid3d> cams_from_world(num_images + 1);
  for (int i = 1; i <= num_images; ++i) {
    const double angle = DegToRad(8.0 * (i - 1));
    cams_from_world[i] = Rigid3d(
        Eigen::Quaterniond(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY())),
        Eigen::Vector3d(-0.8 * (i - 1), 0.1 * ((i % 2) ? 1 : -1), 0.0));

    Frame frame;
    frame.SetFrameId(i);
    frame.SetRigId(1);
    Image img;
    img.SetImageId(i);
    img.SetName("img_" + std::to_string(i) + ".png");
    img.SetCameraId(camera.camera_id);
    img.SetFrameId(i);
    frame.AddDataId(img.DataId());
    cache.AddFrame(frame);
    cache.AddImage(img);
  }

  // Synthesize 2D points per image and matches for all pairs.
  std::vector<std::vector<Eigen::Vector2d>> points_per_image(num_images + 1);
  struct PairData {
    image_t id1;
    image_t id2;
    FeatureMatches static_matches;
    FeatureMatches all_matches;
    Rigid3d true_rel;
  };
  std::vector<PairData> pairs;

  const Rigid3d wrong_rel(
      Eigen::Quaterniond(
          Eigen::AngleAxisd(DegToRad(-45.0), Eigen::Vector3d::UnitX())),
      Eigen::Vector3d(0.0, 1.0, 0.0));

  for (image_t id1 = 1; id1 <= num_images; ++id1) {
    for (image_t id2 = id1 + 1; id2 <= num_images; ++id2) {
      PairData pd;
      pd.id1 = id1;
      pd.id2 = id2;
      pd.true_rel = cams_from_world[id2] * Inverse(cams_from_world[id1]);
      pd.true_rel.translation().normalize();

      AppendSyntheticCorrespondences(camera,
                                     camera,
                                     pd.true_rel,
                                     /*num_points=*/50,
                                     /*noise_px=*/0.3,
                                     rng,
                                     points_per_image[id1],
                                     points_per_image[id2],
                                     pd.static_matches);
      pd.all_matches = pd.static_matches;

      if (id1 == 1 && id2 == 3) {
        // Pair (1, 3) has dominant wrong motion in its initial matches.
        FeatureMatches wrong_matches;
        AppendSyntheticCorrespondences(camera,
                                       camera,
                                       wrong_rel,
                                       /*num_points=*/100,
                                       /*noise_px=*/0.3,
                                       rng,
                                       points_per_image[id1],
                                       points_per_image[id2],
                                       wrong_matches);
        pd.all_matches.insert(
            pd.all_matches.end(), wrong_matches.begin(), wrong_matches.end());
      }
      pairs.push_back(std::move(pd));
    }
  }

  // Rebuild cache with populated 2D points and correspondence graph.
  DatabaseCache full_cache;
  full_cache.AddCamera(camera);
  full_cache.AddRig(rig);
  for (int i = 1; i <= num_images; ++i) {
    Frame frame;
    frame.SetFrameId(i);
    frame.SetRigId(1);
    Image img;
    img.SetImageId(i);
    img.SetName("img_" + std::to_string(i) + ".png");
    img.SetCameraId(camera.camera_id);
    img.SetFrameId(i);
    img.SetPoints2D(points_per_image[i]);
    frame.AddDataId(img.DataId());
    full_cache.AddFrame(frame);
    full_cache.AddImage(img);
  }

  for (const auto& pd : pairs) {
    full_cache.AddMatches(pd.id1, pd.id2, pd.all_matches);
    // Pair (2, 4) is omitted from CorrespondenceGraph to simulate an
    // unverified candidate pair.
    if (pd.id1 == 2 && pd.id2 == 4) {
      continue;
    }
    TwoViewGeometry tvg;
    tvg.config = TwoViewGeometry::CALIBRATED;
    if (pd.id1 == 1 && pd.id2 == 3) {
      // Pair (1, 3) was initially verified with the wrong motion.
      tvg.cam2_from_cam1 = wrong_rel;
      tvg.inlier_matches = FeatureMatches(pd.all_matches.begin() + 50,
                                          pd.all_matches.end());
    } else {
      tvg.cam2_from_cam1 = pd.true_rel;
      tvg.inlier_matches = pd.static_matches;
    }
    full_cache.CorrespondenceGraph()->AddTwoViewGeometry(
        pd.id1, pd.id2, std::move(tvg));
  }
  full_cache.CorrespondenceGraph()->Finalize();

  Reconstruction reconstruction;
  reconstruction.Load(full_cache);
  for (int i = 1; i <= num_images; ++i) {
    reconstruction.Frame(i).SetRigFromWorld(cams_from_world[i]);
  }

  PoseGraph pose_graph;
  pose_graph.Load(*full_cache.CorrespondenceGraph());
  EstimatePoseGraphCovariances(full_cache, pose_graph);
  // Mark corrupted edge (1, 3) as invalid (as rejected by rotation filtering).
  pose_graph.SetInvalidEdge(ImagePairToPairId(1, 3));
  EXPECT_FALSE(pose_graph.IsValid(ImagePairToPairId(1, 3)));
  EXPECT_FALSE(pose_graph.HasEdge(2, 4));

  TwoViewPoseSalvageOptions salvage_options;
  salvage_options.random_seed = 99;
  salvage_options.num_threads = 1;
  salvage_options.min_num_inliers = 30;

  RotationEstimatorOptions ra_options;
  ra_options.random_seed = 99;
  ra_options.num_threads = 1;
  ra_options.backend = RotationAveragingBackend::CERES;
  ra_options.reweighting = RotationAveragingReweighting::COVARIANCE;

  const size_t num_salvaged =
      SalvageTwoViewPoses(salvage_options,
                          ra_options,
                          full_cache,
                          reconstruction,
                          pose_graph,
                          *full_cache.CorrespondenceGraph());
  EXPECT_EQ(num_salvaged, 2u);

  // Both (1, 3) and (2, 4) should now be valid in PoseGraph with accurate
  // relative poses and clean correspondences in CorrespondenceGraph.
  EXPECT_TRUE(pose_graph.IsValid(ImagePairToPairId(1, 3)));
  EXPECT_TRUE(pose_graph.IsValid(ImagePairToPairId(2, 4)));

  const Rigid3d expected_13 = cams_from_world[3] * Inverse(cams_from_world[1]);
  const Rigid3d salvaged_13 = pose_graph.GetEdge(1, 3).cam2_from_cam1;
  EXPECT_LT(RadToDeg(salvaged_13.rotation().angularDistance(
                expected_13.rotation())),
            0.5);

  const Rigid3d expected_24 = cams_from_world[4] * Inverse(cams_from_world[2]);
  const Rigid3d salvaged_24 = pose_graph.GetEdge(2, 4).cam2_from_cam1;
  EXPECT_LT(RadToDeg(salvaged_24.rotation().angularDistance(
                expected_24.rotation())),
            0.5);

  FeatureMatches matches_13;
  full_cache.CorrespondenceGraph()->ExtractMatchesBetweenImages(
      1, 3, matches_13);
  EXPECT_GE(matches_13.size(), 45u);
  EXPECT_LE(matches_13.size(), 55u);

  FeatureMatches matches_24;
  full_cache.CorrespondenceGraph()->ExtractMatchesBetweenImages(
      2, 4, matches_24);
  EXPECT_GE(matches_24.size(), 45u);
}

}  // namespace
}  // namespace colmap
