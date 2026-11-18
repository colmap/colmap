// SPDX-License-Identifier: BSD-3-Clause

// cusift matches near COLMAP's verification and registration thresholds. There,
// one missing correct match decides whether a pair verifies or an image
// registers. Each scene has a known number of correct correspondences, plus
// wrong matches that share a descriptor but not the geometry, so the raw match
// count stays above the threshold and only the inliers decide. cusift must
// return every one of them, and the outcome must flip exactly at the threshold.

#include "colmap/estimators/pose.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/cusift_matcher.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/sift_test_utils.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/util/cuda.h"
#include "colmap/util/logging.h"

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

constexpr int kNumWrongMatches = 10;
// Wrong matches lie at least this far, in pixels, from where a correct match
// would, so RANSAC never counts one as an inlier.
constexpr double kMinWrongMatchError = 100.0;

Camera CreateCalibratedCamera() {
  Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 500.0, 640, 480);
  camera.source = CameraSource::USER;
  return camera;
}

// The second view is 1 m from the first, which sits at the origin. Both look
// at points 3 to 9 m ahead: a deep scene, so the fundamental matrix of the
// correct matches is well determined.
Rigid3d Cam2FromWorld() {
  return Rigid3d(
      Eigen::Quaterniond(Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitY())),
      Eigen::Vector3d(-1, 0, 0));
}

std::vector<Eigen::Vector3d> CreatePoints(const int num_points) {
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < num_points; ++i) {
    points.emplace_back(RandomUniformReal(-2.0, 2.0),
                        RandomUniformReal(-1.5, 1.5),
                        RandomUniformReal(3.0, 9.0));
  }
  return points;
}

Eigen::Vector2d RandomImagePoint(const Camera& camera) {
  return Eigen::Vector2d(RandomUniformReal<double>(0, camera.width),
                         RandomUniformReal<double>(0, camera.height));
}

// A random image point whose `error` is at least kMinWrongMatchError.
Eigen::Vector2d RandomImagePointAwayFrom(
    const Camera& camera,
    const std::function<double(const Eigen::Vector2d&)>& error) {
  Eigen::Vector2d point;
  do {
    point = RandomImagePoint(camera);
  } while (error(point) < kMinWrongMatchError);
  return point;
}

// Distance in pixels of `point2` from the epipolar line of `point1`.
double EpipolarDistance(const Eigen::Matrix3d& F,
                        const Eigen::Vector2d& point1,
                        const Eigen::Vector2d& point2) {
  const Eigen::Vector3d line = F * point1.homogeneous();
  return std::abs(line.dot(point2.homogeneous())) / line.head<2>().norm();
}

FeatureKeypoints ToKeypoints(const std::vector<Eigen::Vector2d>& points) {
  FeatureKeypoints keypoints;
  for (const Eigen::Vector2d& point : points) {
    keypoints.emplace_back(point.x(), point.y());
  }
  return keypoints;
}

// Matches the two keypoint sets with cusift. Both sets share their descriptors
// row by row, so every row must match itself.
FeatureMatches MatchWithCusift(const Camera& camera,
                               const std::vector<Eigen::Vector2d>& points1,
                               const std::vector<Eigen::Vector2d>& points2) {
  const auto descriptors = std::make_shared<FeatureDescriptors>(
      CreateRandomFeatureDescriptors(points1.size()));
  const FeatureMatcher::Image image1{
      1,
      &camera,
      std::make_shared<FeatureKeypoints>(ToKeypoints(points1)),
      descriptors};
  const FeatureMatcher::Image image2{
      2,
      &camera,
      std::make_shared<FeatureKeypoints>(ToKeypoints(points2)),
      descriptors};
  FeatureMatchingOptions options(FeatureMatcherType::CUSIFT_GEMM);
  options.use_gpu = true;
  FeatureMatches matches;
  THROW_CHECK_NOTNULL(CreateCusiftFeatureMatcher(options))
      ->Match(image1, image2, &matches);
  return matches;
}

// Image points of both views: the `num_correct` correct correspondences first,
// then kNumWrongMatches wrong ones, each far from its epipolar line.
struct TwoViewScene {
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
};

TwoViewScene CreateTwoViewScene(const Camera& camera, const int num_correct) {
  const Rigid3d cam2_from_world = Cam2FromWorld();
  const Eigen::Matrix3d K = camera.CalibrationMatrix();
  const Eigen::Matrix3d F = FundamentalFromEssentialMatrix(
      K, EssentialMatrixFromPose(cam2_from_world), K);
  TwoViewScene scene;
  for (const Eigen::Vector3d& point : CreatePoints(num_correct)) {
    scene.points1.push_back(*camera.ImgFromCam(point));
    scene.points2.push_back(*camera.ImgFromCam(cam2_from_world * point));
  }
  for (int i = 0; i < kNumWrongMatches; ++i) {
    const Eigen::Vector2d point1 = RandomImagePoint(camera);
    scene.points1.push_back(point1);
    scene.points2.push_back(
        RandomImagePointAwayFrom(camera, [&](const Eigen::Vector2d& point2) {
          return EpipolarDistance(F, point1, point2);
        }));
  }
  return scene;
}

// 2D-3D correspondences of the second view: the `num_correct` correct ones
// first, then kNumWrongMatches wrong ones, each far from its projection.
struct PoseScene {
  std::vector<Eigen::Vector3d> points3D;
  std::vector<Eigen::Vector2d> points2D;
};

PoseScene CreatePoseScene(const Camera& camera, const int num_correct) {
  const Rigid3d cam_from_world = Cam2FromWorld();
  PoseScene scene;
  scene.points3D = CreatePoints(num_correct);
  for (const Eigen::Vector3d& point : scene.points3D) {
    scene.points2D.push_back(*camera.ImgFromCam(cam_from_world * point));
  }
  for (const Eigen::Vector3d& point : CreatePoints(kNumWrongMatches)) {
    const Eigen::Vector2d projection =
        *camera.ImgFromCam(cam_from_world * point);
    scene.points3D.push_back(point);
    scene.points2D.push_back(
        RandomImagePointAwayFrom(camera, [&](const Eigen::Vector2d& point2D) {
          return (point2D - projection).norm();
        }));
  }
  return scene;
}

// Inliers of the pose estimated as IncrementalMapper::RegisterNextImage does.
size_t CountPoseInliers(const Camera& camera,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D) {
  const IncrementalMapper::Options mapper_options;
  AbsolutePoseEstimationOptions options;
  options.ransac_options.max_error = mapper_options.abs_pose_max_error;
  options.ransac_options.min_inlier_ratio =
      mapper_options.abs_pose_min_inlier_ratio;
  options.ransac_options.random_seed = 1;
  Rigid3d cam_from_world;
  Camera estimated_camera = camera;
  size_t num_inliers = 0;
  std::vector<char> inlier_mask;
  THROW_CHECK(EstimateAbsolutePose(options,
                                   points2D,
                                   points3D,
                                   &cam_from_world,
                                   &estimated_camera,
                                   &num_inliers,
                                   &inlier_mask));
  return num_inliers;
}

class CusiftNearThresholdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
    SetPRNGSeed(1);
    // The correct matches are exact, so a tight error loses none of them. It
    // keeps the fundamental matrix, which has more freedom than the essential
    // matrix, from bending to take in a wrong match: that would also count
    // more F than E inliers and label a calibrated pair UNCALIBRATED.
    options_.ransac_options.max_error = 1.0;
    options_.ransac_options.random_seed = 1;
  }

  struct Verification {
    FeatureMatches matches;
    TwoViewGeometry geometry;
  };

  Verification MatchAndVerify(const int num_correct) const {
    const TwoViewScene scene = CreateTwoViewScene(camera_, num_correct);
    Verification verification;
    verification.matches =
        MatchWithCusift(camera_, scene.points1, scene.points2);
    verification.geometry = EstimateTwoViewGeometry(camera_,
                                                    scene.points1,
                                                    camera_,
                                                    scene.points2,
                                                    verification.matches,
                                                    options_);
    return verification;
  }

  const Camera camera_ = CreateCalibratedCamera();
  TwoViewGeometryOptions options_;
};

using CusiftTwoViewVerificationTest = CusiftNearThresholdTest;

TEST_F(CusiftTwoViewVerificationTest, VerifiesAtMinNumInliers) {
  const int num_correct = options_.min_num_inliers;

  const Verification verification = MatchAndVerify(num_correct);

  ASSERT_EQ(verification.matches.size(),
            static_cast<size_t>(num_correct + kNumWrongMatches));
  EXPECT_EQ(verification.geometry.config, TwoViewGeometry::CALIBRATED);
  EXPECT_EQ(verification.geometry.inlier_matches.size(),
            static_cast<size_t>(num_correct));
}

TEST_F(CusiftTwoViewVerificationTest, RejectsOneBelowMinNumInliers) {
  const int num_correct = options_.min_num_inliers - 1;

  const Verification verification = MatchAndVerify(num_correct);

  ASSERT_EQ(verification.matches.size(),
            static_cast<size_t>(num_correct + kNumWrongMatches));
  // More raw matches than min_num_inliers: RANSAC's inlier count rejects it.
  EXPECT_GT(verification.matches.size(),
            static_cast<size_t>(options_.min_num_inliers));
  EXPECT_EQ(verification.geometry.config, TwoViewGeometry::DEGENERATE);
}

// GetParam() is the offset of the number of correct 2D-3D correspondences
// from abs_pose_min_num_inliers. RegisterNextImage rejects an image whose pose
// has fewer inliers than that, so finding exactly the correct ones makes it
// fail one below the threshold and pass at it. With the wrong matches, the
// 2D-3D correspondences outnumber the threshold, as its earlier count check
// needs.
class CusiftRegistrationTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
    SetPRNGSeed(1);
  }

  const Camera camera_ = CreateCalibratedCamera();
};

TEST_P(CusiftRegistrationTest, FindsExactlyTheCorrectPoseInliers) {
  const int num_correct =
      IncrementalMapper::Options().abs_pose_min_num_inliers + GetParam();
  const PoseScene scene = CreatePoseScene(camera_, num_correct);
  // The reference image only carries the descriptors of the 3D points.
  const std::vector<Eigen::Vector2d> reference_points(scene.points3D.size(),
                                                      Eigen::Vector2d::Zero());

  const FeatureMatches matches =
      MatchWithCusift(camera_, reference_points, scene.points2D);
  std::vector<Eigen::Vector2d> matched_points2D;
  std::vector<Eigen::Vector3d> matched_points3D;
  for (const FeatureMatch& match : matches) {
    matched_points3D.push_back(scene.points3D[match.point2D_idx1]);
    matched_points2D.push_back(scene.points2D[match.point2D_idx2]);
  }

  ASSERT_EQ(matches.size(),
            static_cast<size_t>(num_correct + kNumWrongMatches));
  EXPECT_EQ(CountPoseInliers(camera_, matched_points2D, matched_points3D),
            static_cast<size_t>(num_correct));
}

INSTANTIATE_TEST_SUITE_P(BelowAndAtThreshold,
                         CusiftRegistrationTest,
                         ::testing::Values(-1, 0));

}  // namespace
}  // namespace colmap
