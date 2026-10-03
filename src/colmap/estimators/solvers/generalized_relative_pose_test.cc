// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/generalized_relative_pose.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random_eigen.h"
#include "colmap/optim/loransac.h"
#include "colmap/scene/camera.h"
#include "colmap/util/eigen_alignment.h"

#include <array>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>

namespace colmap {
namespace {

struct GeneralizedRelativePoseProblem {
  std::vector<GRNPObservation> points1;
  std::vector<GRNPObservation> points2;
  Rigid3d rig2_from_rig1;
};

GeneralizedRelativePoseProblem CreateGeneralizedRelativePoseProblem(
    int num_points,
    int num_cameras1,
    int num_cameras2,
    bool panoramic1,
    bool panoramic2) {
  GeneralizedRelativePoseProblem problem;

  const std::array<Rigid3d, 2> rigs_from_world = {
      Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>().normalized()),
      Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>().normalized())};

  problem.rig2_from_rig1 = rigs_from_world[1] * Inverse(rigs_from_world[0]);

  std::vector<Rigid3d> cams_from_rig1(num_cameras1);
  for (int i = 0; i < num_cameras1; ++i) {
    const Eigen::Quaterniond cam1_from_rig_rotation = RandomEigenQuaterniond();
    cams_from_rig1[i] = Rigid3d(
        cam1_from_rig_rotation,
        panoramic1 ? cam1_from_rig_rotation * Eigen::Vector3d(1, 2, 3)
                   : Eigen::Vector3d(RandomEigenVectord<3>().normalized()));
  }

  std::vector<Rigid3d> cams_from_rig2(num_cameras2);
  for (int i = 0; i < num_cameras2; ++i) {
    const Eigen::Quaterniond cam2_from_rig_rotation = RandomEigenQuaterniond();
    cams_from_rig2[i] = Rigid3d(
        cam2_from_rig_rotation,
        panoramic2 ? cam2_from_rig_rotation * Eigen::Vector3d(-3, -2, -1)
                   : Eigen::Vector3d(RandomEigenVectord<3>().normalized()));
  }

  std::vector<Eigen::Vector3d> points3D;
  points3D.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    points3D.emplace_back(RandomEigenVectord<3>());
  }

  problem.points1.reserve(num_points);
  problem.points2.reserve(num_points);
  // GR6P/GR8P::Residuals score in pixel units with the tangent Sampson error,
  // so each observation carries its ray's unprojection Jacobian. A spherical
  // camera maps every bearing to a pixel, keeping the synthetic points valid in
  // all directions (a pinhole would drop the back hemisphere).
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0.0, 1000, 500);
  for (int i = 0; i < num_points; ++i) {
    const size_t cam_idx1 = i % num_cameras1;
    const size_t cam_idx2 = i % num_cameras2;
    const Eigen::Vector3d point3D_in_cam1 =
        cams_from_rig1[cam_idx1] * (rigs_from_world[0] * points3D[i]);
    const Eigen::Vector3d point3D_in_cam2 =
        cams_from_rig2[cam_idx2] * (rigs_from_world[1] * points3D[i]);
    if (point3D_in_cam1.norm() < 1e-8 || point3D_in_cam2.norm() < 1e-8) {
      continue;
    }

    const CamRayWithJac ray1_with_jac =
        camera.CamRayFromImgWithJac(camera.ImgFromCam(point3D_in_cam1).value())
            .value();
    const CamRayWithJac ray2_with_jac =
        camera.CamRayFromImgWithJac(camera.ImgFromCam(point3D_in_cam2).value())
            .value();

    auto& point1 = problem.points1.emplace_back();
    point1.cam_from_rig = cams_from_rig1[cam_idx1];
    point1.ray_with_jac_in_cam = ray1_with_jac;

    auto& point2 = problem.points2.emplace_back();
    point2.cam_from_rig = cams_from_rig2[cam_idx2];
    point2.ray_with_jac_in_cam = ray2_with_jac;
  }

  return problem;
}

class ParameterizedGRNPEstimatorTests
    : public ::testing::TestWithParam<std::tuple</*num_cams1=*/int,
                                                 /*num_cams2=*/int,
                                                 /*panoramic1=*/bool,
                                                 /*panoramic2=*/bool>> {};

TEST_P(ParameterizedGRNPEstimatorTests, GR6P) {
  SetPRNGSeed(1);

  // Note that we can estimate the minimal problem from only 6 points but we
  // use an additional points to choose the correct solution.
  constexpr int kNumPoints = GR6PEstimator::kMinNumSamples + 1;
  constexpr int kNumTrials = 10;
  const auto [kNumCams1, kNumCams2, kPanoramic1, kPanoramic2] = GetParam();

  for (int i = 0; i < kNumTrials; ++i) {
    const auto problem = CreateGeneralizedRelativePoseProblem(
        kNumPoints, kNumCams1, kNumCams2, kPanoramic1, kPanoramic2);

    RANSACOptions options;
    options.max_error = 1.0;  // pixels
    RANSAC<GR6PEstimator> ransac(options);
    const auto report = ransac.Estimate(problem.points1, problem.points2);

    EXPECT_TRUE(report.success);
    EXPECT_THAT(report.model,
                Rigid3dNear(problem.rig2_from_rig1,
                            /*rtol=*/1e-4,
                            /*ttol=*/1e-4));

    std::vector<double> residuals;
    GR6PEstimator::Residuals(
        problem.points1, problem.points2, report.model, &residuals);
    // Residuals are squared pixels. The RANSAC inlier bound is max_error^2.
    for (size_t i = 0; i < residuals.size(); ++i) {
      EXPECT_LE(residuals[i], options.max_error * options.max_error);
    }
  }
}

TEST_P(ParameterizedGRNPEstimatorTests, GR8P) {
  SetPRNGSeed(1);

  // Note that we can estimate the minimal problem from only 8 points but we
  // use the additional points to choose the correct solution.
  // The GR8P estimator is numerically much more sensitive than the GR6P
  // estimator, so we only expect one successful estimation in all trials.
  constexpr int kNumPoints = 2 * GR8PEstimator::kMinNumSamples;
  constexpr int kNumTrials = 10;
  const auto [kNumCams1, kNumCams2, kPanoramic1, kPanoramic2] = GetParam();

  bool success = false;
  for (int i = 0; i < kNumTrials; ++i) {
    const auto problem = CreateGeneralizedRelativePoseProblem(
        kNumPoints, kNumCams1, kNumCams2, kPanoramic1, kPanoramic2);

    RANSACOptions options;
    options.max_num_trials = 1000;
    options.max_error = 5.0;  // pixels
    RANSAC<GR8PEstimator> ransac(options);
    const auto report = ransac.Estimate(problem.points1, problem.points2);

    if (!report.success) {
      continue;
    }

    if (!testing::Value(report.model,
                        Rigid3dNear(problem.rig2_from_rig1,
                                    /*rtol=*/1e-2,
                                    /*ttol=*/1e-2))) {
      continue;
    }

    std::vector<double> residuals;
    GR8PEstimator::Residuals(
        problem.points1, problem.points2, report.model, &residuals);
    for (size_t i = 0; i < residuals.size(); ++i) {
      if (residuals[i] > options.max_error) {
        continue;
      }
    }

    success = true;
    break;
  }

  EXPECT_TRUE(success);
}

// Note that only one of the cameras must be panoramic, otherwise the
// generalized relative pose problem is ill-posed, as we cannot estimate the
// scale between the rigs.
INSTANTIATE_TEST_SUITE_P(GRNPEstimatorTests,
                         ParameterizedGRNPEstimatorTests,
                         ::testing::Values(std::make_tuple(1, 2, false, false),
                                           std::make_tuple(2, 1, false, false),
                                           std::make_tuple(2, 2, false, false),
                                           std::make_tuple(3, 3, false, false),
                                           std::make_tuple(4, 4, false, false),
                                           std::make_tuple(4, 4, false, true),
                                           std::make_tuple(4, 4, true, false)));

// Partition observation indices into those sharing the first observation's
// camera pair and those from any other pair.
std::pair<std::vector<size_t>, std::vector<size_t>> PartitionByCameraPair(
    const GeneralizedRelativePoseProblem& problem) {
  std::vector<size_t> ref_pair_idxs, other_pair_idxs;
  for (size_t i = 0; i < problem.points1.size(); ++i) {
    if (problem.points2[i].cam_from_rig == problem.points2[0].cam_from_rig) {
      ref_pair_idxs.push_back(i);
    } else {
      other_pair_idxs.push_back(i);
    }
  }
  return {ref_pair_idxs, other_pair_idxs};
}

bool AnyModelNear(const std::vector<Rigid3d>& models, const Rigid3d& expected) {
  for (const Rigid3d& model : models) {
    if (testing::Value(model,
                       Rigid3dNear(expected, /*rtol=*/1e-3, /*ttol=*/1e-3))) {
      return true;
    }
  }
  return false;
}

TEST(GR5P1PEstimator, Nominal) {
  // 2x2 cameras alternate between pairs (0,0) and (1,1).
  for (int seed = 1; seed <= 5; ++seed) {
    SetPRNGSeed(seed);
    const auto problem =
        CreateGeneralizedRelativePoseProblem(20, 2, 2, false, false);
    const auto [ref_pair_idxs, other_pair_idxs] =
        PartitionByCameraPair(problem);
    ASSERT_GE(ref_pair_idxs.size(), 7);
    ASSERT_GE(other_pair_idxs.size(), 1);
    for (int offset = 0; offset < 3; ++offset) {
      // Outlier first exercises the internal reorder to 5+1 order.
      std::vector<GRNPObservation> points1, points2;
      for (const size_t i : {other_pair_idxs[0],
                             ref_pair_idxs[offset + 0],
                             ref_pair_idxs[offset + 1],
                             ref_pair_idxs[offset + 2],
                             ref_pair_idxs[offset + 3],
                             ref_pair_idxs[offset + 4]}) {
        points1.push_back(problem.points1[i]);
        points2.push_back(problem.points2[i]);
      }

      std::vector<Rigid3d> models;
      GR5P1PEstimator::Estimate(points1, points2, &models);
      EXPECT_TRUE(AnyModelNear(models, problem.rig2_from_rig1));

      // The GR6P fast path must agree on 5+1-structured samples.
      models.clear();
      GR6PEstimator::Estimate(points1, points2, &models);
      EXPECT_TRUE(AnyModelNear(models, problem.rig2_from_rig1));
    }
  }
}

TEST(GR5P1PEstimator, DistantSixthCamera) {
  // Five correspondences from camera 0 of rig 1 plus a sixth from camera 1,
  // mounted far to the side, all matched to the single camera of rig 2.
  // Viewed from the majority camera pair, the rays of the sixth
  // correspondence diverge, so a cheirality check treating it as a
  // majority-pair correspondence would reject the true pose (see
  // PoseLib/PoseLib#214).
  const Rigid3d rig2_from_rig1(
      Eigen::Quaterniond(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY())),
      Eigen::Vector3d(0.1, 0, -1));
  const std::array<Rigid3d, 2> cams_from_rig1 = {
      Rigid3d(),
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(-10, -2, 0))};
  const Rigid3d cam_from_rig2;
  const std::array<Eigen::Vector3d, 6> points3D_in_rig1 = {
      Eigen::Vector3d(-1, 0.5, 5),
      Eigen::Vector3d(1, -0.5, 6),
      Eigen::Vector3d(0.5, 1, 7),
      Eigen::Vector3d(-0.5, -1, 4),
      Eigen::Vector3d(1.5, 0.3, 8),
      Eigen::Vector3d(6, -1, 3)};

  std::vector<GRNPObservation> points1, points2;
  for (int i = 0; i < 6; ++i) {
    const Rigid3d& cam_from_rig1 = cams_from_rig1[i < 5 ? 0 : 1];
    const Eigen::Vector3d point3D_in_rig2 =
        rig2_from_rig1 * points3D_in_rig1[i];
    points1.push_back({cam_from_rig1,
                       {(cam_from_rig1 * points3D_in_rig1[i]).normalized(),
                        Eigen::Matrix3x2d::Zero()}});
    points2.push_back({cam_from_rig2,
                       {(cam_from_rig2 * point3D_in_rig2).normalized(),
                        Eigen::Matrix3x2d::Zero()}});
  }

  std::vector<Rigid3d> models;
  GR5P1PEstimator::Estimate(points1, points2, &models);
  EXPECT_TRUE(AnyModelNear(models, rig2_from_rig1));

  models.clear();
  GR6PEstimator::Estimate(points1, points2, &models);
  EXPECT_TRUE(AnyModelNear(models, rig2_from_rig1));
}

TEST(GR5P1PEstimator, RejectsNon5P1PSamples) {
  SetPRNGSeed(1);
  const auto problem =
      CreateGeneralizedRelativePoseProblem(20, 1, 2, false, false);
  const auto [ref_pair_idxs, other_pair_idxs] = PartitionByCameraPair(problem);
  ASSERT_GE(ref_pair_idxs.size(), 6);
  ASSERT_GE(other_pair_idxs.size(), 3);

  std::vector<Rigid3d> models;
  // 3+3 mixed sample: no five share a pair.
  std::vector<GRNPObservation> points1, points2;
  for (const size_t i : {ref_pair_idxs[0],
                         ref_pair_idxs[1],
                         ref_pair_idxs[2],
                         other_pair_idxs[0],
                         other_pair_idxs[1],
                         other_pair_idxs[2]}) {
    points1.push_back(problem.points1[i]);
    points2.push_back(problem.points2[i]);
  }
  GR5P1PEstimator::Estimate(points1, points2, &models);
  EXPECT_TRUE(models.empty());

  // All six from one pair: scale unobservable, 5p1pt must decline.
  points1.clear();
  points2.clear();
  for (int k = 0; k < 6; ++k) {
    points1.push_back(problem.points1[ref_pair_idxs[k]]);
    points2.push_back(problem.points2[ref_pair_idxs[k]]);
  }
  GR5P1PEstimator::Estimate(points1, points2, &models);
  EXPECT_TRUE(models.empty());

  // Different camera rotations at the same optical centers still provide no
  // baseline, so scale remains unobservable.
  const Eigen::Vector3d origin1 = points1.back().cam_from_rig.TgtOriginInSrc();
  const Eigen::Vector3d origin2 = points2.back().cam_from_rig.TgtOriginInSrc();
  const Eigen::Quaterniond rotation1 =
      Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()) *
      points1.back().cam_from_rig.rotation();
  const Eigen::Quaterniond rotation2 =
      Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY()) *
      points2.back().cam_from_rig.rotation();
  points1.back().cam_from_rig = Rigid3d(rotation1, rotation1 * -origin1);
  points2.back().cam_from_rig = Rigid3d(rotation2, rotation2 * -origin2);
  GR5P1PEstimator::Estimate(points1, points2, &models);
  EXPECT_TRUE(models.empty());

  // The GR6P fallback still solves the well-posed mixed sample.
  points1.clear();
  points2.clear();
  for (const size_t i : {ref_pair_idxs[0],
                         ref_pair_idxs[1],
                         ref_pair_idxs[2],
                         other_pair_idxs[0],
                         other_pair_idxs[1],
                         other_pair_idxs[2]}) {
    points1.push_back(problem.points1[i]);
    points2.push_back(problem.points2[i]);
  }
  GR6PEstimator::Estimate(points1, points2, &models);
  EXPECT_FALSE(models.empty());
  EXPECT_TRUE(AnyModelNear(models, problem.rig2_from_rig1));
}

}  // namespace
}  // namespace colmap
