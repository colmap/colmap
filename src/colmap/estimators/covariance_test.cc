// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/covariance.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <ceres/normal_prior.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void ExpectNearEigenMatrixXd(const Eigen::MatrixXd& mat1,
                             const Eigen::MatrixXd& mat2,
                             double tol) {
  ASSERT_EQ(mat1.rows(), mat2.rows());
  ASSERT_EQ(mat1.cols(), mat2.cols());
  for (int i = 0; i < mat1.rows(); ++i) {
    for (int j = 0; j < mat1.cols(); ++j) {
      ASSERT_NEAR(mat1(i, j), mat2(i, j), tol);
    }
  }
}

TEST(BACovariance, RelativePoseTangentCoordinates) {
  const Rigid3d cams_from_world[2] = {
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random()),
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random())};
  const BACovariance ba_cov(
      {}, {{1, {0, 6}}, {2, {6, 6}}}, {}, Eigen::MatrixXd::Identity(12, 12));
  const std::optional<Eigen::MatrixXd> cov =
      ba_cov.GetCam2CovFromCam1(1, cams_from_world[0], 2, cams_from_world[1]);
  ASSERT_TRUE(cov.has_value());

  // Numerically differentiate the relative pose in left SE(3) coordinates
  // w.r.t. the Ceres tangent coordinates of both poses.
  const auto manifold = CreateProductManifold(CreateEigenQuaternionManifold(),
                                              CreateEuclideanManifold<3>());
  const Rigid3d cam2_from_cam1 =
      cams_from_world[1] * Inverse(cams_from_world[0]);
  constexpr double kEps = 1e-7;
  Eigen::Matrix<double, 6, 12> J;
  for (int i = 0; i < 12; ++i) {
    Eigen::Vector6d delta = Eigen::Vector6d::Zero();
    delta(i % 6) = kEps;
    Rigid3d perturbed[2] = {cams_from_world[0], cams_from_world[1]};
    manifold->Plus(cams_from_world[i / 6].params.data(),
                   delta.data(),
                   perturbed[i / 6].params.data());
    const Rigid3d diff =
        perturbed[1] * Inverse(perturbed[0]) * Inverse(cam2_from_cam1);
    const Eigen::AngleAxisd diff_rotation(diff.rotation());
    J.col(i) << diff_rotation.angle() * diff_rotation.axis(),
        diff.translation();
  }
  J /= kEps;
  ExpectNearEigenMatrixXd(*cov, J * J.transpose(), /*tol=*/1e-5);
}

TEST(BACovariance, PointTangentCoordinates) {
  Reconstruction reconstruction;
  ceres::Problem problem;
  for (const double weight : {1.0, 2.0}) {
    const point3D_t point3D_id =
        reconstruction.AddPoint3D(Eigen::Vector3d(weight, 0, 0), Track());
    double* xyz = reconstruction.Point3D(point3D_id).xyz.data();
    problem.AddResidualBlock(
        new ceres::NormalPrior(weight * Eigen::Matrix3d::Identity(),
                               Eigen::Vector3d::Zero()),
        nullptr,
        xyz);
    SetManifold(&problem, xyz, CreateSubsetManifold(3, {2}));
  }

  BACovarianceOptions options;
  options.params = BACovarianceOptions::Params::POINTS;
  options.damping = 0;
  const std::optional<BACovariance> ba_cov =
      EstimateBACovarianceFromProblem(options, reconstruction, problem);
  ASSERT_TRUE(ba_cov.has_value());
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    const std::optional<Eigen::MatrixXd> cov = ba_cov->GetPointCov(point3D_id);
    ASSERT_TRUE(cov.has_value());
    const double weight = point3D.xyz.x();
    ExpectNearEigenMatrixXd(
        *cov, Eigen::Matrix2d::Identity() / (weight * weight), /*tol=*/1e-12);
  }
}

TEST(BACovariance, RejectsSingularPointBlock) {
  Reconstruction reconstruction;
  const point3D_t point3D_id =
      reconstruction.AddPoint3D(Eigen::Vector3d::Zero(), Track());
  ceres::Problem problem;
  problem.AddResidualBlock(
      new ceres::NormalPrior(Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero()),
      nullptr,
      reconstruction.Point3D(point3D_id).xyz.data());

  BACovarianceOptions options;
  options.params = BACovarianceOptions::Params::POINTS;
  options.damping = 0;
  EXPECT_FALSE(EstimateBACovarianceFromProblem(options, reconstruction, problem)
                   .has_value());
}

TEST(BACovariance, PointsWithNonRefImages) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const auto& [image_id, _] : reconstruction.Images()) {
    config.AddImage(image_id);
  }
  std::unique_ptr<CeresBundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(
          BundleAdjustmentOptions(), config, reconstruction);

  BACovarianceOptions options;
  options.params = BACovarianceOptions::Params::POINTS;
  const std::optional<BACovariance> ba_cov =
      EstimateBACovariance(options, reconstruction, *bundle_adjuster);
  ASSERT_TRUE(ba_cov.has_value());
  EXPECT_TRUE(ba_cov->GetPointCov(reconstruction.Points3D().begin()->first)
                  .has_value());
}

TEST(BACovariance, OtherParamsNotRequested) {
  Eigen::Vector2d pose = Eigen::Vector2d::Zero();
  Eigen::Vector2d other = Eigen::Vector2d::Zero();
  ceres::Problem problem;
  for (double* params : {pose.data(), other.data()}) {
    problem.AddResidualBlock(new ceres::NormalPrior(Eigen::Matrix2d::Identity(),
                                                    Eigen::Vector2d::Zero()),
                             nullptr,
                             params);
  }

  BACovarianceOptions options;
  options.params = BACovarianceOptions::Params::POSES;
  options.experimental_custom_poses = {{1, pose.data()}};
  const std::optional<BACovariance> ba_cov =
      EstimateBACovarianceFromProblem(options, Reconstruction(), problem);
  ASSERT_TRUE(ba_cov.has_value());
  EXPECT_TRUE(ba_cov->GetCamCovFromWorld(1).has_value());
  EXPECT_FALSE(ba_cov->GetOtherParamsCov(other.data()).has_value());
}

struct BACovarianceTestOptions {
  bool fixed_points = false;
  bool fixed_cam_poses = false;
  bool fixed_cam_intrinsics = false;
};

class ParameterizedBACovarianceTests
    : public ::testing::TestWithParam<
          std::pair<BACovarianceOptions, BACovarianceTestOptions>> {};

TEST_P(ParameterizedBACovarianceTests, CompareWithCeres) {
  const auto [options, test_options] = GetParam();

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

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 200;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.01;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    config.AddImage(image_id);
    if (test_options.fixed_cam_poses) {
      config.SetConstantRigFromWorldPose(image.FrameId());
    }
    if (test_options.fixed_cam_intrinsics) {
      config.SetConstantCamIntrinsics(image.CameraId());
    }
  }

  // Fix the Gauge by always setting at least 3 points as constant.
  CHECK_GT(reconstruction.NumPoints3D(), 3);
  int num_constant_points = 0;
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    if (++num_constant_points <= 3 || test_options.fixed_points) {
      config.AddConstantPoint(point3D_id);
    }
  }

  std::unique_ptr<BundleAdjuster> bundle_adjuster = CreateDefaultBundleAdjuster(
      BundleAdjustmentOptions(), config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  // Cast to CeresBundleAdjuster to access Problem()
  auto* ceres_ba = dynamic_cast<CeresBundleAdjuster*>(bundle_adjuster.get());
  ASSERT_NE(ceres_ba, nullptr);
  std::shared_ptr<ceres::Problem> problem = ceres_ba->Problem();

  const std::optional<BACovariance> ba_cov =
      EstimateBACovariance(options, reconstruction, *ceres_ba);
  ASSERT_TRUE(ba_cov.has_value());

  const std::vector<internal::PointParam> points =
      internal::GetPointParams(reconstruction, *problem);
  if (test_options.fixed_points) {
    ASSERT_TRUE(points.empty());
  } else {
    ASSERT_EQ(points.size(), synthetic_dataset_options.num_points3D - 3);
  }

  const std::vector<internal::PoseParam> poses =
      internal::GetPoseParams(reconstruction, *problem);
  if (test_options.fixed_cam_poses) {
    ASSERT_TRUE(poses.empty());
  } else {
    ASSERT_EQ(poses.size(), synthetic_dataset_options.num_frames_per_rig);
  }

  const std::vector<const double*> others =
      GetOtherParams(*problem, poses, points);
  if (test_options.fixed_cam_intrinsics) {
    ASSERT_TRUE(others.empty());
  } else {
    ASSERT_EQ(others.size(), synthetic_dataset_options.num_cameras_per_rig);
  }

  if (!test_options.fixed_cam_poses && estimate_pose_covs) {
    LOG(INFO) << "Comparing pose covariances";

    std::vector<std::pair<const double*, const double*>> cov_param_pairs;
    for (const auto& pose1 : poses) {
      for (const auto& pose2 : poses) {
        cov_param_pairs.emplace_back(pose1.cam_from_world,
                                     pose2.cam_from_world);
      }
    }

    ceres::Covariance::Options ceres_cov_options;
    ceres::Covariance ceres_cov_computer(ceres_cov_options);
    ASSERT_TRUE(ceres_cov_computer.Compute(cov_param_pairs, problem.get()));

    for (const auto& pose1 : poses) {
      for (const auto& pose2 : poses) {
        std::vector<const double*> param_blocks;

        const int tangent_size1 =
            ParameterBlockTangentSize(*problem, pose1.cam_from_world);
        param_blocks.push_back(pose1.cam_from_world);

        int tangent_size2 = 0;
        if (pose1.image_id != pose2.image_id) {
          tangent_size2 +=
              ParameterBlockTangentSize(*problem, pose2.cam_from_world);
          param_blocks.push_back(pose2.cam_from_world);
        }

        Eigen::MatrixXd ceres_cov(tangent_size1 + tangent_size2,
                                  tangent_size1 + tangent_size2);
        ceres_cov_computer.GetCovarianceMatrixInTangentSpace(param_blocks,
                                                             ceres_cov.data());

        if (pose1.image_id == pose2.image_id) {
          const std::optional<Eigen::MatrixXd> cov =
              ba_cov->GetCamCovFromWorld(pose1.image_id);
          ASSERT_TRUE(cov.has_value());
          ExpectNearEigenMatrixXd(ceres_cov, *cov, /*tol=*/1e-8);
        } else {
          const std::optional<Eigen::MatrixXd> cov =
              ba_cov->GetCamCrossCovFromWorld(pose1.image_id, pose2.image_id);
          ASSERT_TRUE(cov.has_value());
          ExpectNearEigenMatrixXd(
              ceres_cov.block(0, tangent_size1, tangent_size1, tangent_size2),
              *cov,
              /*tol=*/1e-8);
        }
      }
    }

    ASSERT_FALSE(ba_cov->GetCamCovFromWorld(kInvalidImageId).has_value());
    ASSERT_FALSE(
        ba_cov->GetCamCrossCovFromWorld(kInvalidImageId, poses[0].image_id)
            .has_value());
    ASSERT_FALSE(
        ba_cov->GetCamCrossCovFromWorld(poses[0].image_id, kInvalidImageId)
            .has_value());
  }

  if (!test_options.fixed_cam_intrinsics && estimate_other_covs) {
    LOG(INFO) << "Comparing other covariances";

    std::vector<std::pair<const double*, const double*>> cov_param_pairs;
    for (const double* other : others) {
      if (other != nullptr) {
        cov_param_pairs.emplace_back(other, other);
      }
    }

    ceres::Covariance::Options ceres_cov_options;
    ceres::Covariance ceres_cov_computer(ceres_cov_options);
    ASSERT_TRUE(ceres_cov_computer.Compute(cov_param_pairs, problem.get()));

    for (const double* other : others) {
      const int tangent_size = ParameterBlockTangentSize(*problem, other);

      Eigen::MatrixXd ceres_cov(tangent_size, tangent_size);
      ceres_cov_computer.GetCovarianceMatrixInTangentSpace({other},
                                                           ceres_cov.data());

      const std::optional<Eigen::MatrixXd> cov =
          ba_cov->GetOtherParamsCov(other);
      ASSERT_TRUE(cov.has_value());
      ExpectNearEigenMatrixXd(ceres_cov, *cov, /*tol=*/1e-8);
    }

    ASSERT_FALSE(ba_cov->GetOtherParamsCov(nullptr).has_value());
  }

  if (!test_options.fixed_points && estimate_point_covs) {
    LOG(INFO) << "Comparing point covariances";

    // Set all pose/other parameters as constant.
    for (const auto& pose : poses) {
      problem->SetParameterBlockConstant(
          const_cast<double*>(pose.cam_from_world));
    }
    for (const double* other : others) {
      if (other != nullptr) {
        problem->SetParameterBlockConstant(const_cast<double*>(other));
      }
    }

    std::vector<std::pair<const double*, const double*>> cov_param_pairs;
    for (const auto& point : points) {
      if (point.xyz != nullptr) {
        cov_param_pairs.emplace_back(point.xyz, point.xyz);
      }
    }

    ceres::Covariance::Options ceres_cov_options;
    ceres::Covariance ceres_cov_computer(ceres_cov_options);
    ASSERT_TRUE(ceres_cov_computer.Compute(cov_param_pairs, problem.get()));

    for (const auto& point : points) {
      const int tangent_size = ParameterBlockTangentSize(*problem, point.xyz);

      Eigen::MatrixXd ceres_cov(tangent_size, tangent_size);
      ceres_cov_computer.GetCovarianceMatrixInTangentSpace({point.xyz},
                                                           ceres_cov.data());

      const std::optional<Eigen::MatrixXd> cov =
          ba_cov->GetPointCov(point.point3D_id);
      ASSERT_TRUE(cov.has_value());
      ExpectNearEigenMatrixXd(ceres_cov, *cov, /*tol=*/1e-8);
    }

    ASSERT_FALSE(ba_cov->GetPointCov(kInvalidPoint3DId).has_value());
  }
}

INSTANTIATE_TEST_SUITE_P(
    BACovarianceTests,
    ParameterizedBACovarianceTests,
    ::testing::Values(
        std::make_pair(BACovarianceOptions(), BACovarianceTestOptions()),
        []() {
          BACovarianceOptions options;
          options.params = BACovarianceOptions::Params::ALL;
          BACovarianceTestOptions test_options;
          test_options.fixed_points = true;
          return std::make_pair(options, test_options);
        }(),
        []() {
          BACovarianceOptions options;
          options.params = BACovarianceOptions::Params::ALL;
          BACovarianceTestOptions test_options;
          test_options.fixed_cam_intrinsics = true;
          return std::make_pair(options, test_options);
        }(),
        []() {
          BACovarianceOptions options;
          options.params = BACovarianceOptions::Params::ALL;
          BACovarianceTestOptions test_options;
          test_options.fixed_cam_poses = true;
          return std::make_pair(options, test_options);
        }(),
        []() {
          BACovarianceOptions options;
          options.params = BACovarianceOptions::Params::POINTS;
          BACovarianceTestOptions test_options;
          return std::make_pair(options, test_options);
        }(),
        []() {
          BACovarianceOptions options;
          options.params = BACovarianceOptions::Params::POSES;
          BACovarianceTestOptions test_options;
          return std::make_pair(options, test_options);
        }(),
        []() {
          BACovarianceOptions options;
          options.params = BACovarianceOptions::Params::POSES_AND_POINTS;
          BACovarianceTestOptions test_options;
          return std::make_pair(options, test_options);
        }()));

}  // namespace
}  // namespace colmap
