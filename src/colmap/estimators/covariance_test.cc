// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/covariance.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/math/random.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/eigen_matchers.h"

#include <algorithm>

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

// Joint covariance of the stacked pose tangent spaces.
Eigen::MatrixXd JointPoseCov(const BACovariance& ba_cov,
                             const std::vector<image_t>& image_ids) {
  const int num_images = image_ids.size();
  Eigen::MatrixXd cov(6 * num_images, 6 * num_images);
  for (int i = 0; i < num_images; ++i) {
    for (int j = 0; j < num_images; ++j) {
      const std::optional<Eigen::MatrixXd> cov_ij =
          ba_cov.GetCamCrossCovFromWorld(image_ids[i], image_ids[j]);
      THROW_CHECK(cov_ij.has_value());
      cov.block<6, 6>(6 * i, 6 * j) = *cov_ij;
    }
  }
  return cov;
}

// Stacked pose tangent space deviations [rotation, translation] of the
// estimate from the ground truth, consistent with the Ceres quaternion
// manifold: Plus(q, d) = [cos(|d|), sin(|d|) * d / |d|] * q.
Eigen::VectorXd PoseDeviation(const Reconstruction& gt_reconstruction,
                              const Reconstruction& reconstruction,
                              const std::vector<image_t>& image_ids) {
  Eigen::VectorXd deviation(6 * image_ids.size());
  for (size_t i = 0; i < image_ids.size(); ++i) {
    const Rigid3d gt = gt_reconstruction.Image(image_ids[i]).CamFromWorld();
    const Rigid3d estimate = reconstruction.Image(image_ids[i]).CamFromWorld();
    const Eigen::AngleAxisd delta_rotation(estimate.rotation() *
                                           gt.rotation().inverse());
    deviation.segment<3>(6 * i) =
        0.5 * delta_rotation.angle() * delta_rotation.axis();
    deviation.segment<3>(6 * i + 3) = estimate.translation() - gt.translation();
  }
  return deviation;
}

struct MonteCarloProblem {
  Reconstruction gt_reconstruction;
  std::vector<image_t> image_ids;
  std::vector<point3D_t> constant_point3D_ids;
};

MonteCarloProblem CreateMonteCarloProblem(int num_constant_points,
                                          int num_points3D = 60,
                                          double scene_scale = 1) {
  MonteCarloProblem problem;
  SetPRNGSeed(0);
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 6;
  synthetic_dataset_options.num_points3D = num_points3D;
  SynthesizeDataset(synthetic_dataset_options, &problem.gt_reconstruction);
  problem.gt_reconstruction.Transform(Sim3d(
      scene_scale, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()));
  problem.image_ids = problem.gt_reconstruction.RegImageIds();
  std::sort(problem.image_ids.begin(), problem.image_ids.end());
  std::vector<point3D_t> point3D_ids;
  point3D_ids.reserve(problem.gt_reconstruction.NumPoints3D());
  for (const auto& [point3D_id, _] : problem.gt_reconstruction.Points3D()) {
    point3D_ids.push_back(point3D_id);
  }
  std::sort(point3D_ids.begin(), point3D_ids.end());
  problem.constant_point3D_ids.assign(
      point3D_ids.begin(),
      point3D_ids.begin() +
          std::min<size_t>(num_constant_points, point3D_ids.size()));
  return problem;
}

// Runs bundle adjustment to convergence and returns the adjuster.
std::unique_ptr<CeresBundleAdjuster> SolveMonteCarloProblem(
    const MonteCarloProblem& problem,
    bool constant_poses_and_intrinsics,
    Reconstruction& reconstruction) {
  BundleAdjustmentConfig config;
  for (const image_t image_id : problem.image_ids) {
    config.AddImage(image_id);
    if (constant_poses_and_intrinsics) {
      const Image& image = reconstruction.Image(image_id);
      config.SetConstantRigFromWorldPose(image.FrameId());
      config.SetConstantCamIntrinsics(image.CameraId());
    }
  }
  for (const point3D_t point3D_id : problem.constant_point3D_ids) {
    config.AddConstantPoint(point3D_id);
  }
  BundleAdjustmentOptions options;
  options.ceres->solver_options.max_num_iterations = 200;
  options.ceres->solver_options.gradient_tolerance = 1e-12;
  std::unique_ptr<CeresBundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  THROW_CHECK_EQ(bundle_adjuster->Solve()->termination_type,
                 BundleAdjustmentTerminationType::CONVERGENCE);
  return bundle_adjuster;
}

std::optional<BACovariance> EstimateGroundTruthCovariance(
    const MonteCarloProblem& problem,
    Reconstruction& gt_reconstruction,
    const BACovarianceOptions& cov_options = BACovarianceOptions()) {
  std::unique_ptr<CeresBundleAdjuster> bundle_adjuster =
      SolveMonteCarloProblem(problem,
                             /*constant_poses_and_intrinsics=*/false,
                             gt_reconstruction);
  return EstimateBACovariance(cov_options, gt_reconstruction, *bundle_adjuster);
}

constexpr double kMonteCarloPoint2DStddev = 0.5;
constexpr int kNumMonteCarloSamples = 300;

class ParameterizedBACovarianceMonteCarloTests
    : public ::testing::TestWithParam</*num_constant_points=*/int> {};

TEST_P(ParameterizedBACovarianceMonteCarloTests, PosesAndIntrinsics) {
  const int num_constant_points = GetParam();
  const MonteCarloProblem problem =
      CreateMonteCarloProblem(num_constant_points);
  Reconstruction gt_reconstruction = problem.gt_reconstruction;
  const std::optional<BACovariance> ba_cov =
      EstimateGroundTruthCovariance(problem, gt_reconstruction);
  ASSERT_TRUE(ba_cov.has_value());
  const double variance = kMonteCarloPoint2DStddev * kMonteCarloPoint2DStddev;
  const Eigen::MatrixXd pose_cov =
      variance * JointPoseCov(*ba_cov, problem.image_ids);
  Camera& gt_camera = gt_reconstruction.Camera(
      gt_reconstruction.Image(problem.image_ids[0]).CameraId());
  const std::optional<Eigen::MatrixXd> intrinsics_cov =
      ba_cov->GetOtherParamsCov(gt_camera.params.data());
  ASSERT_TRUE(intrinsics_cov.has_value());
  // Tangent space of SIMPLE_RADIAL with constant principal point.
  const std::vector<int> refined_param_idxs = {0, 3};
  ASSERT_EQ(intrinsics_cov->rows(),
            static_cast<int>(refined_param_idxs.size()));

  std::vector<Eigen::VectorXd> pose_samples;
  std::vector<Eigen::VectorXd> intrinsics_samples;
  for (int i = 0; i < kNumMonteCarloSamples; ++i) {
    Reconstruction reconstruction = gt_reconstruction;
    SetPRNGSeed(i + 1);
    SyntheticNoiseOptions synthetic_noise_options;
    synthetic_noise_options.point2D_stddev = kMonteCarloPoint2DStddev;
    SynthesizeNoise(synthetic_noise_options, &reconstruction);
    SolveMonteCarloProblem(
        problem, /*constant_poses_and_intrinsics=*/false, reconstruction);

    pose_samples.push_back(
        PoseDeviation(gt_reconstruction, reconstruction, problem.image_ids));

    const Camera& camera = reconstruction.Camera(gt_camera.camera_id);
    Eigen::VectorXd intrinsics_deviation(refined_param_idxs.size());
    for (size_t j = 0; j < refined_param_idxs.size(); ++j) {
      intrinsics_deviation(j) = camera.params[refined_param_idxs[j]] -
                                gt_camera.params[refined_param_idxs[j]];
    }
    intrinsics_samples.push_back(intrinsics_deviation);
  }

  EXPECT_THAT(pose_samples, SamplesFollowCov(pose_cov));
  EXPECT_THAT(intrinsics_samples, SamplesFollowCov(variance * *intrinsics_cov));
}

INSTANTIATE_TEST_SUITE_P(BACovarianceMonteCarloTests,
                         ParameterizedBACovarianceMonteCarloTests,
                         ::testing::Values(3));

TEST(BACovarianceMonteCarlo, ConditionalPoints) {
  const MonteCarloProblem problem =
      CreateMonteCarloProblem(/*num_constant_points=*/0);
  Reconstruction gt_reconstruction = problem.gt_reconstruction;
  BACovarianceOptions cov_options;
  cov_options.params = BACovarianceOptions::Params::POINTS;
  const std::optional<BACovariance> ba_cov =
      EstimateGroundTruthCovariance(problem, gt_reconstruction, cov_options);
  ASSERT_TRUE(ba_cov.has_value());

  // Point covariances are conditioned on the poses and intrinsics, so the
  // samples keep them constant. Stack all points into one sample.
  const double variance = kMonteCarloPoint2DStddev * kMonteCarloPoint2DStddev;
  std::vector<point3D_t> point3D_ids;
  for (const auto& [point3D_id, _] : gt_reconstruction.Points3D()) {
    point3D_ids.push_back(point3D_id);
  }
  Eigen::MatrixXd points_cov =
      Eigen::MatrixXd::Zero(3 * point3D_ids.size(), 3 * point3D_ids.size());
  for (size_t i = 0; i < point3D_ids.size(); ++i) {
    const std::optional<Eigen::MatrixXd> point_cov =
        ba_cov->GetPointCov(point3D_ids[i]);
    ASSERT_TRUE(point_cov.has_value());
    points_cov.block<3, 3>(3 * i, 3 * i) = variance * *point_cov;
  }

  std::vector<Eigen::VectorXd> samples;
  for (int i = 0; i < kNumMonteCarloSamples; ++i) {
    Reconstruction reconstruction = gt_reconstruction;
    SetPRNGSeed(i + 1);
    SyntheticNoiseOptions synthetic_noise_options;
    synthetic_noise_options.point2D_stddev = kMonteCarloPoint2DStddev;
    SynthesizeNoise(synthetic_noise_options, &reconstruction);
    SolveMonteCarloProblem(
        problem, /*constant_poses_and_intrinsics=*/true, reconstruction);
    Eigen::VectorXd sample(3 * point3D_ids.size());
    for (size_t j = 0; j < point3D_ids.size(); ++j) {
      sample.segment<3>(3 * j) = reconstruction.Point3D(point3D_ids[j]).xyz -
                                 gt_reconstruction.Point3D(point3D_ids[j]).xyz;
    }
    samples.push_back(sample);
  }
  EXPECT_THAT(samples, SamplesFollowCov(points_cov));
}

}  // namespace
}  // namespace colmap
