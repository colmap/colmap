// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/covariance.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <ceres/evaluation_callback.h>
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

TEST(BACovariance, PointOnlySupportsNonReferenceRigImages) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 2;
  dataset_options.num_frames_per_rig = 4;
  dataset_options.num_points3D = 20;
  SynthesizeDataset(dataset_options, &reconstruction);

  BundleAdjustmentConfig config;
  size_t num_non_reference_images = 0;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    config.AddImage(image_id);
    num_non_reference_images += !image.IsRefInFrame();
  }
  ASSERT_GT(num_non_reference_images, 0);
  auto ba = CreateDefaultCeresBundleAdjuster(
      BundleAdjustmentOptions(), config, reconstruction);
  ceres::Problem& problem = *ba->Problem();
  BACovarianceOptions options;
  options.damping = 0;
  // Pose covariance for non-reference images remains unsupported.
  options.params = BACovarianceOptions::Params::POSES;
  EXPECT_THROW(EstimateBACovariance(options, reconstruction, *ba),
               std::invalid_argument);
  options.params = BACovarianceOptions::Params::POINTS;
  const auto cov = EstimateBACovariance(options, reconstruction, *ba);
  ASSERT_TRUE(cov.has_value());

  // The point covariance is conditional on all other variables held fixed.
  // Build the independent Ceres oracle only after observing the target result.
  const auto points = internal::GetPointParams(reconstruction, problem);
  ASSERT_EQ(points.size(), dataset_options.num_points3D);
  FlatHashSet<const double*> point_params;
  std::vector<std::pair<const double*, const double*>> point_pairs;
  std::vector<Eigen::Matrix3d> actual_point_covs;
  for (const auto& point : points) {
    point_params.insert(point.xyz);
    point_pairs.emplace_back(point.xyz, point.xyz);
    const auto actual = cov->GetPointCov(point.point3D_id);
    ASSERT_TRUE(actual.has_value());
    actual_point_covs.push_back(*actual);
  }
  std::vector<double*> all_params;
  problem.GetParameterBlocks(&all_params);
  for (double* params : all_params) {
    if (point_params.count(params) == 0) {
      problem.SetParameterBlockConstant(params);
    }
  }
  ceres::Covariance oracle{ceres::Covariance::Options()};
  ASSERT_TRUE(oracle.Compute(point_pairs, &problem));
  for (size_t i = 0; i < points.size(); ++i) {
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> expected;
    ASSERT_TRUE(oracle.GetCovarianceBlockInTangentSpace(
        points[i].xyz, points[i].xyz, expected.data()));
    ExpectNearEigenMatrixXd(actual_point_covs[i], expected, 1e-8);
  }
  for (const auto& [image_id, image] : reconstruction.Images()) {
    EXPECT_FALSE(cov->GetCamCovFromWorld(image_id).has_value());
    EXPECT_FALSE(
        cov->GetOtherParamsCov(image.CameraPtr()->params.data()).has_value());
  }
}

TEST(BACovariance, PointOnlyEmptySelectionDoesNotEvaluateOtherParameters) {
  struct EvaluationCounter : ceres::EvaluationCallback {
    void PrepareForEvaluation(bool, bool) override { ++calls; }
    int calls = 0;
  } counter;
  ceres::Problem::Options problem_options;
  problem_options.evaluation_callback = &counter;
  ceres::Problem problem(problem_options);
  Reconstruction reconstruction;
  const point3D_t point_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(1, 2, 3), Track());
  double* point = reconstruction.Point3D(point_id).xyz.data();
  Eigen::Vector3d other(4, 5, 6);
  for (double* params : {point, other.data()}) {
    problem.AddResidualBlock(new ceres::NormalPrior(Eigen::Matrix3d::Identity(),
                                                    Eigen::Vector3d::Zero()),
                             nullptr,
                             params);
  }
  problem.SetParameterBlockConstant(point);
  BACovarianceOptions options;
  options.params = BACovarianceOptions::Params::POINTS;
  options.damping = 0;
  const auto empty =
      EstimateBACovarianceFromProblem(options, reconstruction, problem);
  ASSERT_TRUE(empty.has_value());
  EXPECT_FALSE(empty->GetPointCov(point_id).has_value());
  EXPECT_FALSE(empty->GetOtherParamsCov(other.data()).has_value());
  EXPECT_EQ(counter.calls, 0);
  EXPECT_TRUE(problem.IsParameterBlockConstant(point));
  EXPECT_EQ(reconstruction.Point3D(point_id).xyz, Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(other, Eigen::Vector3d(4, 5, 6));

  // A subsequent nonempty request must still evaluate the real point block.
  problem.SetParameterBlockVariable(point);
  const auto populated =
      EstimateBACovarianceFromProblem(options, reconstruction, problem);
  ASSERT_TRUE(populated.has_value());
  const auto point_cov = populated->GetPointCov(point_id);
  ASSERT_TRUE(point_cov.has_value());
  ExpectNearEigenMatrixXd(*point_cov, Eigen::Matrix3d::Identity(), 1e-12);
  EXPECT_EQ(counter.calls, 1);
  EXPECT_FALSE(populated->GetOtherParamsCov(other.data()).has_value());
}

struct LinearCovarianceCost {
  template <typename T>
  bool operator()(const T* pose,
                  const T* other,
                  const T* point,
                  T* residuals) const {
    residuals[0] = T(2) * pose[4] + T(3) * other[0] + point[0];
    residuals[1] = T(3) * pose[5] + point[1];
    residuals[2] = T(5) * other[0] + point[2];
    residuals[3] = T(2) * point[0];
    residuals[4] = T(3) * point[1];
    residuals[5] = T(4) * point[2];
    return true;
  }
};

TEST(BACovariance, ReturnedCovarianceOwnsItsResults) {
  // Columns are pose tx/ty, the other parameter, and point x/y/z.
  Eigen::Matrix<double, 6, 6> jacobian;
  jacobian << 2, 0, 3, 1, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 5, 0, 0, 1, 0, 0, 0, 2,
      0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 4;
  const Eigen::Matrix<double, 6, 6> hessian = jacobian.transpose() * jacobian;
  const Eigen::Matrix<double, 6, 6> expected = hessian.inverse();
  const Eigen::Matrix3d expected_point =
      hessian.bottomRightCorner<3, 3>().inverse();

  for (const auto params : {BACovarianceOptions::Params::POINTS,
                            BACovarianceOptions::Params::ALL}) {
    SCOPED_TRACE(static_cast<int>(params));
    Reconstruction reconstruction;
    SyntheticDatasetOptions dataset;
    dataset.num_rigs = 1;
    dataset.num_cameras_per_rig = 1;
    dataset.num_frames_per_rig = 1;
    dataset.num_points3D = 1;
    dataset.num_points2D_without_point3D = 0;
    SynthesizeDataset(dataset, &reconstruction);
    const image_t image_id = reconstruction.RegImageIds().front();
    const point3D_t point_id = reconstruction.Points3D().begin()->first;
    double* const pose =
        reconstruction.Frame(reconstruction.Image(image_id).FrameId())
            .RigFromWorld()
            .params.data();
    double* const point = reconstruction.Point3D(point_id).xyz.data();
    double other = 0.25;
    BACovarianceOptions options;
    options.params = params;
    options.damping = 0;
    const auto covariance = [&]() {
      ceres::Problem problem;
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<LinearCovarianceCost, 6, 7, 1, 3>(
              new LinearCovarianceCost),
          nullptr,
          pose,
          &other,
          point);
      SetManifold(&problem, pose, CreateSubsetManifold(7, {0, 1, 2, 3, 6}));
      return EstimateBACovarianceFromProblem(options, reconstruction, problem);
    }();
    ASSERT_TRUE(covariance.has_value());

    // The problem and its evaluation scratch are gone before any observation.
    // The result's public readers must still return the known covariance.
    const auto point_cov = covariance->GetPointCov(point_id);
    ASSERT_TRUE(point_cov.has_value());
    ExpectNearEigenMatrixXd(*point_cov, expected_point, 1e-12);
    if (params == BACovarianceOptions::Params::ALL) {
      const auto pose_cov = covariance->GetCamCovFromWorld(image_id);
      const auto other_cov = covariance->GetOtherParamsCov(&other);
      ASSERT_TRUE(pose_cov.has_value());
      ASSERT_TRUE(other_cov.has_value());
      ExpectNearEigenMatrixXd(*pose_cov, expected.topLeftCorner<2, 2>(), 1e-12);
      ExpectNearEigenMatrixXd(*other_cov, expected.block<1, 1>(2, 2), 1e-12);
    } else {
      EXPECT_FALSE(covariance->GetCamCovFromWorld(image_id).has_value());
      EXPECT_FALSE(covariance->GetOtherParamsCov(&other).has_value());
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

}  // namespace
}  // namespace colmap
