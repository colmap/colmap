// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/estimators/covariance.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/math/random.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <unordered_set>

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
  SetPRNGSeed(42);

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

// Helper to set up a solved BA problem and return the covariance, the
// reconstruction, and the ceres problem for further querying.
struct SolvedBAProblem {
  Reconstruction reconstruction;
  std::unique_ptr<BundleAdjuster> bundle_adjuster;
  std::shared_ptr<ceres::Problem> problem;
  std::optional<BACovariance> ba_cov;
  std::vector<internal::PoseParam> poses;
};

SolvedBAProblem SetUpSolvedBA(const BACovarianceOptions& cov_options,
                              bool fixed_cam_intrinsics = false) {
  SetPRNGSeed(0);

  SolvedBAProblem result;

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &result.reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.01;
  SynthesizeNoise(synthetic_noise_options, &result.reconstruction);

  BundleAdjustmentConfig config;
  for (const auto& [image_id, image] : result.reconstruction.Images()) {
    config.AddImage(image_id);
    if (fixed_cam_intrinsics) {
      config.SetConstantCamIntrinsics(image.CameraId());
    }
  }

  // Fix Gauge by setting 3 points constant.
  int num_constant = 0;
  for (const auto& [point3D_id, _] : result.reconstruction.Points3D()) {
    if (++num_constant <= 3) {
      config.AddConstantPoint(point3D_id);
    }
  }

  result.bundle_adjuster = CreateDefaultBundleAdjuster(
      BundleAdjustmentOptions(), config, result.reconstruction);
  const auto summary = result.bundle_adjuster->Solve();
  CHECK(summary->IsSolutionUsable());

  auto* ceres_ba =
      dynamic_cast<CeresBundleAdjuster*>(result.bundle_adjuster.get());
  CHECK_NOTNULL(ceres_ba);
  result.problem = ceres_ba->Problem();

  auto ba_cov_opt =
      EstimateBACovariance(cov_options, result.reconstruction, *ceres_ba);
  CHECK(ba_cov_opt.has_value());
  result.ba_cov.emplace(std::move(*ba_cov_opt));

  result.poses =
      internal::GetPoseParams(result.reconstruction, *result.problem);

  return result;
}

// Test GetCam2CovFromCam1 success path: compute relative pose covariance
// between two cameras and verify it produces a 6x6 symmetric PSD matrix.
TEST(BACovarianceTest, GetCam2CovFromCam1Success) {
  BACovarianceOptions options;
  options.params = BACovarianceOptions::Params::ALL;
  auto solved = SetUpSolvedBA(options);

  ASSERT_GE(solved.poses.size(), 2);

  const image_t id1 = solved.poses[0].image_id;
  const image_t id2 = solved.poses[1].image_id;
  const Rigid3d& cam1_from_world =
      solved.reconstruction.Image(id1).CamFromWorld();
  const Rigid3d& cam2_from_world =
      solved.reconstruction.Image(id2).CamFromWorld();

  const auto rel_cov = solved.ba_cov->GetCam2CovFromCam1(
      id1, cam1_from_world, id2, cam2_from_world);
  ASSERT_TRUE(rel_cov.has_value());
  EXPECT_EQ(rel_cov->rows(), 6);
  EXPECT_EQ(rel_cov->cols(), 6);

  // Covariance matrix should be symmetric.
  ExpectNearEigenMatrixXd(*rel_cov, rel_cov->transpose(), 1e-10);

  // Covariance matrix should be positive semi-definite (all eigenvalues >= 0).
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(*rel_cov);
  ASSERT_EQ(eigensolver.info(), Eigen::Success);
  for (int i = 0; i < eigensolver.eigenvalues().size(); ++i) {
    EXPECT_GE(eigensolver.eigenvalues()(i), -1e-10);
  }

  // All diagonal entries should be positive (variances).
  for (int i = 0; i < 6; ++i) {
    EXPECT_GT((*rel_cov)(i, i), 0.0);
  }
}

// Test GetCam2CovFromCam1 returns nullopt when image_id1 is invalid.
TEST(BACovarianceTest, GetCam2CovFromCam1InvalidImage1) {
  BACovarianceOptions options;
  options.params = BACovarianceOptions::Params::ALL;
  auto solved = SetUpSolvedBA(options);

  ASSERT_GE(solved.poses.size(), 2);
  const image_t id2 = solved.poses[1].image_id;
  const Rigid3d& cam2_from_world =
      solved.reconstruction.Image(id2).CamFromWorld();

  const auto rel_cov = solved.ba_cov->GetCam2CovFromCam1(
      kInvalidImageId, Rigid3d(), id2, cam2_from_world);
  ASSERT_FALSE(rel_cov.has_value());
}

// Test GetCam2CovFromCam1 returns nullopt when image_id2 is invalid.
TEST(BACovarianceTest, GetCam2CovFromCam1InvalidImage2) {
  BACovarianceOptions options;
  options.params = BACovarianceOptions::Params::ALL;
  auto solved = SetUpSolvedBA(options);

  ASSERT_GE(solved.poses.size(), 1);
  const image_t id1 = solved.poses[0].image_id;
  const Rigid3d& cam1_from_world =
      solved.reconstruction.Image(id1).CamFromWorld();

  const auto rel_cov = solved.ba_cov->GetCam2CovFromCam1(
      id1, cam1_from_world, kInvalidImageId, Rigid3d());
  ASSERT_FALSE(rel_cov.has_value());
}

// Test GetCam2CovFromCam1 returns nullopt when poses are partially constant
// (covariance rows != 6). Uses constant_rig_from_world_rotation to fix
// rotation for all poses, making tangent size 3 (translation only).
TEST(BACovarianceTest, GetCam2CovFromCam1PartiallyConstantPoses) {
  SetPRNGSeed(0);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.01;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentOptions ba_options;
  // Fix rotation for all poses, only refine translation. This makes the
  // tangent size 3 instead of 6.
  ba_options.constant_rig_from_world_rotation = true;

  BundleAdjustmentConfig config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    config.AddImage(image_id);
    config.SetConstantCamIntrinsics(image.CameraId());
  }

  int num_constant = 0;
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    if (++num_constant <= 3) {
      config.AddConstantPoint(point3D_id);
    }
  }

  auto bundle_adjuster =
      CreateDefaultBundleAdjuster(ba_options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  auto* ceres_ba = dynamic_cast<CeresBundleAdjuster*>(bundle_adjuster.get());
  ASSERT_NE(ceres_ba, nullptr);

  BACovarianceOptions cov_options;
  cov_options.params = BACovarianceOptions::Params::POSES;
  auto ba_cov_opt =
      EstimateBACovariance(cov_options, reconstruction, *ceres_ba);
  ASSERT_TRUE(ba_cov_opt.has_value());

  auto problem = ceres_ba->Problem();
  const auto poses = internal::GetPoseParams(reconstruction, *problem);
  ASSERT_GE(poses.size(), 2);

  // GetCamCovFromWorld should return 3x3 matrices (translation only).
  for (const auto& pose : poses) {
    const auto cov = ba_cov_opt->GetCamCovFromWorld(pose.image_id);
    ASSERT_TRUE(cov.has_value());
    EXPECT_EQ(cov->rows(), 3);
    EXPECT_EQ(cov->cols(), 3);
  }

  // GetCam2CovFromCam1 should return nullopt because poses are partially
  // constant (rows != 6).
  const Rigid3d& cam1_from_world =
      reconstruction.Image(poses[0].image_id).CamFromWorld();
  const Rigid3d& cam2_from_world =
      reconstruction.Image(poses[1].image_id).CamFromWorld();
  const auto rel_cov = ba_cov_opt->GetCam2CovFromCam1(
      poses[0].image_id, cam1_from_world, poses[1].image_id, cam2_from_world);
  ASSERT_FALSE(rel_cov.has_value());
}

// Test EstimateBACovarianceFromProblem directly (not through the
// CeresBundleAdjuster wrapper).
TEST(BACovarianceTest, EstimateBACovarianceFromProblemDirect) {
  SetPRNGSeed(0);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.01;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    config.AddImage(image_id);
  }

  int num_constant = 0;
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    if (++num_constant <= 3) {
      config.AddConstantPoint(point3D_id);
    }
  }

  auto bundle_adjuster = CreateDefaultBundleAdjuster(
      BundleAdjustmentOptions(), config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  auto* ceres_ba = dynamic_cast<CeresBundleAdjuster*>(bundle_adjuster.get());
  ASSERT_NE(ceres_ba, nullptr);
  auto problem = ceres_ba->Problem();

  // Call EstimateBACovarianceFromProblem directly.
  BACovarianceOptions cov_options;
  cov_options.params = BACovarianceOptions::Params::ALL;
  auto ba_cov_from_problem =
      EstimateBACovarianceFromProblem(cov_options, reconstruction, *problem);
  ASSERT_TRUE(ba_cov_from_problem.has_value());

  // Also get covariance via the wrapper for comparison.
  auto ba_cov_from_wrapper =
      EstimateBACovariance(cov_options, reconstruction, *ceres_ba);
  ASSERT_TRUE(ba_cov_from_wrapper.has_value());

  // Both should produce the same pose covariances.
  const auto poses = internal::GetPoseParams(reconstruction, *problem);
  for (const auto& pose : poses) {
    const auto cov_direct =
        ba_cov_from_problem->GetCamCovFromWorld(pose.image_id);
    const auto cov_wrapper =
        ba_cov_from_wrapper->GetCamCovFromWorld(pose.image_id);
    ASSERT_TRUE(cov_direct.has_value());
    ASSERT_TRUE(cov_wrapper.has_value());
    ExpectNearEigenMatrixXd(*cov_direct, *cov_wrapper, 1e-10);
  }
}

// Test experimental_custom_poses: supply a custom set of pose parameter
// blocks for covariance estimation.
TEST(BACovarianceTest, ExperimentalCustomPoses) {
  SetPRNGSeed(0);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.01;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    config.AddImage(image_id);
  }

  int num_constant = 0;
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    if (++num_constant <= 3) {
      config.AddConstantPoint(point3D_id);
    }
  }

  auto bundle_adjuster = CreateDefaultBundleAdjuster(
      BundleAdjustmentOptions(), config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  auto* ceres_ba = dynamic_cast<CeresBundleAdjuster*>(bundle_adjuster.get());
  ASSERT_NE(ceres_ba, nullptr);
  auto problem = ceres_ba->Problem();

  // Get all poses normally.
  const auto all_poses = internal::GetPoseParams(reconstruction, *problem);
  ASSERT_GE(all_poses.size(), 3);

  // Use only a subset of poses as custom_poses.
  std::vector<internal::PoseParam> custom_poses(all_poses.begin(),
                                                all_poses.begin() + 2);

  BACovarianceOptions cov_options;
  cov_options.params = BACovarianceOptions::Params::POSES;
  cov_options.experimental_custom_poses = custom_poses;

  auto ba_cov_opt =
      EstimateBACovarianceFromProblem(cov_options, reconstruction, *problem);
  ASSERT_TRUE(ba_cov_opt.has_value());

  // The custom poses should have covariance.
  for (const auto& pose : custom_poses) {
    const auto cov = ba_cov_opt->GetCamCovFromWorld(pose.image_id);
    ASSERT_TRUE(cov.has_value());
    EXPECT_EQ(cov->rows(), 6);
    EXPECT_EQ(cov->cols(), 6);
  }

  // Poses not in the custom set should not have covariance.
  const auto cov_excluded =
      ba_cov_opt->GetCamCovFromWorld(all_poses[2].image_id);
  ASSERT_FALSE(cov_excluded.has_value());
}

// Test POSES_AND_POINTS mode with fixed intrinsics: exercises
// SchurEliminateOtherParams code path where other params exist but their
// covariance is not requested.
TEST(BACovarianceTest, PosesAndPointsWithIntrinsicsSchurElimination) {
  BACovarianceOptions options;
  options.params = BACovarianceOptions::Params::POSES_AND_POINTS;
  // Intrinsics are variable but we only ask for POSES_AND_POINTS, so the
  // other params (intrinsics) must be Schur-eliminated.
  auto solved = SetUpSolvedBA(options, /*fixed_cam_intrinsics=*/false);

  // Pose covariance should be available.
  ASSERT_FALSE(solved.poses.empty());
  for (const auto& pose : solved.poses) {
    const auto cov = solved.ba_cov->GetCamCovFromWorld(pose.image_id);
    ASSERT_TRUE(cov.has_value());
    EXPECT_EQ(cov->rows(), 6);
    EXPECT_EQ(cov->cols(), 6);

    // Verify symmetry.
    ExpectNearEigenMatrixXd(*cov, cov->transpose(), 1e-10);

    // Verify positive semi-definite.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(*cov);
    ASSERT_EQ(eigensolver.info(), Eigen::Success);
    for (int i = 0; i < eigensolver.eigenvalues().size(); ++i) {
      EXPECT_GE(eigensolver.eigenvalues()(i), -1e-10);
    }
  }

  // Point covariance should also be available.
  const auto points =
      internal::GetPointParams(solved.reconstruction, *solved.problem);
  ASSERT_FALSE(points.empty());
  for (const auto& point : points) {
    const auto cov = solved.ba_cov->GetPointCov(point.point3D_id);
    ASSERT_TRUE(cov.has_value());
  }

  // Other params covariance should NOT be available (not requested).
  const auto others = GetOtherParams(*solved.problem, solved.poses, points);
  for (const double* other : others) {
    ASSERT_FALSE(solved.ba_cov->GetOtherParamsCov(other).has_value());
  }
}

// Test POINTS-only mode with all poses and intrinsics fixed: exercises the
// early return path where no pose/other covs are computed and L_inv is empty.
TEST(BACovarianceTest, PointsOnlyWithFixedPosesAndIntrinsics) {
  SetPRNGSeed(0);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.01;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    config.AddImage(image_id);
    config.SetConstantRigFromWorldPose(image.FrameId());
    config.SetConstantCamIntrinsics(image.CameraId());
  }

  int num_constant = 0;
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    if (++num_constant <= 3) {
      config.AddConstantPoint(point3D_id);
    }
  }

  auto bundle_adjuster = CreateDefaultBundleAdjuster(
      BundleAdjustmentOptions(), config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  auto* ceres_ba = dynamic_cast<CeresBundleAdjuster*>(bundle_adjuster.get());
  ASSERT_NE(ceres_ba, nullptr);

  BACovarianceOptions cov_options;
  cov_options.params = BACovarianceOptions::Params::POINTS;
  auto ba_cov_opt =
      EstimateBACovariance(cov_options, reconstruction, *ceres_ba);
  ASSERT_TRUE(ba_cov_opt.has_value());

  // Point covs should be available.
  const auto problem = ceres_ba->Problem();
  const auto points = internal::GetPointParams(reconstruction, *problem);
  ASSERT_FALSE(points.empty());
  for (const auto& point : points) {
    const auto cov = ba_cov_opt->GetPointCov(point.point3D_id);
    ASSERT_TRUE(cov.has_value());
    EXPECT_EQ(cov->rows(), 3);
    EXPECT_EQ(cov->cols(), 3);

    // Verify symmetry.
    ExpectNearEigenMatrixXd(*cov, cov->transpose(), 1e-10);

    // Verify positive semi-definite.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(*cov);
    ASSERT_EQ(eigensolver.info(), Eigen::Success);
    for (int i = 0; i < eigensolver.eigenvalues().size(); ++i) {
      EXPECT_GE(eigensolver.eigenvalues()(i), -1e-10);
    }
  }

  // Pose covs should not be available (not requested and poses are constant).
  const auto poses = internal::GetPoseParams(reconstruction, *problem);
  ASSERT_TRUE(poses.empty());  // All poses are constant.
}

// Test that pose covariance matrices are symmetric and positive semi-definite.
TEST(BACovarianceTest, PoseCovarianceIsSymmetricAndPSD) {
  BACovarianceOptions options;
  options.params = BACovarianceOptions::Params::POSES;
  auto solved = SetUpSolvedBA(options, /*fixed_cam_intrinsics=*/true);

  ASSERT_FALSE(solved.poses.empty());

  for (const auto& pose : solved.poses) {
    const auto cov = solved.ba_cov->GetCamCovFromWorld(pose.image_id);
    ASSERT_TRUE(cov.has_value());
    EXPECT_EQ(cov->rows(), cov->cols());

    // Symmetry.
    ExpectNearEigenMatrixXd(*cov, cov->transpose(), 1e-10);

    // Positive semi-definite.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(*cov);
    ASSERT_EQ(eigensolver.info(), Eigen::Success);
    for (int i = 0; i < eigensolver.eigenvalues().size(); ++i) {
      EXPECT_GE(eigensolver.eigenvalues()(i), -1e-10);
    }
  }
}

// Test cross-covariance relationship: cov(A, B) = cov(B, A)^T.
TEST(BACovarianceTest, CrossCovarianceTransposeRelationship) {
  BACovarianceOptions options;
  options.params = BACovarianceOptions::Params::ALL;
  auto solved = SetUpSolvedBA(options);

  ASSERT_GE(solved.poses.size(), 2);
  const image_t id1 = solved.poses[0].image_id;
  const image_t id2 = solved.poses[1].image_id;

  const auto cov_12 = solved.ba_cov->GetCamCrossCovFromWorld(id1, id2);
  const auto cov_21 = solved.ba_cov->GetCamCrossCovFromWorld(id2, id1);
  ASSERT_TRUE(cov_12.has_value());
  ASSERT_TRUE(cov_21.has_value());

  // cov(1,2) should equal cov(2,1)^T.
  ExpectNearEigenMatrixXd(*cov_12, cov_21->transpose(), 1e-10);
}

// Test that self-cross-covariance equals the regular covariance.
TEST(BACovarianceTest, SelfCrossCovarianceEqualsCovariance) {
  BACovarianceOptions options;
  options.params = BACovarianceOptions::Params::ALL;
  auto solved = SetUpSolvedBA(options);

  ASSERT_FALSE(solved.poses.empty());
  const image_t id = solved.poses[0].image_id;

  const auto cov = solved.ba_cov->GetCamCovFromWorld(id);
  const auto cross_cov = solved.ba_cov->GetCamCrossCovFromWorld(id, id);
  ASSERT_TRUE(cov.has_value());
  ASSERT_TRUE(cross_cov.has_value());

  ExpectNearEigenMatrixXd(*cov, *cross_cov, 1e-10);
}

// Test that damping affects the result: higher damping should generally
// produce smaller covariance values (regularization effect).
TEST(BACovarianceTest, DampingAffectsCovariance) {
  SetPRNGSeed(0);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.01;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    config.AddImage(image_id);
    config.SetConstantCamIntrinsics(image.CameraId());
  }

  int num_constant = 0;
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    if (++num_constant <= 3) {
      config.AddConstantPoint(point3D_id);
    }
  }

  auto bundle_adjuster = CreateDefaultBundleAdjuster(
      BundleAdjustmentOptions(), config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  auto* ceres_ba = dynamic_cast<CeresBundleAdjuster*>(bundle_adjuster.get());
  ASSERT_NE(ceres_ba, nullptr);

  // Low damping.
  BACovarianceOptions options_low;
  options_low.params = BACovarianceOptions::Params::POSES_AND_POINTS;
  options_low.damping = 1e-10;
  auto ba_cov_low =
      EstimateBACovariance(options_low, reconstruction, *ceres_ba);
  ASSERT_TRUE(ba_cov_low.has_value());

  // High damping.
  BACovarianceOptions options_high;
  options_high.params = BACovarianceOptions::Params::POSES_AND_POINTS;
  options_high.damping = 1.0;
  auto ba_cov_high =
      EstimateBACovariance(options_high, reconstruction, *ceres_ba);
  ASSERT_TRUE(ba_cov_high.has_value());

  const auto poses =
      internal::GetPoseParams(reconstruction, *ceres_ba->Problem());
  ASSERT_FALSE(poses.empty());

  // Higher damping should produce covariance with smaller or equal trace
  // (sum of variances). The damping adds to the diagonal of the Hessian,
  // shrinking the inverse.
  const auto cov_low = ba_cov_low->GetCamCovFromWorld(poses[0].image_id);
  const auto cov_high = ba_cov_high->GetCamCovFromWorld(poses[0].image_id);
  ASSERT_TRUE(cov_low.has_value());
  ASSERT_TRUE(cov_high.has_value());
  EXPECT_GT(cov_low->trace(), cov_high->trace());
}

// Test GetOtherParams excludes both pose and point params, returning only
// remaining variable blocks (e.g., intrinsics).
TEST(BACovarianceTest, GetOtherParamsExcludesPosesAndPoints) {
  SetPRNGSeed(0);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    config.AddImage(image_id);
  }

  int num_constant = 0;
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    if (++num_constant <= 3) {
      config.AddConstantPoint(point3D_id);
    }
  }

  auto bundle_adjuster = CreateDefaultBundleAdjuster(
      BundleAdjustmentOptions(), config, reconstruction);
  bundle_adjuster->Solve();

  auto* ceres_ba = dynamic_cast<CeresBundleAdjuster*>(bundle_adjuster.get());
  ASSERT_NE(ceres_ba, nullptr);
  auto problem = ceres_ba->Problem();

  const auto poses = internal::GetPoseParams(reconstruction, *problem);
  const auto points = internal::GetPointParams(reconstruction, *problem);
  const auto others = internal::GetOtherParams(*problem, poses, points);

  // With 1 camera, intrinsics are the "other" params.
  ASSERT_EQ(others.size(), 1);

  // Verify none of the other params are pose or point params.
  std::unordered_set<const double*> pose_and_point_ptrs;
  for (const auto& pose : poses) {
    pose_and_point_ptrs.insert(pose.cam_from_world);
  }
  for (const auto& point : points) {
    pose_and_point_ptrs.insert(point.xyz);
  }
  for (const double* other : others) {
    EXPECT_EQ(pose_and_point_ptrs.count(other), 0);
  }
}

// Test GetPointParams and GetPoseParams skip constant parameter blocks.
TEST(BACovarianceTest, GetParamsSkipsConstantBlocks) {
  SetPRNGSeed(0);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  BundleAdjustmentConfig config;
  bool first_pose_fixed = false;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    config.AddImage(image_id);
    if (!first_pose_fixed) {
      config.SetConstantRigFromWorldPose(image.FrameId());
      first_pose_fixed = true;
    }
  }

  // Fix all points.
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    config.AddConstantPoint(point3D_id);
  }

  auto bundle_adjuster = CreateDefaultBundleAdjuster(
      BundleAdjustmentOptions(), config, reconstruction);
  bundle_adjuster->Solve();

  auto* ceres_ba = dynamic_cast<CeresBundleAdjuster*>(bundle_adjuster.get());
  ASSERT_NE(ceres_ba, nullptr);
  auto problem = ceres_ba->Problem();

  const auto poses = internal::GetPoseParams(reconstruction, *problem);
  // One pose is fixed, so we should have num_frames - 1 variable poses.
  EXPECT_EQ(poses.size(), 4);

  const auto points = internal::GetPointParams(reconstruction, *problem);
  // All points are fixed.
  EXPECT_TRUE(points.empty());
}

// Test GetOtherParams with all intrinsics set constant: should return empty.
TEST(BACovarianceTest, GetOtherParamsEmptyWhenIntrinsicsConstant) {
  SetPRNGSeed(0);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 20;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  BundleAdjustmentConfig config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    config.AddImage(image_id);
    config.SetConstantCamIntrinsics(image.CameraId());
  }

  int num_constant = 0;
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    if (++num_constant <= 3) {
      config.AddConstantPoint(point3D_id);
    }
  }

  auto bundle_adjuster = CreateDefaultBundleAdjuster(
      BundleAdjustmentOptions(), config, reconstruction);
  bundle_adjuster->Solve();

  auto* ceres_ba = dynamic_cast<CeresBundleAdjuster*>(bundle_adjuster.get());
  ASSERT_NE(ceres_ba, nullptr);
  auto problem = ceres_ba->Problem();

  const auto poses = internal::GetPoseParams(reconstruction, *problem);
  const auto points = internal::GetPointParams(reconstruction, *problem);
  const auto others = internal::GetOtherParams(*problem, poses, points);
  EXPECT_TRUE(others.empty());
}

}  // namespace
}  // namespace colmap
