// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/covariance.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/pose_prior.h"
#include "colmap/math/random.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/eigen_matchers.h"

#include <algorithm>
#include <array>
#include <tuple>

#include <ceres/crs_matrix.h>
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
          options.params = BACovarianceOptions::Params::ALL;
          BACovarianceTestOptions test_options;
          test_options.fixed_cam_poses = true;
          test_options.fixed_cam_intrinsics = true;
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
          options.params = BACovarianceOptions::Params::POSES;
          BACovarianceTestOptions test_options;
          test_options.fixed_cam_intrinsics = true;
          return std::make_pair(options, test_options);
        }(),
        []() {
          BACovarianceOptions options;
          options.params = BACovarianceOptions::Params::POSES_AND_POINTS;
          BACovarianceTestOptions test_options;
          return std::make_pair(options, test_options);
        }()));

// Orthonormal basis of the infinitesimal similarity transforms of the world
// in the stacked pose tangent spaces [rotation, translation]. Under
// X' = (1 + s) * (I + [w]_x) * X + t, cam_from_world changes by the rotation
// vector -R * w (left perturbation) and the translation s * t' - R * t, where
// t' is the camera translation. The tangent space of the Ceres quaternion
// manifold is half the rotation vector.
Eigen::MatrixXd SimilarityNullSpace(const Reconstruction& reconstruction,
                                    const std::vector<image_t>& image_ids) {
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6 * image_ids.size(), 7);
  for (size_t i = 0; i < image_ids.size(); ++i) {
    const Rigid3d cam_from_world =
        reconstruction.Image(image_ids[i]).CamFromWorld();
    const Eigen::Matrix3d R = cam_from_world.rotation().toRotationMatrix();
    H.block<3, 3>(6 * i, 3) = -0.5 * R;
    H.block<3, 3>(6 * i + 3, 0) = -R;
    H.block<3, 1>(6 * i + 3, 6) = cam_from_world.translation();
  }
  return Eigen::HouseholderQR<Eigen::MatrixXd>(H).householderQ() *
         Eigen::MatrixXd::Identity(H.rows(), 7);
}

// S-transforms `fixed_pose_cov` into the minimum-trace gauge (Förstner &
// Wrobel 2016, Eq. 4.224 with W = I) and compares in correlation-normalized
// coordinates so scale differences between rotation and translation cannot
// mask errors in the smaller block.
void ExpectPoseCovNear(const Eigen::MatrixXd& actual,
                       const Eigen::MatrixXd& fixed_pose_cov,
                       const Eigen::MatrixXd& null_space,
                       double tol) {
  ASSERT_EQ(actual.rows(), fixed_pose_cov.rows());
  ASSERT_EQ(actual.cols(), fixed_pose_cov.cols());
  const Eigen::MatrixXd S =
      Eigen::MatrixXd::Identity(actual.rows(), actual.cols()) -
      null_space * null_space.transpose();
  const Eigen::MatrixXd expected = S * fixed_pose_cov * S;
  const Eigen::VectorXd inv_std =
      expected.diagonal().cwiseSqrt().cwiseInverse();
  const Eigen::MatrixXd norm_actual =
      inv_std.asDiagonal() * actual * inv_std.asDiagonal();
  const Eigen::MatrixXd norm_expected =
      inv_std.asDiagonal() * expected * inv_std.asDiagonal();
  EXPECT_THAT(norm_actual, EigenMatrixNear(norm_expected, tol));
}

// Joint covariance of the stacked pose tangent spaces.
Eigen::MatrixXd JointPoseCov(const BACovariance& ba_cov,
                             const std::vector<image_t>& image_ids) {
  const int num_images = image_ids.size();
  Eigen::MatrixXd cov(6 * num_images, 6 * num_images);
  for (int i = 0; i < num_images; ++i) {
    for (int j = 0; j < num_images; ++j) {
      cov.block<6, 6>(6 * i, 6 * j) =
          ba_cov.GetCamCrossCovFromWorld(image_ids[i], image_ids[j]).value();
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
                                          double scene_scale = 1.0) {
  MonteCarloProblem problem;
  SetPRNGSeed(0);
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 6;
  synthetic_dataset_options.num_points3D = num_points3D;
  SynthesizeDataset(synthetic_dataset_options, &problem.gt_reconstruction);
  if (scene_scale != 1.0) {
    problem.gt_reconstruction.Transform(Sim3d(
        scene_scale, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()));
  }
  problem.image_ids = problem.gt_reconstruction.RegImageIds();
  std::vector<point3D_t> point3D_ids;
  point3D_ids.reserve(problem.gt_reconstruction.NumPoints3D());
  for (const auto& [point3D_id, _] : problem.gt_reconstruction.Points3D()) {
    point3D_ids.push_back(point3D_id);
  }
  std::sort(point3D_ids.begin(), point3D_ids.end());
  THROW_CHECK_LE(num_constant_points, static_cast<int>(point3D_ids.size()));
  problem.constant_point3D_ids.assign(
      point3D_ids.begin(), point3D_ids.begin() + num_constant_points);
  return problem;
}

struct SolvedBA {
  std::unique_ptr<CeresBundleAdjuster> adjuster;
  std::shared_ptr<CeresBundleAdjustmentSummary> summary;
};

// Runs bundle adjustment to tight convergence and returns adjuster + summary.
SolvedBA SolveBA(const BundleAdjustmentConfig& config,
                 Reconstruction& reconstruction) {
  BundleAdjustmentOptions options;
  options.ceres->solver_options.max_num_iterations = 200;
  options.ceres->solver_options.gradient_tolerance = 1e-12;
  std::unique_ptr<CeresBundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = std::dynamic_pointer_cast<CeresBundleAdjustmentSummary>(
      bundle_adjuster->Solve());
  THROW_CHECK_NOTNULL(summary);
  THROW_CHECK_EQ(summary->termination_type,
                 BundleAdjustmentTerminationType::CONVERGENCE);
  return {std::move(bundle_adjuster), summary};
}

// Runs bundle adjustment to convergence and returns the adjuster.
std::unique_ptr<CeresBundleAdjuster> SolveMonteCarloProblem(
    const MonteCarloProblem& problem, Reconstruction& reconstruction) {
  BundleAdjustmentConfig config;
  for (const image_t image_id : problem.image_ids) {
    config.AddImage(image_id);
  }
  for (const point3D_t point3D_id : problem.constant_point3D_ids) {
    config.AddConstantPoint(point3D_id);
  }
  return SolveBA(config, reconstruction).adjuster;
}

constexpr double kMonteCarloPoint2DStddev = 0.5;
constexpr int kNumMonteCarloSamples = 300;

class ParameterizedBACovarianceMonteCarloTests
    : public ::testing::TestWithParam</*num_constant_points=*/int> {};

TEST_P(ParameterizedBACovarianceMonteCarloTests, PosesAndIntrinsics) {
  const int num_constant_points = GetParam();
  const MonteCarloProblem problem =
      CreateMonteCarloProblem(num_constant_points);
  // A free gauge leaves a 7-DoF null space in the minimum-trace gauge.
  const int null_dim = num_constant_points == 0 ? 7 : 0;
  Reconstruction gt_reconstruction = problem.gt_reconstruction;
  std::unique_ptr<CeresBundleAdjuster> gt_bundle_adjuster =
      SolveMonteCarloProblem(problem, gt_reconstruction);
  const std::optional<BACovariance> ba_cov = EstimateBACovariance(
      BACovarianceOptions(), gt_reconstruction, *gt_bundle_adjuster);
  ASSERT_TRUE(ba_cov.has_value());
  const double variance = kMonteCarloPoint2DStddev * kMonteCarloPoint2DStddev;
  const Eigen::MatrixXd pose_cov =
      variance * JointPoseCov(*ba_cov, problem.image_ids);
  const Camera& gt_camera = gt_reconstruction.Camera(
      gt_reconstruction.Image(problem.image_ids[0]).CameraId());
  const std::optional<Eigen::MatrixXd> intrinsics_cov =
      ba_cov->GetOtherParamsCov(gt_camera.params.data());
  ASSERT_TRUE(intrinsics_cov.has_value());
  // Tangent space of SIMPLE_RADIAL with constant principal point.
  ASSERT_EQ(intrinsics_cov->rows(), 2);

  const Eigen::MatrixXd null_space =
      SimilarityNullSpace(gt_reconstruction, problem.image_ids);
  std::vector<Eigen::VectorXd> pose_samples;
  std::vector<Eigen::VectorXd> intrinsics_samples;
  for (int i = 0; i < kNumMonteCarloSamples; ++i) {
    Reconstruction reconstruction = gt_reconstruction;
    SetPRNGSeed(i + 1);
    SyntheticNoiseOptions synthetic_noise_options;
    synthetic_noise_options.point2D_stddev = kMonteCarloPoint2DStddev;
    SynthesizeNoise(synthetic_noise_options, &reconstruction);
    SolveMonteCarloProblem(problem, reconstruction);

    Eigen::VectorXd pose_deviation =
        PoseDeviation(gt_reconstruction, reconstruction, problem.image_ids);
    if (null_dim > 0) {
      // S-transformation of the deviations into the minimum-trace gauge
      // (Förstner & Wrobel 2016, Eq. 4.222).
      pose_deviation -= null_space * (null_space.transpose() * pose_deviation);
    }
    pose_samples.push_back(pose_deviation);

    const Camera& camera = reconstruction.Camera(gt_camera.camera_id);
    intrinsics_samples.push_back(
        Eigen::Vector2d(camera.params[0] - gt_camera.params[0],
                        camera.params[3] - gt_camera.params[3]));
  }

  EXPECT_THAT(pose_samples, SamplesFollowCov(pose_cov, null_dim));
  EXPECT_THAT(intrinsics_samples, SamplesFollowCov(variance * *intrinsics_cov));
}

INSTANTIATE_TEST_SUITE_P(BACovarianceMonteCarloTests,
                         ParameterizedBACovarianceMonteCarloTests,
                         ::testing::Values(0, 3));

// Fixes the remaining gauge minimally on `ceres_problem`: keeps the pose of
// `ref_image_id` constant if `pin_ref_pose` is true, and keeps one translation
// component of `scale_image_id` constant to fix the scale. Returns the index
// of that translation component.
int FixGaugeMinimally(Reconstruction& reconstruction,
                      ceres::Problem& ceres_problem,
                      image_t ref_image_id,
                      image_t scale_image_id,
                      bool pin_ref_pose) {
  if (pin_ref_pose) {
    Rigid3d& ref_from_world =
        reconstruction.Frame(reconstruction.Image(ref_image_id).FrameId())
            .RigFromWorld();
    ceres_problem.SetParameterBlockConstant(ref_from_world.params.data());
  }
  const Rigid3d ref_from_world =
      reconstruction.Image(ref_image_id).CamFromWorld();
  Rigid3d& scale_from_world =
      reconstruction.Frame(reconstruction.Image(scale_image_id).FrameId())
          .RigFromWorld();
  // Pick the largest component of the scale tangent direction
  // t_scale - R_scale * R_ref^T * t_ref to ensure a transverse gauge cut.
  const Eigen::Vector3d scale_dir =
      scale_from_world.translation() -
      (scale_from_world.rotation() * ref_from_world.rotation().inverse()) *
          ref_from_world.translation();
  int fixed_translation_idx = 0;
  scale_dir.cwiseAbs().maxCoeff(&fixed_translation_idx);
  SetManifold(
      &ceres_problem,
      scale_from_world.params.data(),
      CreateProductManifold(CreateEigenQuaternionManifold(),
                            CreateSubsetManifold(3, {fixed_translation_idx})));
  return fixed_translation_idx;
}

// Fixes the remaining gauge minimally on `ceres_problem` (pinning
// `free_image_ids[0]` when `pin_first_pose` is true, plus one translation
// component of the next pose to fix scale) and returns the zero-padded pose
// covariance on `free_image_ids` together with the underlying BACovariance.
std::pair<Eigen::MatrixXd, std::optional<BACovariance>>
EstimateMinimallyFixedGaugeCov(Reconstruction& reconstruction,
                               ceres::Problem& ceres_problem,
                               const std::vector<image_t>& free_image_ids,
                               image_t fixed_ref_image_id,
                               bool pin_first_pose) {
  const int scale_pose_idx = pin_first_pose ? 1 : 0;
  const int fixed_translation_idx =
      FixGaugeMinimally(reconstruction,
                        ceres_problem,
                        fixed_ref_image_id,
                        free_image_ids[scale_pose_idx],
                        pin_first_pose);

  std::optional<BACovariance> fixed_cov = EstimateBACovarianceFromProblem(
      BACovarianceOptions(), reconstruction, ceres_problem);
  if (!fixed_cov.has_value()) {
    return {Eigen::MatrixXd(), std::nullopt};
  }

  const int num_images = free_image_ids.size();
  const int first_var_idx = pin_first_pose ? 1 : 0;
  Eigen::MatrixXd fixed_pose_cov =
      Eigen::MatrixXd::Zero(6 * num_images, 6 * num_images);
  std::vector<std::vector<int>> tangent_idxs(num_images);
  for (int i = first_var_idx; i < num_images; ++i) {
    for (int k = 0; k < 6; ++k) {
      if (i != scale_pose_idx || k != 3 + fixed_translation_idx) {
        tangent_idxs[i].push_back(6 * i + k);
      }
    }
  }
  for (int i = first_var_idx; i < num_images; ++i) {
    for (int j = first_var_idx; j < num_images; ++j) {
      const std::optional<Eigen::MatrixXd> cov_ij =
          fixed_cov->GetCamCrossCovFromWorld(free_image_ids[i],
                                             free_image_ids[j]);
      THROW_CHECK(cov_ij.has_value());
      for (size_t r = 0; r < tangent_idxs[i].size(); ++r) {
        for (size_t c = 0; c < tangent_idxs[j].size(); ++c) {
          fixed_pose_cov(tangent_idxs[i][r], tangent_idxs[j][c]) =
              (*cov_ij)(r, c);
        }
      }
    }
  }
  return {fixed_pose_cov, std::move(fixed_cov)};
}

class ParameterizedFreeGaugeBACovarianceTests
    : public ::testing::TestWithParam<
          std::tuple</*num_points3D=*/int,
                     /*scene_scale=*/double,
                     /*world_offset=*/Eigen::Vector3d>> {};

TEST_P(ParameterizedFreeGaugeBACovarianceTests, IsMinimumTraceGauge) {
  const auto [num_points3D, scene_scale, world_offset] = GetParam();
  const MonteCarloProblem problem = CreateMonteCarloProblem(
      /*num_constant_points=*/0, num_points3D, scene_scale);
  Reconstruction reconstruction = problem.gt_reconstruction;
  if (!world_offset.isZero()) {
    reconstruction.Transform(
        Sim3d(1.0, Eigen::Quaterniond::Identity(), world_offset));
  }
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  std::unique_ptr<CeresBundleAdjuster> bundle_adjuster =
      SolveMonteCarloProblem(problem, reconstruction);
  ceres::Problem& ceres_problem = *bundle_adjuster->Problem();

  const std::optional<BACovariance> free_cov = EstimateBACovarianceFromProblem(
      BACovarianceOptions(), reconstruction, ceres_problem);
  ASSERT_TRUE(free_cov.has_value());
  const Eigen::MatrixXd free_pose_cov =
      JointPoseCov(*free_cov, problem.image_ids);

  const auto [fixed_pose_cov, fixed_cov] =
      EstimateMinimallyFixedGaugeCov(reconstruction,
                                     ceres_problem,
                                     problem.image_ids,
                                     problem.image_ids[0],
                                     /*pin_first_pose=*/true);
  ASSERT_TRUE(fixed_cov.has_value());

  const Eigen::MatrixXd null_space =
      SimilarityNullSpace(reconstruction, problem.image_ids);
  ExpectPoseCovNear(free_pose_cov, fixed_pose_cov, null_space, /*tol=*/1e-6);

  // Intrinsics are invariant to the gauge.
  const double* camera_params =
      reconstruction
          .Camera(reconstruction.Image(problem.image_ids[0]).CameraId())
          .params.data();
  const Eigen::MatrixXd free_intrinsics_cov =
      *free_cov->GetOtherParamsCov(camera_params);
  EXPECT_LT((free_intrinsics_cov - *fixed_cov->GetOtherParamsCov(camera_params))
                .norm(),
            1e-6 * free_intrinsics_cov.norm());
}

INSTANTIATE_TEST_SUITE_P(
    FreeGaugeBACovarianceTests,
    ParameterizedFreeGaugeBACovarianceTests,
    // The gauge must be detected independently of the scene scale, world
    // origin offset, and number of points.
    ::testing::Values(
        std::make_tuple(60, 1.0, Eigen::Vector3d::Zero()),
        std::make_tuple(60, 1e-3, Eigen::Vector3d::Zero()),
        std::make_tuple(60, 1.0, Eigen::Vector3d(10.0, 20.0, 1.0)),
        std::make_tuple(1000, 30.0, Eigen::Vector3d::Zero()),
        std::make_tuple(1000, 3000.0, Eigen::Vector3d::Zero())));

TEST(EstimateBACovariance, PartialGaugeFreedom) {
  const MonteCarloProblem problem =
      CreateMonteCarloProblem(/*num_constant_points=*/0);
  Reconstruction reconstruction = problem.gt_reconstruction;
  // Fixing the first pose leaves a 1-DoF scale gauge freedom on the remaining
  // poses.
  BundleAdjustmentConfig config;
  for (const image_t image_id : problem.image_ids) {
    config.AddImage(image_id);
  }
  const image_t ref_image_id = problem.image_ids[0];
  config.SetConstantRigFromWorldPose(
      reconstruction.Image(ref_image_id).FrameId());
  auto ceres_ba = CreateDefaultCeresBundleAdjuster(
      BundleAdjustmentOptions(), config, reconstruction);
  ceres::Problem& ceres_problem = *ceres_ba->Problem();

  const std::vector<image_t> free_image_ids(problem.image_ids.begin() + 1,
                                            problem.image_ids.end());
  const std::optional<BACovariance> free_cov = EstimateBACovarianceFromProblem(
      BACovarianceOptions(), reconstruction, ceres_problem);
  ASSERT_TRUE(free_cov.has_value());
  const Eigen::MatrixXd free_pose_cov = JointPoseCov(*free_cov, free_image_ids);

  const auto [fixed_pose_cov, fixed_cov] =
      EstimateMinimallyFixedGaugeCov(reconstruction,
                                     ceres_problem,
                                     free_image_ids,
                                     ref_image_id,
                                     /*pin_first_pose=*/false);
  ASSERT_TRUE(fixed_cov.has_value());

  const Rigid3d ref_from_world =
      reconstruction.Image(ref_image_id).CamFromWorld();
  Eigen::VectorXd scale_null = Eigen::VectorXd::Zero(6 * free_image_ids.size());
  for (size_t i = 0; i < free_image_ids.size(); ++i) {
    const Rigid3d cam_from_world =
        reconstruction.Image(free_image_ids[i]).CamFromWorld();
    scale_null.segment<3>(6 * i + 3) =
        cam_from_world.translation() -
        (cam_from_world.rotation() * ref_from_world.rotation().inverse()) *
            ref_from_world.translation();
  }
  scale_null.normalize();

  ExpectPoseCovNear(free_pose_cov, fixed_pose_cov, scale_null, /*tol=*/1e-6);
}

TEST(EstimateBACovariance, FreeGaugeWithIllConditionedAndDegeneratePoints) {
  const MonteCarloProblem problem =
      CreateMonteCarloProblem(/*num_constant_points=*/0);
  Reconstruction reconstruction = problem.gt_reconstruction;
  std::vector<image_t> image_ids = problem.image_ids;

  // Add a 7th camera co-located with image_ids[0] (pure rotation) that observes
  // all existing scene points so its pose is well-constrained while allowing a
  // zero-baseline 2-view point.
  const Image& ref_image = reconstruction.Image(image_ids[0]);
  const Camera& camera = reconstruction.Camera(ref_image.CameraId());
  const Rigid3d ref_from_world = ref_image.CamFromWorld();
  const Eigen::Quaterniond small_rot(
      Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitY()));
  const Rigid3d colocated_from_world(small_rot * ref_from_world.rotation(),
                                     small_rot * ref_from_world.translation());
  const image_t colocated_image_id = image_ids.back() + 1;
  Image colocated_image;
  colocated_image.SetImageId(colocated_image_id);
  colocated_image.SetCameraId(camera.camera_id);
  colocated_image.SetName("colocated.png");
  std::vector<Eigen::Vector2d> colocated_points2D;
  std::vector<point3D_t> colocated_point3D_ids;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    const std::optional<Eigen::Vector2d> uv =
        camera.ImgFromCam(colocated_from_world * point3D.xyz);
    if (uv.has_value()) {
      colocated_points2D.push_back(*uv);
      colocated_point3D_ids.push_back(point3D_id);
    }
  }
  colocated_image.SetPoints2D(colocated_points2D);
  reconstruction.AddImageWithTrivialFrame(colocated_image,
                                          colocated_from_world);
  for (size_t i = 0; i < colocated_point3D_ids.size(); ++i) {
    reconstruction.AddObservation(colocated_point3D_ids[i],
                                  TrackElement(colocated_image_id, i));
  }
  // Add an unconfigured image that shares only one 2-view point with
  // image_ids[0], so that point gets a single (2-row, k < 3) residual in Ceres
  // while leaving all 60 original points variable.
  const image_t unconfigured_image_id = colocated_image_id + 1;
  Image unconfigured_image;
  unconfigured_image.SetImageId(unconfigured_image_id);
  unconfigured_image.SetCameraId(camera.camera_id);
  unconfigured_image.SetName("unconfigured.png");
  reconstruction.AddImageWithTrivialFrame(unconfigured_image,
                                          colocated_from_world);
  image_ids.push_back(colocated_image_id);

  // Pick a second configured camera whose optical axis points in the same
  // half-space as image_ids[0] so distant points stay in front of both cameras.
  const Eigen::Vector3d ref_z =
      ref_from_world.rotation().inverse() * Eigen::Vector3d::UnitZ();
  image_t pair_image_id = image_ids[1];
  double best_dot = -1.0;
  for (size_t i = 1; i + 1 < image_ids.size(); ++i) {
    const Eigen::Vector3d cand_z =
        reconstruction.Image(image_ids[i]).CamFromWorld().rotation().inverse() *
        Eigen::Vector3d::UnitZ();
    if (ref_z.dot(cand_z) > best_dot) {
      best_dot = ref_z.dot(cand_z);
      pair_image_id = image_ids[i];
    }
  }
  const Eigen::Vector3d avg_dir =
      (ref_z +
       reconstruction.Image(pair_image_id).CamFromWorld().rotation().inverse() *
           Eigen::Vector3d::UnitZ())
          .normalized();
  const Eigen::Vector3d ref_center = ref_image.ProjectionCenter();

  auto add_point = [&](const Eigen::Vector3d& xyz,
                       const std::vector<image_t>& obs_ids) {
    Track track;
    for (const image_t obs_id : obs_ids) {
      Image& img = reconstruction.Image(obs_id);
      const std::optional<Eigen::Vector2d> uv =
          camera.ImgFromCam(img.CamFromWorld() * xyz);
      THROW_CHECK(uv.has_value());
      const point2D_t pt_idx = img.NumPoints2D();
      img.Points2D().push_back(Point2D{*uv});
      track.AddElement(obs_id, pt_idx);
    }
    return reconstruction.AddPoint3D(xyz, track);
  };

  // Moderate low-parallax (dist=150), extreme low-parallax (dist=1e7),
  // single-residual rank-2 (k=2 < 3), and zero-baseline co-located 2-view
  // (k=4, rank-2) points.
  add_point(ref_center + 150.0 * avg_dir, {image_ids[0], pair_image_id});
  add_point(ref_center + 1e7 * avg_dir, {image_ids[0], pair_image_id});
  const point3D_t single_residual_pt = add_point(
      ref_center + 5.0 * ref_z, {image_ids[0], unconfigured_image_id});
  const point3D_t zero_baseline_pt =
      add_point(ref_center + 2.0 * ref_z, {image_ids[0], colocated_image_id});

  BundleAdjustmentConfig config;
  for (const image_t image_id : image_ids) {
    config.AddImage(image_id);
  }
  auto ceres_ba = CreateDefaultCeresBundleAdjuster(
      BundleAdjustmentOptions(), config, reconstruction);
  ceres::Problem& ceres_problem = *ceres_ba->Problem();
  // Keep the single-residual point variable to test k < tangent_size SVD, and
  // add a zero-residual variable point to test the empty-rows path.
  ceres_problem.SetParameterBlockVariable(
      reconstruction.Point3D(single_residual_pt).xyz.data());
  const point3D_t zero_residual_pt =
      reconstruction.AddPoint3D(Eigen::Vector3d(1.0, 2.0, 3.0), Track());
  ceres_problem.AddParameterBlock(
      reconstruction.Point3D(zero_residual_pt).xyz.data(), 3);

  // Check both Params::ALL and Params::POSES before fixing the gauge on
  // ceres_problem.
  const BACovarianceOptions all_options;
  BACovarianceOptions poses_options;
  poses_options.params = BACovarianceOptions::Params::POSES;
  const std::optional<BACovariance> all_cov = EstimateBACovarianceFromProblem(
      all_options, reconstruction, ceres_problem);
  const std::optional<BACovariance> poses_cov = EstimateBACovarianceFromProblem(
      poses_options, reconstruction, ceres_problem);
  ASSERT_TRUE(all_cov.has_value());
  ASSERT_TRUE(poses_cov.has_value());

  // Verify damped point covariances for zero-residual, single-residual
  // (k=2 < 3), and zero-baseline rank-2 points.
  const std::optional<Eigen::MatrixXd> zero_res_cov =
      all_cov->GetPointCov(zero_residual_pt);
  ASSERT_TRUE(zero_res_cov.has_value());
  ExpectNearEigenMatrixXd(
      *zero_res_cov,
      (1.0 / all_options.damping) * Eigen::Matrix3d::Identity(),
      /*tol=*/1e-8);
  for (const point3D_t pt_id : {single_residual_pt, zero_baseline_pt}) {
    const std::optional<Eigen::MatrixXd> pt_cov = all_cov->GetPointCov(pt_id);
    ASSERT_TRUE(pt_cov.has_value());
    ASSERT_EQ(pt_cov->rows(), 3);
    ASSERT_EQ(pt_cov->cols(), 3);

    ceres::Problem::EvaluateOptions eval_opts;
    eval_opts.parameter_blocks = {reconstruction.Point3D(pt_id).xyz.data()};
    ceres::CRSMatrix J_crs;
    ASSERT_TRUE(
        ceres_problem.Evaluate(eval_opts, nullptr, nullptr, nullptr, &J_crs));
    Eigen::Matrix3d H_pp = Eigen::Matrix3d::Zero();
    for (int r = 0; r < J_crs.num_rows; ++r) {
      Eigen::Vector3d row = Eigen::Vector3d::Zero();
      for (int idx = J_crs.rows[r]; idx < J_crs.rows[r + 1]; ++idx) {
        row(J_crs.cols[idx]) = J_crs.values[idx];
      }
      H_pp.noalias() += row * row.transpose();
    }
    // Build an orthonormal basis [u1, u2, ref_z] so we invert only the
    // well-conditioned 2x2 transverse Hessian block rather than the ~1e13
    // conditioned 3x3 (H_pp + damping * I).
    Eigen::Matrix3d basis;
    basis.col(2) = ref_z;
    basis.col(0) =
        ref_z
            .cross(std::abs(ref_z.x()) < 0.9 ? Eigen::Vector3d::UnitX()
                                             : Eigen::Vector3d::UnitY())
            .normalized();
    basis.col(1) = ref_z.cross(basis.col(0));
    const Eigen::Matrix2d H_transverse =
        basis.leftCols<2>().transpose() * H_pp * basis.leftCols<2>();
    const Eigen::Matrix2d expected_transverse_cov =
        (H_transverse + all_options.damping * Eigen::Matrix2d::Identity())
            .inverse();
    const Eigen::Matrix3d pt_cov_in_basis =
        basis.transpose() * (*pt_cov) * basis;
    // pt_cov has norm 1 / damping = 1e8, so storing it in world coordinates
    // limits transverse accuracy to O(eps_mach / damping) ~ 2e-8.
    EXPECT_LT((pt_cov_in_basis.topLeftCorner<2, 2>() - expected_transverse_cov)
                  .norm(),
              1e-6 * expected_transverse_cov.norm() + 1e-15 * pt_cov->norm());
    EXPECT_NEAR(pt_cov_in_basis(2, 2),
                1.0 / all_options.damping,
                1e-6 / all_options.damping);
    EXPECT_LT(
        pt_cov_in_basis.col(2).head<2>().norm(),
        1e-6 * std::sqrt(expected_transverse_cov.norm() / all_options.damping));
  }

  const Eigen::MatrixXd free_all_pose_cov = JointPoseCov(*all_cov, image_ids);
  const Eigen::MatrixXd free_poses_pose_cov =
      JointPoseCov(*poses_cov, image_ids);

  const auto [fixed_pose_cov, fixed_cov] =
      EstimateMinimallyFixedGaugeCov(reconstruction,
                                     ceres_problem,
                                     image_ids,
                                     image_ids[0],
                                     /*pin_first_pose=*/true);
  ASSERT_TRUE(fixed_cov.has_value());
  const Eigen::MatrixXd null_space =
      SimilarityNullSpace(reconstruction, image_ids);
  ExpectPoseCovNear(
      free_all_pose_cov, fixed_pose_cov, null_space, /*tol=*/1e-6);
  ExpectPoseCovNear(
      free_poses_pose_cov, fixed_pose_cov, null_space, /*tol=*/1e-6);
}

TEST(EstimateBACovariance, OffsetOriginAndIllConditionedGaugeProjection) {
  // Large world offsets or extreme scene scales make unscaled gauge projection
  // too ill-conditioned; estimation must fail loudly rather than return a
  // silently corrupted covariance.
  const std::vector<Sim3d> ill_conditioned_transforms = {
      Sim3d(
          1.0, Eigen::Quaterniond::Identity(), Eigen::Vector3d(1e3, 2e3, 1e2)),
      Sim3d(1e8, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()),
      Sim3d(1e-8, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()),
  };
  for (const Sim3d& transform : ill_conditioned_transforms) {
    const MonteCarloProblem problem =
        CreateMonteCarloProblem(/*num_constant_points=*/0);
    Reconstruction reconstruction = problem.gt_reconstruction;
    reconstruction.Transform(transform);
    BundleAdjustmentConfig config;
    for (const image_t image_id : problem.image_ids) {
      config.AddImage(image_id);
    }
    std::unique_ptr<CeresBundleAdjuster> bundle_adjuster =
        CreateDefaultCeresBundleAdjuster(
            BundleAdjustmentOptions(), config, reconstruction);
    EXPECT_FALSE(EstimateBACovariance(
                     BACovarianceOptions(), reconstruction, *bundle_adjuster)
                     .has_value());
  }
}

TEST(EstimateBACovariance, FailsForNullSpaceBeyondGauge) {
  const MonteCarloProblem problem =
      CreateMonteCarloProblem(/*num_constant_points=*/0);
  Reconstruction reconstruction = problem.gt_reconstruction;
  // Two observations leave 2 DoF of the first pose unconstrained.
  const Image& image = reconstruction.Image(problem.image_ids[0]);
  int num_kept_observations = 0;
  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    if (image.Point2D(point2D_idx).HasPoint3D() &&
        ++num_kept_observations > 2) {
      reconstruction.DeleteObservation(image.ImageId(), point2D_idx);
    }
  }
  ASSERT_EQ(image.NumPoints3D(), 2);
  std::unique_ptr<CeresBundleAdjuster> bundle_adjuster =
      SolveMonteCarloProblem(problem, reconstruction);
  EXPECT_FALSE(EstimateBACovariance(
                   BACovarianceOptions(), reconstruction, *bundle_adjuster)
                   .has_value());

  // An unconstrained pose with no residuals produces exact zeros on the Schur
  // complement diagonal and must fail cleanly.
  Reconstruction unconstrained_rec = problem.gt_reconstruction;
  std::unique_ptr<CeresBundleAdjuster> unconstrained_ba =
      SolveMonteCarloProblem(problem, unconstrained_rec);
  ceres::Problem& unconstrained_problem = *unconstrained_ba->Problem();
  double unconstrained_pose[6] = {};
  unconstrained_problem.AddParameterBlock(unconstrained_pose, 6);
  BACovarianceOptions unconstrained_options;
  unconstrained_options.params = BACovarianceOptions::Params::POSES;
  unconstrained_options.experimental_custom_poses = {
      {problem.image_ids.back() + 1, unconstrained_pose}};
  EXPECT_FALSE(EstimateBACovarianceFromProblem(unconstrained_options,
                                               unconstrained_rec,
                                               unconstrained_problem)
                   .has_value());
}

struct LinearComboCost {
  double a;
  double b;
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = T(a) * x[0] + T(b) * x[1];
    return true;
  }
};

TEST(EstimateBACovariance, FailsForNullSpaceInOtherParams) {
  const MonteCarloProblem problem =
      CreateMonteCarloProblem(/*num_constant_points=*/3);
  Reconstruction reconstruction = problem.gt_reconstruction;
  std::unique_ptr<CeresBundleAdjuster> bundle_adjuster =
      SolveMonteCarloProblem(problem, reconstruction);
  ceres::Problem& ceres_problem = *bundle_adjuster->Problem();

  double extra_params[2] = {1.0, 1.0};
  // Rank-1 residual leaves a 1-DoF null space in `other` params while keeping
  // the Schur complement diagonal strictly positive; must be rejected as a
  // non-gauge null space.
  ceres_problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<LinearComboCost, 1, 2>(
          new LinearComboCost{1.0, 1.0}),
      nullptr,
      extra_params);
  EXPECT_FALSE(EstimateBACovarianceFromProblem(
                   BACovarianceOptions(), reconstruction, ceres_problem)
                   .has_value());
}

TEST(EstimateBACovariance, WeakPosePriorsMatchExactInverse) {
  const MonteCarloProblem problem =
      CreateMonteCarloProblem(/*num_constant_points=*/0);
  Reconstruction reconstruction = problem.gt_reconstruction;
  std::unique_ptr<CeresBundleAdjuster> bundle_adjuster =
      SolveMonteCarloProblem(problem, reconstruction);
  ceres::Problem& ceres_problem = *bundle_adjuster->Problem();

  // Attach weak position priors (stddev = 10 scene units, so scaled prior
  // eigenvalues sit around 3e-9, well above round-off). The problem is
  // full-rank and must use the exact inverse rather than projecting out the
  // prior-constrained directions as a free gauge.
  const Eigen::Matrix3d prior_cov = 100.0 * Eigen::Matrix3d::Identity();
  std::vector<const double*> all_pose_blocks;
  all_pose_blocks.reserve(problem.image_ids.size());
  for (const image_t image_id : problem.image_ids) {
    Rigid3d& rig_from_world =
        reconstruction.Frame(reconstruction.Image(image_id).FrameId())
            .RigFromWorld();
    ceres_problem.AddResidualBlock(
        CovarianceWeightedCostFunctor<AbsolutePosePositionPriorCostFunctor>::
            Create(prior_cov,
                   reconstruction.Image(image_id).ProjectionCenter()),
        nullptr,
        rig_from_world.params.data());
    all_pose_blocks.push_back(rig_from_world.params.data());
  }

  BACovarianceOptions options;
  options.params = BACovarianceOptions::Params::POSES;
  const std::optional<BACovariance> ba_cov =
      EstimateBACovarianceFromProblem(options, reconstruction, ceres_problem);
  ASSERT_TRUE(ba_cov.has_value());
  const Eigen::MatrixXd pose_cov = JointPoseCov(*ba_cov, problem.image_ids);

  std::vector<std::pair<const double*, const double*>> cov_param_pairs;
  for (const double* ptr1 : all_pose_blocks) {
    for (const double* ptr2 : all_pose_blocks) {
      cov_param_pairs.emplace_back(ptr1, ptr2);
    }
  }
  ceres::Covariance::Options ceres_cov_options;
  ceres_cov_options.algorithm_type = ceres::DENSE_SVD;
  ceres_cov_options.min_reciprocal_condition_number = 1e-14;
  ceres::Covariance ceres_cov_computer(ceres_cov_options);
  ASSERT_TRUE(ceres_cov_computer.Compute(cov_param_pairs, &ceres_problem));

  Eigen::MatrixXd ceres_pose_cov(pose_cov.rows(), pose_cov.cols());
  ASSERT_TRUE(ceres_cov_computer.GetCovarianceMatrixInTangentSpace(
      all_pose_blocks, ceres_pose_cov.data()));
  EXPECT_LT((pose_cov - ceres_pose_cov).norm(), 1e-6 * ceres_pose_cov.norm());
}

Reconstruction SynthesizeSingleCameraReconstruction(int num_frames,
                                                    int num_points3D,
                                                    double point2D_stddev,
                                                    int seed,
                                                    int width = 1024,
                                                    int height = 768) {
  SetPRNGSeed(seed);
  Reconstruction reconstruction;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 1;
  dataset_options.num_frames_per_rig = num_frames;
  dataset_options.num_points3D = num_points3D;
  dataset_options.camera_width = width;
  dataset_options.camera_height = height;
  dataset_options.camera_params = {1280.0, 0.5 * width, 0.5 * height, 0.05};
  SynthesizeDataset(dataset_options, &reconstruction);
  if (point2D_stddev > 0.0) {
    SyntheticNoiseOptions noise_options;
    noise_options.point2D_stddev = point2D_stddev;
    SynthesizeNoise(noise_options, &reconstruction);
  }
  return reconstruction;
}

BACovarianceOptions MakeCovarianceOptions(
    BACovarianceOptions::VarianceScaling variance_scaling,
    BACovarianceOptions::Params params = BACovarianceOptions::Params::ALL) {
  BACovarianceOptions options;
  options.variance_scaling = variance_scaling;
  options.params = params;
  return options;
}

BundleAdjustmentConfig CreateAllImagesBAConfig(
    const Reconstruction& reconstruction, bool constant_poses = false) {
  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
    if (constant_poses) {
      config.SetConstantRigFromWorldPose(
          reconstruction.Image(image_id).FrameId());
    }
  }
  return config;
}

void AddSpatialTileBias(Reconstruction& reconstruction,
                        const std::vector<image_t>& image_ids,
                        double tile_stddev,
                        double iid_stddev,
                        int seed) {
  SetPRNGSeed(seed);
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction.Image(image_id);
    const Camera& camera = *image.CameraPtr();
    const int nx = camera.width >= camera.height ? 3 : 2;
    const int ny = camera.width >= camera.height ? 2 : 3;
    std::vector<Eigen::Vector2d> tile_bias(nx * ny);
    for (auto& b : tile_bias) {
      b = Eigen::Vector2d(RandomGaussian<double>(0.0, tile_stddev),
                          RandomGaussian<double>(0.0, tile_stddev));
    }
    for (Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D()) {
        continue;
      }
      const int gx = std::clamp(
          static_cast<int>(point2D.xy.x() / camera.width * nx), 0, nx - 1);
      const int gy = std::clamp(
          static_cast<int>(point2D.xy.y() / camera.height * ny), 0, ny - 1);
      point2D.xy += tile_bias[gy * nx + gx];
      if (iid_stddev > 0.0) {
        point2D.xy += Eigen::Vector2d(RandomGaussian<double>(0.0, iid_stddev),
                                      RandomGaussian<double>(0.0, iid_stddev));
      }
    }
  }
}

TEST(EstimateBACovariance, VarianceScalingPosteriorAndSpatialDeffIID) {
  constexpr double kNoiseStddev = 2.5;
  Reconstruction reconstruction = SynthesizeSingleCameraReconstruction(
      /*num_frames=*/10, /*num_points3D=*/300, kNoiseStddev, /*seed=*/0);
  const std::vector<image_t> image_ids = reconstruction.RegImageIds();

  const auto [ceres_ba, summary] =
      SolveBA(CreateAllImagesBAConfig(reconstruction), reconstruction);

  const std::optional<BACovariance> none_cov = EstimateBACovariance(
      MakeCovarianceOptions(BACovarianceOptions::VarianceScaling::NONE),
      reconstruction,
      *ceres_ba);
  ASSERT_TRUE(none_cov.has_value());
  EXPECT_EQ(none_cov->PosteriorVarianceFactor(), 1.0);
  EXPECT_EQ(none_cov->SpatialDesignEffect(), 1.0);
  EXPECT_EQ(none_cov->VarianceFactor(), 1.0);

  const std::optional<BACovariance> post_cov = EstimateBACovariance(
      MakeCovarianceOptions(BACovarianceOptions::VarianceScaling::POSTERIOR),
      reconstruction,
      *ceres_ba);
  ASSERT_TRUE(post_cov.has_value());

  const int expected_redundancy =
      summary->ceres_summary.num_residuals_reduced -
      (summary->ceres_summary.num_effective_parameters_reduced - 7);
  const double expected_sigma0_sq =
      (2.0 * summary->ceres_summary.final_cost) / expected_redundancy;
  EXPECT_NEAR(post_cov->PosteriorVarianceFactor(), expected_sigma0_sq, 1e-10);
  EXPECT_GT(post_cov->PosteriorVarianceFactor(),
            0.7 * kNoiseStddev * kNoiseStddev);
  EXPECT_LT(post_cov->PosteriorVarianceFactor(),
            1.3 * kNoiseStddev * kNoiseStddev);
  EXPECT_EQ(post_cov->SpatialDesignEffect(), 1.0);
  EXPECT_EQ(post_cov->VarianceFactor(), post_cov->PosteriorVarianceFactor());

  // Params::POSES must yield the exact same posterior variance factor as ALL
  // because eliminated other parameters still consume degrees of freedom.
  const double* cam_params =
      reconstruction.Cameras().begin()->second.params.data();
  const std::optional<BACovariance> post_poses_cov = EstimateBACovariance(
      MakeCovarianceOptions(BACovarianceOptions::VarianceScaling::POSTERIOR,
                            BACovarianceOptions::Params::POSES),
      reconstruction,
      *ceres_ba);
  ASSERT_TRUE(post_poses_cov.has_value());
  EXPECT_NEAR(post_poses_cov->PosteriorVarianceFactor(),
              post_cov->PosteriorVarianceFactor(),
              1e-12);
  EXPECT_FALSE(post_poses_cov->GetOtherParamsCov(cam_params).has_value());

  // Under i.i.d. noise, SPATIAL_DEFF stays near 1.0 and scales point_covs by
  // sigma_0^2 only while scaling poses and others by sigma_0^2 * DEFF.
  const std::optional<BACovariance> deff_cov = EstimateBACovariance(
      MakeCovarianceOptions(BACovarianceOptions::VarianceScaling::SPATIAL_DEFF),
      reconstruction,
      *ceres_ba);
  ASSERT_TRUE(deff_cov.has_value());
  EXPECT_NEAR(deff_cov->PosteriorVarianceFactor(), expected_sigma0_sq, 1e-10);
  EXPECT_LT(deff_cov->SpatialDesignEffect(), 1.5);
  EXPECT_NEAR(
      deff_cov->VarianceFactor(),
      deff_cov->PosteriorVarianceFactor() * deff_cov->SpatialDesignEffect(),
      1e-12);

  const image_t id1 = image_ids[0];
  const point3D_t pt_id = reconstruction.Points3D().begin()->first;

  const double post_scale = post_cov->PosteriorVarianceFactor();
  const double deff_pose_scale = deff_cov->VarianceFactor();

  ExpectNearEigenMatrixXd(*post_cov->GetCamCovFromWorld(id1),
                          post_scale * (*none_cov->GetCamCovFromWorld(id1)),
                          1e-10);
  ExpectNearEigenMatrixXd(
      *deff_cov->GetCamCovFromWorld(id1),
      deff_pose_scale * (*none_cov->GetCamCovFromWorld(id1)),
      1e-10);
  ExpectNearEigenMatrixXd(
      *deff_cov->GetOtherParamsCov(cam_params),
      deff_pose_scale * (*none_cov->GetOtherParamsCov(cam_params)),
      1e-10);
  // Point covariance is conditional on fixed poses, so it scales by
  // posterior_variance_factor only (not spatial_design_effect).
  ExpectNearEigenMatrixXd(*post_cov->GetPointCov(pt_id),
                          post_scale * (*none_cov->GetPointCov(pt_id)),
                          1e-10);
  ExpectNearEigenMatrixXd(*deff_cov->GetPointCov(pt_id),
                          post_scale * (*none_cov->GetPointCov(pt_id)),
                          1e-10);
}

TEST(EstimateBACovariance,
     SpatialDesignEffectDetectsCorrelatedNoiseAndScalesWithDensity) {
  auto run_correlated = [](int num_points3D) {
    Reconstruction reconstruction = SynthesizeSingleCameraReconstruction(
        /*num_frames=*/10, num_points3D, /*point2D_stddev=*/0.0, /*seed=*/42);
    const std::vector<image_t> image_ids = reconstruction.RegImageIds();
    AddSpatialTileBias(reconstruction,
                       image_ids,
                       /*tile_stddev=*/1.0,
                       /*iid_stddev=*/0.5,
                       /*seed=*/123);

    auto ceres_ba =
        SolveBA(CreateAllImagesBAConfig(reconstruction), reconstruction)
            .adjuster;

    const std::optional<BACovariance> post_cov = EstimateBACovariance(
        MakeCovarianceOptions(BACovarianceOptions::VarianceScaling::POSTERIOR),
        reconstruction,
        *ceres_ba);
    THROW_CHECK(post_cov.has_value());

    const std::optional<BACovariance> deff_cov = EstimateBACovariance(
        MakeCovarianceOptions(
            BACovarianceOptions::VarianceScaling::SPATIAL_DEFF),
        reconstruction,
        *ceres_ba);
    THROW_CHECK(deff_cov.has_value());

    double post_trace = 0.0;
    double deff_trace = 0.0;
    for (const image_t image_id : image_ids) {
      post_trace += post_cov->GetCamCovFromWorld(image_id)->trace();
      deff_trace += deff_cov->GetCamCovFromWorld(image_id)->trace();
    }
    return std::make_tuple(
        deff_cov->SpatialDesignEffect(), post_trace, deff_trace);
  };

  const auto [deff_200, post_trace_200, deff_trace_200] = run_correlated(200);
  const auto [deff_800, post_trace_800, deff_trace_800] = run_correlated(800);

  // Spatial correlation must be strongly detected and scale up with density.
  EXPECT_GT(deff_200, 4.0);
  EXPECT_GT(deff_800, 2.0 * deff_200);
  // While unadjusted posterior variance collapses ~4x as points quadruple,
  // SPATIAL_DEFF variance stays roughly density-invariant.
  EXPECT_GT(post_trace_200 / post_trace_800, 2.5);
  EXPECT_LT(deff_trace_200 / deff_trace_800, 1.8);
  EXPECT_GT(deff_trace_200 / deff_trace_800, 0.5);

  // Portrait orientation (width < height) must use 2x3 rather than 3x2 tiles.
  // Inject a 3-row horizontal band bias [+b, -2b, +b] along y: each erroneous
  // 3x2 half-height tile averages (2/3)(+b) + (1/3)(-2b) = 0 (DEFF ~ 1), while
  // the correct 2x3 grid resolves all 3 bands cleanly (DEFF >> 4).
  {
    constexpr int kWidth = 768;
    constexpr int kHeight = 1024;
    Reconstruction portrait_recon = SynthesizeSingleCameraReconstruction(
        /*num_frames=*/10,
        /*num_points3D=*/300,
        /*point2D_stddev=*/0.5,
        /*seed=*/42,
        kWidth,
        kHeight);
    constexpr double kBandBias = 2.0;
    const std::array<double, 3> row_bias = {
        kBandBias, -2.0 * kBandBias, kBandBias};
    for (const image_t image_id : portrait_recon.RegImageIds()) {
      Image& image = portrait_recon.Image(image_id);
      for (Point2D& point2D : image.Points2D()) {
        if (!point2D.HasPoint3D()) {
          continue;
        }
        const int gy =
            std::clamp(static_cast<int>(point2D.xy.y() / kHeight * 3), 0, 2);
        point2D.xy.x() += row_bias[gy];
      }
    }
    auto ceres_ba =
        SolveBA(CreateAllImagesBAConfig(portrait_recon), portrait_recon)
            .adjuster;

    const std::optional<BACovariance> portrait_cov = EstimateBACovariance(
        MakeCovarianceOptions(
            BACovarianceOptions::VarianceScaling::SPATIAL_DEFF),
        portrait_recon,
        *ceres_ba);
    ASSERT_TRUE(portrait_cov.has_value());
    EXPECT_GT(portrait_cov->SpatialDesignEffect(), 10.0);
  }
}

TEST(EstimateBACovariance,
     VarianceScalingFailsOnInsufficientTilesAndNoiselessResiduals) {
  const BACovarianceOptions post_opts =
      MakeCovarianceOptions(BACovarianceOptions::VarianceScaling::POSTERIOR);
  const BACovarianceOptions deff_opts =
      MakeCovarianceOptions(BACovarianceOptions::VarianceScaling::SPATIAL_DEFF);

  // Insufficient observations per tile: POSTERIOR succeeds, SPATIAL_DEFF fails.
  {
    Reconstruction reconstruction = SynthesizeSingleCameraReconstruction(
        /*num_frames=*/6, /*num_points3D=*/15, /*point2D_stddev=*/1.0, 0);
    auto ceres_ba = CreateDefaultCeresBundleAdjuster(
        BundleAdjustmentOptions(),
        CreateAllImagesBAConfig(reconstruction),
        reconstruction);
    ASSERT_TRUE(ceres_ba->Solve()->IsSolutionUsable());

    EXPECT_TRUE(
        EstimateBACovariance(post_opts, reconstruction, *ceres_ba).has_value());
    EXPECT_FALSE(
        EstimateBACovariance(deff_opts, reconstruction, *ceres_ba).has_value());
  }

  // Exact noiseless residuals: POSTERIOR returns 0 variance, SPATIAL_DEFF fails
  // loudly due to zero residual variance.
  {
    Reconstruction reconstruction = SynthesizeSingleCameraReconstruction(
        /*num_frames=*/6, /*num_points3D=*/200, /*point2D_stddev=*/0.0, 0);
    auto ceres_ba = CreateDefaultCeresBundleAdjuster(
        BundleAdjustmentOptions(),
        CreateAllImagesBAConfig(reconstruction),
        reconstruction);

    const std::optional<BACovariance> post_cov =
        EstimateBACovariance(post_opts, reconstruction, *ceres_ba);
    ASSERT_TRUE(post_cov.has_value());
    EXPECT_NEAR(post_cov->PosteriorVarianceFactor(), 0.0, 1e-12);
    EXPECT_FALSE(
        EstimateBACovariance(deff_opts, reconstruction, *ceres_ba).has_value());
  }
}

TEST(EstimateBACovariance, VarianceScalingFailsOnNonPositiveRedundancy) {
  // Zero redundancy (2 residuals, 2 parameters): unscaled succeeds, variance
  // scaling fails loudly.
  Reconstruction reconstruction;
  ceres::Problem ceres_problem;
  double x[2] = {1.0, 2.0};
  ceres_problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<LinearComboCost, 1, 2>(
          new LinearComboCost{1.0, 1.0}),
      nullptr,
      x);
  ceres_problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<LinearComboCost, 1, 2>(
          new LinearComboCost{1.0, -1.0}),
      nullptr,
      x);
  EXPECT_TRUE(EstimateBACovarianceFromProblem(
                  BACovarianceOptions(), reconstruction, ceres_problem)
                  .has_value());
  EXPECT_FALSE(EstimateBACovarianceFromProblem(
                   MakeCovarianceOptions(
                       BACovarianceOptions::VarianceScaling::POSTERIOR),
                   reconstruction,
                   ceres_problem)
                   .has_value());
}

TEST(EstimateBACovariance, VarianceScalingWithFixedPosesAndPointsOnly) {
  const Reconstruction base_recon = SynthesizeSingleCameraReconstruction(
      /*num_frames=*/6, /*num_points3D=*/200, /*point2D_stddev=*/1.5, 0);

  // 1. Fixed camera poses (poses empty; intrinsics and points variable).
  {
    Reconstruction fixed_recon = base_recon;
    const auto [ceres_ba, summary] =
        SolveBA(CreateAllImagesBAConfig(fixed_recon, /*constant_poses=*/true),
                fixed_recon);

    const int expected_redundancy =
        summary->ceres_summary.num_residuals_reduced -
        summary->ceres_summary.num_effective_parameters_reduced;
    const double expected_sigma0_sq =
        (2.0 * summary->ceres_summary.final_cost) / expected_redundancy;

    const std::optional<BACovariance> deff_all = EstimateBACovariance(
        MakeCovarianceOptions(
            BACovarianceOptions::VarianceScaling::SPATIAL_DEFF),
        fixed_recon,
        *ceres_ba);
    ASSERT_TRUE(deff_all.has_value());
    EXPECT_NEAR(deff_all->PosteriorVarianceFactor(), expected_sigma0_sq, 1e-10);
    EXPECT_LT(deff_all->SpatialDesignEffect(), 1.5);

    const std::optional<BACovariance> deff_pts = EstimateBACovariance(
        MakeCovarianceOptions(
            BACovarianceOptions::VarianceScaling::SPATIAL_DEFF,
            BACovarianceOptions::Params::POINTS),
        fixed_recon,
        *ceres_ba);
    ASSERT_TRUE(deff_pts.has_value());
    EXPECT_NEAR(deff_pts->PosteriorVarianceFactor(), expected_sigma0_sq, 1e-10);
    EXPECT_NEAR(deff_pts->SpatialDesignEffect(),
                deff_all->SpatialDesignEffect(),
                1e-12);
    const point3D_t pt_id = fixed_recon.Points3D().begin()->first;
    ExpectNearEigenMatrixXd(
        *deff_pts->GetPointCov(pt_id), *deff_all->GetPointCov(pt_id), 1e-12);
  }

  // 2. Free-gauge BA with Params::POINTS (does not factor S, so num_null == 0).
  {
    Reconstruction free_recon = base_recon;
    const auto [ceres_ba, summary] =
        SolveBA(CreateAllImagesBAConfig(free_recon), free_recon);

    const int expected_redundancy_no_null =
        summary->ceres_summary.num_residuals_reduced -
        summary->ceres_summary.num_effective_parameters_reduced;
    const double expected_sigma0_sq =
        (2.0 * summary->ceres_summary.final_cost) / expected_redundancy_no_null;

    const std::optional<BACovariance> none_pts = EstimateBACovariance(
        MakeCovarianceOptions(BACovarianceOptions::VarianceScaling::NONE,
                              BACovarianceOptions::Params::POINTS),
        free_recon,
        *ceres_ba);
    ASSERT_TRUE(none_pts.has_value());

    const std::optional<BACovariance> deff_pts = EstimateBACovariance(
        MakeCovarianceOptions(
            BACovarianceOptions::VarianceScaling::SPATIAL_DEFF,
            BACovarianceOptions::Params::POINTS),
        free_recon,
        *ceres_ba);
    ASSERT_TRUE(deff_pts.has_value());
    EXPECT_NEAR(deff_pts->PosteriorVarianceFactor(), expected_sigma0_sq, 1e-10);
    EXPECT_LT(deff_pts->SpatialDesignEffect(), 1.6);
    const point3D_t pt_id = free_recon.Points3D().begin()->first;
    ExpectNearEigenMatrixXd(
        *deff_pts->GetPointCov(pt_id),
        deff_pts->PosteriorVarianceFactor() * (*none_pts->GetPointCov(pt_id)),
        1e-10);
  }
}

TEST(EstimateBACovariance, SpatialDesignEffectSubsetAndMixedConstantPoses) {
  Reconstruction reconstruction = SynthesizeSingleCameraReconstruction(
      /*num_frames=*/10, /*num_points3D=*/200, /*point2D_stddev=*/1.0, 0);
  const std::vector<image_t> image_ids = reconstruction.RegImageIds();

  // Configure only the first 5 images with constant poses.
  BundleAdjustmentConfig subset_config;
  for (size_t i = 0; i < 5; ++i) {
    subset_config.AddImage(image_ids[i]);
    subset_config.SetConstantRigFromWorldPose(
        reconstruction.Image(image_ids[i]).FrameId());
  }
  auto ba_before = CreateDefaultCeresBundleAdjuster(
      BundleAdjustmentOptions(), subset_config, reconstruction);
  ASSERT_TRUE(ba_before->Solve()->IsSolutionUsable());
  const BACovarianceOptions deff_opts =
      MakeCovarianceOptions(BACovarianceOptions::VarianceScaling::SPATIAL_DEFF);
  const std::optional<BACovariance> cov_before =
      EstimateBACovariance(deff_opts, reconstruction, *ba_before);
  ASSERT_TRUE(cov_before.has_value());

  // Heavily perturb the unconfigured images 5..9 sharing the same camera;
  // subset DEFF must remain identical.
  for (size_t i = 5; i < image_ids.size(); ++i) {
    Image& outside_image = reconstruction.Image(image_ids[i]);
    for (Point2D& point2D : outside_image.Points2D()) {
      if (point2D.HasPoint3D()) {
        point2D.xy += Eigen::Vector2d(50.0, -50.0);
      }
    }
  }
  const std::optional<BACovariance> cov_after =
      EstimateBACovariance(deff_opts, reconstruction, *ba_before);
  ASSERT_TRUE(cov_after.has_value());
  EXPECT_NEAR(cov_after->SpatialDesignEffect(),
              cov_before->SpatialDesignEffect(),
              1e-12);

  // Mixed constant + variable poses: images 0..3 constant with strong tile
  // bias, images 4..7 variable with i.i.d. noise (with fixed points so the tile
  // bias stays on images 0..3). Constant images must contribute to DEFF.
  Reconstruction mixed_recon = SynthesizeSingleCameraReconstruction(
      /*num_frames=*/10, /*num_points3D=*/200, /*point2D_stddev=*/1.0, 1);
  const std::vector<image_t> mixed_ids = mixed_recon.RegImageIds();
  AddSpatialTileBias(
      mixed_recon,
      std::vector<image_t>(mixed_ids.begin(), mixed_ids.begin() + 4),
      /*tile_stddev=*/4.0,
      /*iid_stddev=*/0.0,
      /*seed=*/7);

  BundleAdjustmentConfig mixed_config;
  for (size_t i = 0; i < 8; ++i) {
    mixed_config.AddImage(mixed_ids[i]);
    if (i < 4) {
      mixed_config.SetConstantRigFromWorldPose(
          mixed_recon.Image(mixed_ids[i]).FrameId());
    }
  }
  BundleAdjustmentOptions mixed_ba_opts;
  mixed_ba_opts.refine_points3D = false;
  auto mixed_ba = CreateDefaultCeresBundleAdjuster(
      mixed_ba_opts, mixed_config, mixed_recon);
  ASSERT_TRUE(mixed_ba->Solve()->IsSolutionUsable());
  const std::optional<BACovariance> mixed_cov =
      EstimateBACovariance(deff_opts, mixed_recon, *mixed_ba);
  ASSERT_TRUE(mixed_cov.has_value());
  EXPECT_GT(mixed_cov->SpatialDesignEffect(), 5.0);
}

TEST(EstimateBACovariance,
     SpatialDesignEffectExcludesSparseImagesAndHandlesAllConstant) {
  Reconstruction recon_base = SynthesizeSingleCameraReconstruction(
      /*num_frames=*/7, /*num_points3D=*/200, /*point2D_stddev=*/1.0, 2);
  const std::vector<image_t> ids = recon_base.RegImageIds();
  AddSpatialTileBias(recon_base,
                     std::vector<image_t>(ids.begin(), ids.begin() + 6),
                     /*tile_stddev=*/2.0,
                     /*iid_stddev=*/0.0,
                     /*seed=*/99);

  // Keep only 3 observations on image 6 (< kMinObsPerSpatialTile in every
  // tile) and give them huge residuals; image 6 will have image_tile_dof == 0
  // and must not alter DEFF when added with constant pose and fixed points.
  const image_t sparse_id = ids[6];
  Image& sparse_img = recon_base.Image(sparse_id);
  int kept = 0;
  for (point2D_t p2d_idx = 0; p2d_idx < sparse_img.NumPoints2D(); ++p2d_idx) {
    if (!sparse_img.Point2D(p2d_idx).HasPoint3D()) {
      continue;
    }
    if (kept < 3) {
      sparse_img.Point2D(p2d_idx).xy += Eigen::Vector2d(100.0, -100.0);
      ++kept;
    } else {
      recon_base.DeleteObservation(sparse_id, p2d_idx);
    }
  }

  BundleAdjustmentOptions ba_opts;
  ba_opts.refine_points3D = false;
  ba_opts.refine_focal_length = false;
  ba_opts.refine_extra_params = false;
  ba_opts.refine_rig_from_world = false;

  BundleAdjustmentConfig cfg_without;
  for (size_t i = 0; i < 6; ++i) {
    cfg_without.AddImage(ids[i]);
  }
  auto ba_without =
      CreateDefaultCeresBundleAdjuster(ba_opts, cfg_without, recon_base);

  BundleAdjustmentConfig cfg_with = cfg_without;
  cfg_with.AddImage(sparse_id);
  auto ba_with =
      CreateDefaultCeresBundleAdjuster(ba_opts, cfg_with, recon_base);

  const BACovarianceOptions deff_opts =
      MakeCovarianceOptions(BACovarianceOptions::VarianceScaling::SPATIAL_DEFF);
  const std::optional<BACovariance> cov_without =
      EstimateBACovariance(deff_opts, recon_base, *ba_without);
  const std::optional<BACovariance> cov_with =
      EstimateBACovariance(deff_opts, recon_base, *ba_with);
  ASSERT_TRUE(cov_without.has_value());
  ASSERT_TRUE(cov_with.has_value());
  EXPECT_GT(cov_without->SpatialDesignEffect(), 4.0);
  EXPECT_NEAR(cov_with->SpatialDesignEffect(),
              cov_without->SpatialDesignEffect(),
              1e-12);

  // Also verifies the all-constant / empty parameter_blocks path in
  // ComputeSchurComplement for both Params::ALL and Params::POINTS.
  const std::optional<BACovariance> cov_pts = EstimateBACovariance(
      MakeCovarianceOptions(BACovarianceOptions::VarianceScaling::SPATIAL_DEFF,
                            BACovarianceOptions::Params::POINTS),
      recon_base,
      *ba_without);
  ASSERT_TRUE(cov_pts.has_value());
  EXPECT_NEAR(cov_pts->PosteriorVarianceFactor(),
              cov_without->PosteriorVarianceFactor(),
              1e-12);
  EXPECT_NEAR(cov_pts->SpatialDesignEffect(),
              cov_without->SpatialDesignEffect(),
              1e-12);
}

TEST(EstimateBACovariance, SpatialDesignEffectWrapsEquirectangularSeam) {
  SetPRNGSeed(321);
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 1;
  dataset_options.num_frames_per_rig = 10;
  dataset_options.num_points3D = 240;
  dataset_options.camera_width = 1000;
  dataset_options.camera_height = 500;
  dataset_options.camera_model_id = CameraModelId::kEquirectangular;
  dataset_options.camera_params = {1000.0, 500.0};
  Reconstruction eq_recon;
  SynthesizeDataset(dataset_options, &eq_recon);

  // Spread points around the cameras to populate all 3x2 tiles and place a
  // handful near image 0's ±pi horizontal seam (-z in camera frame) so noise
  // wraps observations across x = 0 / x = width.
  const Rigid3d world_from_cam0 =
      Inverse(eq_recon.Image(eq_recon.RegImageIds()[0]).CamFromWorld());
  const FlatHashSet<point3D_t> point3D_id_set = eq_recon.Point3DIds();
  std::vector<point3D_t> point3D_ids(point3D_id_set.begin(),
                                     point3D_id_set.end());
  std::sort(point3D_ids.begin(), point3D_ids.end());
  for (size_t i = 0; i < point3D_ids.size(); ++i) {
    Point3D& point3D = eq_recon.Point3D(point3D_ids[i]);
    if (i < 12) {
      point3D.xyz = world_from_cam0 *
                    Eigen::Vector3d(0.0, RandomUniformReal(-5.0, 5.0), -10.0);
    } else {
      point3D.xyz *= 10.0;
    }
  }

  int num_seam_crossings = 0;
  for (const image_t image_id : eq_recon.RegImageIds()) {
    Image& image = eq_recon.Image(image_id);
    const Camera& camera = *image.CameraPtr();
    const Rigid3d cam_from_world = image.CamFromWorld();
    for (Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D()) {
        continue;
      }
      const Eigen::Vector2d proj = *camera.ImgFromCam(
          cam_from_world * eq_recon.Point3D(point2D.point3D_id).xyz);
      Eigen::Vector2d xy = proj + Eigen::Vector2d(RandomGaussian(0.0, 1.0),
                                                  RandomGaussian(0.0, 1.0));
      xy.x() -= camera.width * std::floor(xy.x() / camera.width);
      if (std::abs(proj.x() - xy.x()) > 0.5 * camera.width) {
        ++num_seam_crossings;
      }
      point2D.xy = xy;
    }
  }
  ASSERT_GT(num_seam_crossings, 0);

  BundleAdjustmentOptions eq_ba_opts;
  eq_ba_opts.refine_focal_length = false;
  eq_ba_opts.refine_principal_point = false;
  eq_ba_opts.refine_extra_params = false;
  auto eq_ba = CreateDefaultCeresBundleAdjuster(
      eq_ba_opts, CreateAllImagesBAConfig(eq_recon), eq_recon);
  ASSERT_TRUE(eq_ba->Solve()->IsSolutionUsable());

  const std::optional<BACovariance> eq_cov = EstimateBACovariance(
      MakeCovarianceOptions(BACovarianceOptions::VarianceScaling::SPATIAL_DEFF),
      eq_recon,
      *eq_ba);
  ASSERT_TRUE(eq_cov.has_value());
  EXPECT_NEAR(eq_cov->PosteriorVarianceFactor(), 1.0, 0.35);
  EXPECT_LT(eq_cov->SpatialDesignEffect(), 1.6);
}

TEST(EstimateBACovariance,
     VarianceScalingFromProblemCustomPosesAndPointConfig) {
  const Reconstruction base_recon = SynthesizeSingleCameraReconstruction(
      /*num_frames=*/8, /*num_points3D=*/200, /*point2D_stddev=*/1.2, 4);
  const BACovarianceOptions deff_opts =
      MakeCovarianceOptions(BACovarianceOptions::VarianceScaling::SPATIAL_DEFF);

  // 1. Free gauge and minimally fixed gauge: EstimateBACovarianceFromProblem
  //    and experimental_custom_poses match EstimateBACovariance exactly
  //    (including preserving the constant-pose gauge image).
  {
    Reconstruction recon = base_recon;
    const std::vector<image_t> ids = recon.RegImageIds();
    auto ceres_ba = CreateDefaultCeresBundleAdjuster(
        BundleAdjustmentOptions(), CreateAllImagesBAConfig(recon), recon);
    ASSERT_TRUE(ceres_ba->Solve()->IsSolutionUsable());

    // Free-gauge: experimental_custom_poses via EstimateBACovarianceFromProblem
    // matches standard EstimateBACovariance.
    const std::optional<BACovariance> free_cov =
        EstimateBACovariance(deff_opts, recon, *ceres_ba);
    ASSERT_TRUE(free_cov.has_value());
    BACovarianceOptions free_custom_opts = deff_opts;
    free_custom_opts.experimental_custom_poses =
        internal::GetPoseParams(recon, *ceres_ba->Problem());
    ASSERT_FALSE(free_custom_opts.experimental_custom_poses.empty());
    const std::optional<BACovariance> custom_prob_cov =
        EstimateBACovarianceFromProblem(
            free_custom_opts, recon, *ceres_ba->Problem());
    ASSERT_TRUE(custom_prob_cov.has_value());
    EXPECT_NEAR(custom_prob_cov->PosteriorVarianceFactor(),
                free_cov->PosteriorVarianceFactor(),
                1e-12);
    EXPECT_NEAR(custom_prob_cov->SpatialDesignEffect(),
                free_cov->SpatialDesignEffect(),
                1e-12);

    // Minimally fixed gauge: EstimateBACovarianceFromProblem and
    // EstimateBACovariance + experimental_custom_poses both preserve the
    // constant-pose gauge image and match EstimateBACovariance exactly.
    FixGaugeMinimally(
        recon, *ceres_ba->Problem(), ids[0], ids[1], /*pin_ref_pose=*/true);

    const std::optional<BACovariance> cov_ba =
        EstimateBACovariance(deff_opts, recon, *ceres_ba);
    const std::optional<BACovariance> cov_prob =
        EstimateBACovarianceFromProblem(deff_opts, recon, *ceres_ba->Problem());
    ASSERT_TRUE(cov_ba.has_value());
    ASSERT_TRUE(cov_prob.has_value());
    EXPECT_NEAR(cov_prob->PosteriorVarianceFactor(),
                cov_ba->PosteriorVarianceFactor(),
                1e-12);
    EXPECT_NEAR(
        cov_prob->SpatialDesignEffect(), cov_ba->SpatialDesignEffect(), 1e-12);

    BACovarianceOptions custom_opts = deff_opts;
    custom_opts.experimental_custom_poses =
        internal::GetPoseParams(recon, *ceres_ba->Problem());
    ASSERT_EQ(custom_opts.experimental_custom_poses.size(),
              recon.NumRegImages() - 1);
    const std::optional<BACovariance> custom_ba_cov =
        EstimateBACovariance(custom_opts, recon, *ceres_ba);
    ASSERT_TRUE(custom_ba_cov.has_value());
    EXPECT_NEAR(custom_ba_cov->PosteriorVarianceFactor(),
                cov_ba->PosteriorVarianceFactor(),
                1e-12);
    EXPECT_NEAR(custom_ba_cov->SpatialDesignEffect(),
                cov_ba->SpatialDesignEffect(),
                1e-12);
  }

  // 2. EstimateBACovarianceFromProblem fallback when all poses are baked into
  //    constant-pose cost functors (no pose parameter blocks in problem).
  {
    Reconstruction recon = base_recon;
    auto ceres_ba = CreateDefaultCeresBundleAdjuster(
        BundleAdjustmentOptions(),
        CreateAllImagesBAConfig(recon, /*constant_poses=*/true),
        recon);
    ASSERT_TRUE(ceres_ba->Solve()->IsSolutionUsable());

    const std::optional<BACovariance> cov_ba =
        EstimateBACovariance(deff_opts, recon, *ceres_ba);
    const std::optional<BACovariance> cov_prob =
        EstimateBACovarianceFromProblem(deff_opts, recon, *ceres_ba->Problem());
    ASSERT_TRUE(cov_ba.has_value());
    ASSERT_TRUE(cov_prob.has_value());
    EXPECT_NEAR(
        cov_prob->SpatialDesignEffect(), cov_ba->SpatialDesignEffect(), 1e-12);
  }

  // 3. Point-only BundleAdjustmentConfig (external track images via
  //    VariablePoints()) with min_track_length filtering and unconfigured
  //    points perturbed on external images.
  {
    Reconstruction recon = base_recon;
    std::vector<point3D_t> point_ids;
    point_ids.reserve(recon.NumPoints3D());
    for (const auto& [pt_id, pt] : recon.Points3D()) {
      point_ids.push_back(pt_id);
    }
    std::sort(point_ids.begin(), point_ids.end());

    // Trim the last 10 points to track length 2 and give them huge residuals so
    // min_track_length == 4 must exclude them from SPATIAL_DEFF.
    for (size_t i = point_ids.size() - 10; i < point_ids.size(); ++i) {
      const point3D_t pt_id = point_ids[i];
      const std::vector<TrackElement> elems =
          recon.Point3D(pt_id).track.Elements();
      for (size_t k = 2; k < elems.size(); ++k) {
        recon.DeleteObservation(elems[k].image_id, elems[k].point2D_idx);
      }
      for (const TrackElement& el : recon.Point3D(pt_id).track.Elements()) {
        recon.Image(el.image_id).Point2D(el.point2D_idx).xy +=
            Eigen::Vector2d(200.0, -200.0);
      }
    }

    // Add points 0..179 and the 10 short-track points 190..199 via
    // VariablePoints() (testing external-track images and min_track_length
    // filtering), while leaving points 180..189 unconfigured and heavily
    // perturbed on those external images.
    BundleAdjustmentConfig pt_config;
    for (size_t i = 0; i < 180; ++i) {
      pt_config.AddVariablePoint(point_ids[i]);
    }
    for (size_t i = 190; i < point_ids.size(); ++i) {
      pt_config.AddVariablePoint(point_ids[i]);
    }
    for (size_t i = 180; i < 190; ++i) {
      for (const TrackElement& el :
           recon.Point3D(point_ids[i]).track.Elements()) {
        recon.Image(el.image_id).Point2D(el.point2D_idx).xy +=
            Eigen::Vector2d(200.0, -200.0);
      }
    }

    BundleAdjustmentOptions ba_opts;
    ba_opts.min_track_length = 4;
    auto ceres_ba = CreateDefaultCeresBundleAdjuster(ba_opts, pt_config, recon);
    ASSERT_TRUE(ceres_ba->Solve()->IsSolutionUsable());

    const std::optional<BACovariance> cov =
        EstimateBACovariance(deff_opts, recon, *ceres_ba);
    ASSERT_TRUE(cov.has_value());
    EXPECT_NEAR(cov->PosteriorVarianceFactor(), 1.2 * 1.2, 0.35);
    EXPECT_LT(cov->SpatialDesignEffect(), 1.6);
  }
}

}  // namespace
}  // namespace colmap
