// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/covariance.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/math/random.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/eigen_matchers.h"

#include <algorithm>

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
          test_options.fixed_cam_poses = true;
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

MonteCarloProblem CreateMonteCarloProblem(int num_constant_points) {
  MonteCarloProblem problem;
  SetPRNGSeed(0);
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 6;
  synthetic_dataset_options.num_points3D = 60;
  SynthesizeDataset(synthetic_dataset_options, &problem.gt_reconstruction);
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
  BundleAdjustmentOptions options;
  options.ceres->solver_options.max_num_iterations = 200;
  options.ceres->solver_options.gradient_tolerance = 1e-12;
  std::unique_ptr<CeresBundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  THROW_CHECK_EQ(bundle_adjuster->Solve()->termination_type,
                 BundleAdjustmentTerminationType::CONVERGENCE);
  return bundle_adjuster;
}

constexpr double kMonteCarloPoint2DStddev = 0.5;
constexpr int kNumMonteCarloSamples = 300;

TEST(BACovarianceMonteCarlo, PosesAndIntrinsics) {
  const MonteCarloProblem problem =
      CreateMonteCarloProblem(/*num_constant_points=*/3);
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

  std::vector<Eigen::VectorXd> pose_samples;
  std::vector<Eigen::VectorXd> intrinsics_samples;
  for (int i = 0; i < kNumMonteCarloSamples; ++i) {
    Reconstruction reconstruction = gt_reconstruction;
    SetPRNGSeed(i + 1);
    SyntheticNoiseOptions synthetic_noise_options;
    synthetic_noise_options.point2D_stddev = kMonteCarloPoint2DStddev;
    SynthesizeNoise(synthetic_noise_options, &reconstruction);
    SolveMonteCarloProblem(problem, reconstruction);

    pose_samples.push_back(
        PoseDeviation(gt_reconstruction, reconstruction, problem.image_ids));

    const Camera& camera = reconstruction.Camera(gt_camera.camera_id);
    intrinsics_samples.push_back(
        Eigen::Vector2d(camera.params[0] - gt_camera.params[0],
                        camera.params[3] - gt_camera.params[3]));
  }

  EXPECT_THAT(pose_samples, SamplesFollowCov(pose_cov));
  EXPECT_THAT(intrinsics_samples, SamplesFollowCov(variance * *intrinsics_cov));
}

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
TEST(EstimateBACovariance, IllConditionedAndDegeneratePoints) {
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
  FixGaugeMinimally(reconstruction,
                    ceres_problem,
                    image_ids[0],
                    image_ids[1],
                    /*pin_ref_pose=*/true);
  // Keep the single-residual point variable to test k < tangent_size SVD, and
  // add a zero-residual variable point to test the empty-rows path.
  ceres_problem.SetParameterBlockVariable(
      reconstruction.Point3D(single_residual_pt).xyz.data());
  const point3D_t zero_residual_pt =
      reconstruction.AddPoint3D(Eigen::Vector3d(1.0, 2.0, 3.0), Track());
  ceres_problem.AddParameterBlock(
      reconstruction.Point3D(zero_residual_pt).xyz.data(), 3);

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
}

}  // namespace
}  // namespace colmap
