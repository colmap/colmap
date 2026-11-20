// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/covariance.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <array>
#include <tuple>

#include <ceres/evaluation_callback.h>
#include <ceres/normal_prior.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(BACovariance, PointOnlyRejectsNonFiniteCovariance) {
  for (const double constrained : {0.0, 1.0}) {
    SCOPED_TRACE(constrained);
    Reconstruction reconstruction;
    const point3D_t point_id =
        reconstruction.AddPoint3D(Eigen::Vector3d(1, 2, 3), Track());
    Eigen::Matrix3d weight = Eigen::Matrix3d::Zero();
    weight(0, 0) = constrained;
    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::NormalPrior(weight, Eigen::Vector3d::Zero()),
        nullptr,
        reconstruction.Point3D(point_id).xyz.data());

    BACovarianceOptions options;
    options.params = BACovarianceOptions::Params::POINTS;
    options.damping = 0;
    EXPECT_FALSE(
        EstimateBACovarianceFromProblem(options, reconstruction, problem)
            .has_value());

    options.damping = BACovarianceOptions().damping;
    const auto cov =
        EstimateBACovarianceFromProblem(options, reconstruction, problem);
    ASSERT_TRUE(cov.has_value());
    const auto point_cov = cov->GetPointCov(point_id);
    ASSERT_TRUE(point_cov.has_value());
    ASSERT_EQ(point_cov->rows(), 3);
    ASSERT_EQ(point_cov->cols(), 3);
    const Eigen::Vector3d hessian_diag(
        constrained * constrained + options.damping,
        options.damping,
        options.damping);
    EXPECT_TRUE(point_cov->allFinite());
    EXPECT_TRUE((*point_cov * hessian_diag.asDiagonal())
                    .isApprox(Eigen::Matrix3d::Identity(), 1e-12));
  }
}

TEST(BACovariance, PointOnlyRejectsInvalidHessian) {
  for (const int tangent_size : {1, 2}) {
    for (const bool well_conditioned : {true, false}) {
      SCOPED_TRACE(tangent_size);
      SCOPED_TRACE(well_conditioned);
      Reconstruction reconstruction;
      const point3D_t point_id =
          reconstruction.AddPoint3D(Eigen::Vector3d::Zero(), Track());
      double* point = reconstruction.Point3D(point_id).xyz.data();
      Eigen::Matrix3d weight = Eigen::Matrix3d::Zero();
      if (tangent_size == 1) {
        weight(0, 0) = well_conditioned ? 1.0 : 1e155;
      } else {
        const double scale = well_conditioned ? 1e-8 : 1.0;
        weight.topLeftCorner<2, 2>() << 370392643.9011732 * scale,
            235456757.2678747 * scale, 1.0, 0.0;
      }
      ceres::Problem problem;
      problem.AddResidualBlock(
          new ceres::NormalPrior(weight, Eigen::Vector3d::Zero()),
          nullptr,
          point);
      const std::vector<int> fixed =
          tangent_size == 1 ? std::vector<int>{1, 2} : std::vector<int>{2};
      SetManifold(&problem, point, CreateSubsetManifold(3, fixed));
      BACovarianceOptions options;
      options.params = BACovarianceOptions::Params::POINTS;
      const auto covariance =
          EstimateBACovarianceFromProblem(options, reconstruction, problem);
      if (well_conditioned) {
        ASSERT_TRUE(covariance.has_value());
      }
      if (tangent_size == 1 && !well_conditioned) {
        EXPECT_FALSE(covariance.has_value());
      }
      if (covariance.has_value()) {
        const auto point_cov = covariance->GetPointCov(point_id);
        ASSERT_TRUE(point_cov.has_value());
        EXPECT_TRUE(point_cov->allFinite());
        EXPECT_GT(point_cov->diagonal().minCoeff(), 0.0);
        if (well_conditioned) {
          const Eigen::MatrixXd J = weight.leftCols(tangent_size);
          const Eigen::MatrixXd expected =
              (J.transpose() * J +
               options.damping *
                   Eigen::MatrixXd::Identity(tangent_size, tangent_size))
                  .inverse();
          EXPECT_TRUE(point_cov->isApprox(expected, 1e-12));
        }
      }
    }
  }
}

TEST(BACovariance, RejectsNonRepresentableCovariance) {
  for (const int size : {1, 3, 80}) {
    SCOPED_TRACE(size);
    Reconstruction reconstruction;
    Eigen::VectorXd other = Eigen::VectorXd::Zero(size);
    Eigen::MatrixXd weight = Eigen::MatrixXd::Identity(size, size);
    if (size == 1) {
      weight(0, 0) = 1e155;
    } else {
      weight.diagonal(1).setConstant(100);
    }
    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::NormalPrior(weight, Eigen::VectorXd::Zero(size)),
        nullptr,
        other.data());
    BACovarianceOptions options;
    options.params = BACovarianceOptions::Params::ALL;
    const auto covariance =
        EstimateBACovarianceFromProblem(options, reconstruction, problem);
    if (size == 1) {
      // A finite Jacobian can overflow when forming the normal equations.
      EXPECT_FALSE(covariance.has_value());
    } else {
      if (size == 3) {
        ASSERT_TRUE(covariance.has_value());
      }
      if (covariance.has_value()) {
        const auto other_cov = covariance->GetOtherParamsCov(other.data());
        ASSERT_TRUE(other_cov.has_value());
        // The longer chain can overflow covariance even with a finite factor.
        EXPECT_TRUE(other_cov->allFinite());
        if (size == 3) {
          const Eigen::MatrixXd inverse = weight.inverse();
          EXPECT_TRUE(
              other_cov->isApprox(inverse * inverse.transpose(), 1e-12));
        }
      }
    }
  }
}

struct CoupledSchurCovarianceCost {
  double a;
  double b;
  int retained_index = 0;

  template <typename T>
  bool operator()(const T* retained, const T* eliminated, T* residual) const {
    residual[0] = T(a) * retained[retained_index] + T(b) * eliminated[0];
    residual[1] = retained[retained_index];
    return true;
  }
};

TEST(BACovariance, NuisanceEliminationDoesNotHideOverflow) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions data;
  data.num_rigs = 1;
  data.num_cameras_per_rig = 1;
  data.num_frames_per_rig = 1;
  data.num_points3D = 1;
  data.num_points2D_without_point3D = 0;
  SynthesizeDataset(data, &reconstruction);
  const image_t image_id = reconstruction.RegImageIds().front();
  auto& pose = reconstruction.Image(image_id).FramePtr()->RigFromWorld();
  pose = Rigid3d();
  for (const double scale : {1.0, 1e155}) {
    SCOPED_TRACE(scale);
    double nuisance = 0;
    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<CoupledSchurCovarianceCost, 2, 7, 1>(
            new CoupledSchurCovarianceCost{1.0, scale, 4}),
        nullptr,
        pose.params.data(),
        &nuisance);
    SetManifold(&problem,
                pose.params.data(),
                CreateSubsetManifold(7, {0, 1, 2, 3, 5, 6}));
    for (const auto mode : {BACovarianceOptions::Params::ALL,
                            BACovarianceOptions::Params::POSES}) {
      SCOPED_TRACE(static_cast<int>(mode));
      BACovarianceOptions options;
      options.params = mode;
      const auto covariance =
          EstimateBACovarianceFromProblem(options, reconstruction, problem);
      if (scale == 1.0) {
        ASSERT_TRUE(covariance.has_value());
        const auto pose_cov = covariance->GetCamCovFromWorld(image_id);
        ASSERT_TRUE(pose_cov.has_value());
        const double damping =
            mode == BACovarianceOptions::Params::POSES ? options.damping : 0.0;
        const double expected = 1.0 / (2.0 - 1.0 / (1.0 + damping));
        EXPECT_NEAR((*pose_cov)(0, 0), expected, 1e-12);
      } else {
        EXPECT_FALSE(covariance.has_value());
      }
    }
  }
}

TEST(BACovariance, SchurCancellationDoesNotReturnNonFiniteCovariance) {
  for (const double scale : {1e-8, 1.0}) {
    SCOPED_TRACE(scale);
    Reconstruction reconstruction;
    const point3D_t point_id =
        reconstruction.AddPoint3D(Eigen::Vector3d::Zero(), Track());
    double* point = reconstruction.Point3D(point_id).xyz.data();
    double other = 0;
    const double a = 612342538.330063 * scale;
    const double b = 578478850.648649 * scale;
    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<CoupledSchurCovarianceCost, 2, 1, 3>(
            new CoupledSchurCovarianceCost{a, b}),
        nullptr,
        &other,
        point);
    SetManifold(&problem, point, CreateSubsetManifold(3, {1, 2}));
    BACovarianceOptions options;
    options.params = BACovarianceOptions::Params::ALL;
    const auto covariance =
        EstimateBACovarianceFromProblem(options, reconstruction, problem);
    // Cancellation can make this full-rank problem's Schur pivot negative.
    // Allow rejection or a finite result across different rounding behavior.
    if (scale == 1e-8) {
      ASSERT_TRUE(covariance.has_value());
    }
    if (covariance.has_value()) {
      const auto other_cov = covariance->GetOtherParamsCov(&other);
      ASSERT_TRUE(other_cov.has_value());
      EXPECT_TRUE(other_cov->allFinite());
      if (scale == 1e-8) {
        const double expected =
            1.0 / (1.0 + a * a * options.damping / (b * b + options.damping));
        EXPECT_NEAR((*other_cov)(0, 0), expected, 1e-12);
      }
    }
  }
}

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

TEST(BACovariance, RelativePoseUsesLeftSE3TangentCoordinates) {
  const auto manifold = CreateProductManifold(CreateEigenQuaternionManifold(),
                                              CreateEuclideanManifold<3>());
  std::array<std::pair<Rigid3d, Rigid3d>, 3> poses;
  poses[1].second.translation() = Eigen::Vector3d(10, 0, 0);
  poses[2].first.rotation() =
      Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, 2, 3).normalized());
  poses[2].second.rotation() =
      Eigen::AngleAxisd(-0.7, Eigen::Vector3d(3, -2, 1).normalized());
  poses[2].first.translation() = Eigen::Vector3d(1, 2, 3);
  poses[2].second.translation() = Eigen::Vector3d(-2, 0.5, 4);

  for (size_t i = 0; i < poses.size(); ++i) {
    SCOPED_TRACE(i);
    const auto& [cam1_from_world, cam2_from_world] = poses[i];
    Eigen::Matrix<double, 12, 12> factor =
        Eigen::Matrix<double, 12, 12>::Identity();
    if (i == 2) {
      for (int row = 0; row < 12; ++row) {
        for (int col = 0; col < 12; ++col) {
          factor(row, col) += 0.03 * ((3 * row + 5 * col) % 7 - 3);
        }
      }
    }
    const BACovariance covariance({}, {{1, {0, 6}}, {2, {6, 6}}}, {}, factor);
    const auto actual =
        covariance.GetCam2CovFromCam1(1, cam1_from_world, 2, cam2_from_world);
    ASSERT_TRUE(actual.has_value());

    const Rigid3d cam2_from_cam1 = cam2_from_world * Inverse(cam1_from_world);
    Eigen::Matrix<double, 6, 12> J;
    constexpr double kStep = 1e-6;
    for (int parameter = 0; parameter < 12; ++parameter) {
      Eigen::Vector6d differences[2];
      for (int side = 0; side < 2; ++side) {
        Eigen::Vector6d delta = Eigen::Vector6d::Zero();
        delta(parameter % 6) = (side ? 1 : -1) * kStep;
        Rigid3d perturbed_cam1 = cam1_from_world;
        Rigid3d perturbed_cam2 = cam2_from_world;
        Rigid3d& perturbed = parameter < 6 ? perturbed_cam1 : perturbed_cam2;
        const Rigid3d& nominal =
            parameter < 6 ? cam1_from_world : cam2_from_world;
        ASSERT_TRUE(manifold->Plus(
            nominal.params.data(), delta.data(), perturbed.params.data()));
        const Rigid3d relative = perturbed_cam2 * Inverse(perturbed_cam1);
        const Rigid3d left_difference = relative * Inverse(cam2_from_cam1);
        const Eigen::AngleAxisd rotation_difference(left_difference.rotation());
        differences[side].head<3>() =
            rotation_difference.angle() * rotation_difference.axis();
        // At zero, this translation has the same derivative as the SE(3) log.
        differences[side].tail<3>() = left_difference.translation();
      }
      J.col(parameter) = (differences[1] - differences[0]) / (2 * kStep);
    }
    ExpectNearEigenMatrixXd(
        *actual, J * factor.transpose() * factor * J.transpose(), 1e-7);
    if (i == 1) {
      const Eigen::Vector6d expected_diagonal(8, 8, 8, 2, 802, 802);
      ExpectNearEigenMatrixXd(actual->diagonal(), expected_diagonal, 1e-10);
    }
    const auto identity_cov =
        covariance.GetCam2CovFromCam1(1, cam1_from_world, 1, cam1_from_world);
    ASSERT_TRUE(identity_cov.has_value());
    ExpectNearEigenMatrixXd(*identity_cov, Eigen::Matrix6d::Zero(), 1e-10);
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

class PointTangentBACovarianceTests
    : public ::testing::TestWithParam<
          std::tuple<BACovarianceOptions::Params, int>> {};

TEST_P(PointTangentBACovarianceTests, CompareWithCeres) {
  const auto [params, tangent_size] = GetParam();
  Reconstruction reconstruction;
  SyntheticDatasetOptions dataset;
  dataset.num_rigs = 1;
  dataset.num_cameras_per_rig = 1;
  dataset.num_frames_per_rig = 1;
  dataset.num_points3D = 2;
  dataset.num_points2D_without_point3D = 0;
  SynthesizeDataset(dataset, &reconstruction);
  const image_t image_id = reconstruction.RegImageIds().front();
  double* const pose =
      reconstruction.Frame(reconstruction.Image(image_id).FrameId())
          .RigFromWorld()
          .params.data();
  double other = 0.25;
  std::vector<int> fixed_coordinates;
  for (int i = tangent_size; i < 3; ++i) {
    fixed_coordinates.push_back(i);
  }

  ceres::Problem problem;
  std::vector<std::pair<const double*, const double*>> point_pairs;
  for (const auto& [point_id, _] : reconstruction.Points3D()) {
    double* const point = reconstruction.Point3D(point_id).xyz.data();
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<LinearCovarianceCost, 6, 7, 1, 3>(
            new LinearCovarianceCost),
        new ceres::ScaledLoss(
            nullptr, static_cast<double>(point_id), ceres::TAKE_OWNERSHIP),
        pose,
        &other,
        point);
    SetManifold(&problem, point, CreateSubsetManifold(3, fixed_coordinates));
    ASSERT_EQ(ParameterBlockTangentSize(problem, point), tangent_size);
    point_pairs.emplace_back(point, point);
  }
  ASSERT_EQ(point_pairs.size(), 2);
  ASSERT_EQ(problem.NumResiduals(), 12);
  SetManifold(&problem, pose, CreateSubsetManifold(7, {0, 1, 2, 3, 6}));

  BACovarianceOptions options;
  options.params = params;
  options.damping = 0;
  const auto covariance =
      EstimateBACovarianceFromProblem(options, reconstruction, problem);
  ASSERT_TRUE(covariance.has_value());
  if (params == BACovarianceOptions::Params::POSES_AND_POINTS) {
    // Pose covariance marginalizes both point blocks and the other parameter.
    ceres::Covariance oracle{ceres::Covariance::Options()};
    ASSERT_TRUE(oracle.Compute(std::vector<const double*>{pose}, &problem));
    Eigen::Matrix<double, 2, 2, Eigen::RowMajor> expected;
    ASSERT_TRUE(
        oracle.GetCovarianceBlockInTangentSpace(pose, pose, expected.data()));
    const auto pose_cov = covariance->GetCamCovFromWorld(image_id);
    ASSERT_TRUE(pose_cov.has_value());
    ExpectNearEigenMatrixXd(*pose_cov, expected, 1e-12);
  } else {
    EXPECT_FALSE(covariance->GetCamCovFromWorld(image_id).has_value());
  }
  // Point covariance is conditional even when pose covariance is requested.
  problem.SetParameterBlockConstant(pose);
  problem.SetParameterBlockConstant(&other);
  ceres::Covariance oracle{ceres::Covariance::Options()};
  ASSERT_TRUE(oracle.Compute(point_pairs, &problem));
  for (const auto& [point_id, point] : reconstruction.Points3D()) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        expected(tangent_size, tangent_size);
    ASSERT_TRUE(oracle.GetCovarianceBlockInTangentSpace(
        point.xyz.data(), point.xyz.data(), expected.data()));
    const auto point_cov = covariance->GetPointCov(point_id);
    ASSERT_TRUE(point_cov.has_value());
    ExpectNearEigenMatrixXd(*point_cov, expected, 1e-12);
  }
}

INSTANTIATE_TEST_SUITE_P(
    BACovarianceTests,
    PointTangentBACovarianceTests,
    ::testing::Combine(
        ::testing::Values(BACovarianceOptions::Params::POINTS,
                          BACovarianceOptions::Params::POSES_AND_POINTS),
        ::testing::Values(3, 2, 1)));

class OtherCovarianceRequestTests
    : public ::testing::TestWithParam<BACovarianceOptions::Params> {};

TEST_P(OtherCovarianceRequestTests, OnlyReturnsRequestedCovariance) {
  Eigen::Vector2d pose = Eigen::Vector2d::Zero();
  Eigen::Vector<double, 1> other = Eigen::Vector<double, 1>::Zero();
  ceres::Problem problem;
  problem.AddResidualBlock(new ceres::NormalPrior(Eigen::Matrix2d::Identity(),
                                                  Eigen::Vector2d::Zero()),
                           nullptr,
                           pose.data());
  problem.AddResidualBlock(
      new ceres::NormalPrior(Eigen::Matrix<double, 1, 1>::Identity(), other),
      nullptr,
      other.data());
  BACovarianceOptions options;
  options.params = GetParam();
  options.damping = 0;
  options.experimental_custom_poses = {{1, pose.data()}};
  const auto covariance =
      EstimateBACovarianceFromProblem(options, Reconstruction(), problem);
  ASSERT_TRUE(covariance.has_value());
  const auto pose_cov = covariance->GetCamCovFromWorld(1);
  ASSERT_TRUE(pose_cov.has_value());
  ExpectNearEigenMatrixXd(*pose_cov, Eigen::Matrix2d::Identity(), 1e-12);
  const auto other_cov = covariance->GetOtherParamsCov(other.data());
  if (options.params == BACovarianceOptions::Params::ALL) {
    ASSERT_TRUE(other_cov.has_value());
    ExpectNearEigenMatrixXd(
        *other_cov, Eigen::Matrix<double, 1, 1>::Identity(), 1e-12);
  } else {
    EXPECT_FALSE(other_cov.has_value());
  }
}

INSTANTIATE_TEST_SUITE_P(
    BACovarianceTests,
    OtherCovarianceRequestTests,
    ::testing::Values(BACovarianceOptions::Params::POSES,
                      BACovarianceOptions::Params::POSES_AND_POINTS,
                      BACovarianceOptions::Params::ALL));

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
