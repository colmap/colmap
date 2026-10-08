// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/rotation_averaging_statistics.h"

#include "colmap/estimators/rotation_averaging.h"
#include "colmap/estimators/rotation_averaging_ceres.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <algorithm>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

Eigen::Quaterniond ExpMap(const Eigen::Vector3d& delta) {
  const double angle = delta.norm();
  if (angle < 1e-12) {
    return Eigen::Quaterniond::Identity();
  }
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, delta / angle));
}

Eigen::Vector3d RandomGaussianVector() {
  return Eigen::Vector3d(RandomGaussian(0.0, 1.0),
                         RandomGaussian(0.0, 1.0),
                         RandomGaussian(0.0, 1.0));
}

Eigen::Quaterniond RandomRotation() {
  return ExpMap(M_PI / 2 * RandomGaussianVector());
}

// Random covariance with standard deviations in [min_sigma_deg,
// max_sigma_deg] along random axes.
Eigen::Matrix3d RandomCovariance(double min_sigma_deg, double max_sigma_deg) {
  const Eigen::Matrix3d axes = RandomRotation().toRotationMatrix();
  Eigen::Vector3d variances;
  for (int i = 0; i < 3; ++i) {
    const double sigma =
        DegToRad(RandomUniformReal(min_sigma_deg, max_sigma_deg));
    variances(i) = sigma * sigma;
  }
  return axes * variances.asDiagonal() * axes.transpose();
}

Eigen::Matrix3d IsotropicCovariance(double sigma_deg) {
  return DegToRad(sigma_deg) * DegToRad(sigma_deg) *
         Eigen::Matrix3d::Identity();
}

// Images with trivial rigs and random rotations, without poses.
Reconstruction CreateReconstruction(
    int num_images, std::vector<Eigen::Quaterniond>& rotations) {
  Reconstruction reconstruction;
  const Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 10, 10, 5);
  reconstruction.AddCameraWithTrivialRig(camera);
  rotations.clear();
  for (int i = 0; i < num_images; ++i) {
    Image image;
    image.SetImageId(i + 1);
    image.SetCameraId(camera.camera_id);
    reconstruction.AddImageWithTrivialFrame(image);
    rotations.push_back(RandomRotation());
  }
  return reconstruction;
}

// Adds an edge whose relative rotation is perturbed by noise drawn from the
// given covariance (right perturbation), offset by an optional bias.
void AddNoisyEdge(const std::vector<Eigen::Quaterniond>& rotations,
                  image_t image_id1,
                  image_t image_id2,
                  const Eigen::Matrix3d& cov,
                  PoseGraph& pose_graph,
                  const Eigen::Vector3d& bias = Eigen::Vector3d::Zero()) {
  const Eigen::Quaterniond cam2_from_cam1 =
      rotations[image_id2 - 1] * rotations[image_id1 - 1].inverse();
  const Eigen::Vector3d noise =
      cov.llt().matrixL() * RandomGaussianVector() + bias;
  PoseGraph::Edge edge(
      Rigid3d(cam2_from_cam1 * ExpMap(noise), Eigen::Vector3d::Zero()));
  edge.cam2_from_cam1_rotation_cov = cov;
  edge.num_matches = 100;
  pose_graph.AddEdge(image_id1, image_id2, std::move(edge));
}

// Connects each image to the next one (ring) and to random other images.
std::vector<std::pair<image_t, image_t>> RandomGraphPairs(int num_images,
                                                          int num_neighbors) {
  std::vector<std::pair<image_t, image_t>> pairs;
  FlatHashSet<image_pair_t> pair_ids;
  const auto add_pair = [&](image_t image_id1, image_t image_id2) {
    if (image_id1 != image_id2 &&
        pair_ids.insert(ImagePairToPairId(image_id1, image_id2)).second) {
      pairs.emplace_back(image_id1, image_id2);
    }
  };
  for (int i = 0; i < num_images; ++i) {
    add_pair(i + 1, (i + 1) % num_images + 1);
    for (int j = 0; j < num_neighbors; ++j) {
      add_pair(i + 1, RandomUniformInteger(1, num_images));
    }
  }
  return pairs;
}

RotationEstimatorOptions CreateCovarianceOptions() {
  RotationEstimatorOptions options;
  options.backend = RotationAveragingBackend::CERES;
  options.reweighting = RotationAveragingReweighting::COVARIANCE;
  options.covariance_sigma_floor_deg = 0;
  options.rotation_statistics.estimate_variance_factor = false;
  options.num_threads = 1;
  options.ceres->solver_options.num_threads = 1;
  options.ceres->solver_options.function_tolerance = 1e-12;
  options.ceres->solver_options.gradient_tolerance = 1e-12;
  options.ceres->solver_options.parameter_tolerance = 1e-12;
  return options;
}

void SolveRotationAveraging(const RotationEstimatorOptions& options,
                            const PoseGraph& pose_graph,
                            Reconstruction& reconstruction) {
  CeresRotationAverager averager(options, pose_graph, reconstruction);
  ASSERT_TRUE(averager.Solve().IsSolutionUsable());
}

TEST(EstimateRotationAveragingStatistics, CalibratedWithoutOutliers) {
  SetPRNGSeed(0);
  RotationEstimatorOptions options = CreateCovarianceOptions();
  // Without robust loss, the solution is the maximum likelihood estimate, such
  // that the statistics exactly follow the chi-squared distribution.
  options.ceres->loss_function_type = CeresLossFunctionType::TRIVIAL;
  options.ceres->max_num_warm_start_iterations = 0;

  constexpr int kNumTrials = 20;
  constexpr int kNumImages = 30;
  constexpr double kSignificance = 0.01;
  std::vector<double> statistics;
  int num_rejected = 0;
  int num_tests = 0;
  std::vector<double> variance_factors;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    std::vector<Eigen::Quaterniond> rotations;
    Reconstruction reconstruction = CreateReconstruction(kNumImages, rotations);
    PoseGraph pose_graph;
    for (const auto& [image_id1, image_id2] :
         RandomGraphPairs(kNumImages, /*num_neighbors=*/3)) {
      AddNoisyEdge(rotations,
                   image_id1,
                   image_id2,
                   RandomCovariance(0.5, 3),
                   pose_graph);
    }
    SolveRotationAveraging(options, pose_graph, reconstruction);

    const std::optional<RotationAveragingStatistics> result =
        EstimateRotationAveragingStatistics(
            options, pose_graph, reconstruction);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->variance_factor, 1.0);
    EXPECT_EQ(result->edges.size(), pose_graph.NumEdges());
    for (const auto& [pair_id, edge] : result->edges) {
      EXPECT_GE(edge.min_redundancy, 0);
      EXPECT_LE(edge.min_redundancy, 1);
      if (edge.num_dofs == 3) {
        statistics.push_back(edge.statistic);
      }
      if (edge.num_dofs > 0) {
        ++num_tests;
        num_rejected += edge.p_value < kSignificance;
      }
    }

    RotationEstimatorOptions variance_factor_options = options;
    variance_factor_options.rotation_statistics.estimate_variance_factor = true;
    variance_factors.push_back(
        EstimateRotationAveragingStatistics(
            variance_factor_options, pose_graph, reconstruction)
            ->variance_factor);
  }

  // The median of chi2(3) is 2.366.
  EXPECT_NEAR(Median(statistics), 2.366, 0.15);
  const double rejection_rate = static_cast<double>(num_rejected) / num_tests;
  EXPECT_GT(rejection_rate, 0.5 * kSignificance);
  EXPECT_LT(rejection_rate, 2 * kSignificance);
  EXPECT_NEAR(Median(variance_factors), 1.0, 0.15);
}

TEST(EstimateRotationAveragingStatistics, EstimatesVarianceFactor) {
  SetPRNGSeed(1);
  RotationEstimatorOptions options = CreateCovarianceOptions();
  options.rotation_statistics.estimate_variance_factor = true;
  constexpr int kNumImages = 50;
  constexpr double kCovarianceScale = 4;
  std::vector<Eigen::Quaterniond> rotations;
  Reconstruction reconstruction = CreateReconstruction(kNumImages, rotations);
  PoseGraph pose_graph;
  for (const auto& [image_id1, image_id2] :
       RandomGraphPairs(kNumImages, /*num_neighbors=*/4)) {
    AddNoisyEdge(
        rotations, image_id1, image_id2, IsotropicCovariance(1), pose_graph);
  }
  // The covariances overestimate the noise.
  for (auto& [pair_id, edge] : pose_graph.Edges()) {
    *edge.cam2_from_cam1_rotation_cov *= kCovarianceScale;
  }
  SolveRotationAveraging(options, pose_graph, reconstruction);

  const std::optional<RotationAveragingStatistics> result =
      EstimateRotationAveragingStatistics(options, pose_graph, reconstruction);
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->variance_factor, 1 / kCovarianceScale, 0.1);
  std::vector<double> statistics;
  for (const auto& [pair_id, edge] : result->edges) {
    if (edge.num_dofs == 3) {
      statistics.push_back(edge.statistic);
    }
  }
  EXPECT_NEAR(Median(statistics), 2.365973884375338, 1e-9);
}

TEST(EstimateRotationAveragingStatistics, DetectsSmallBiasOnCertainEdges) {
  SetPRNGSeed(2);
  RotationEstimatorOptions options = CreateCovarianceOptions();
  constexpr int kNumImages = 20;
  std::vector<Eigen::Quaterniond> rotations;
  Reconstruction reconstruction = CreateReconstruction(kNumImages, rotations);
  PoseGraph pose_graph;
  const std::vector<std::pair<image_t, image_t>> pairs =
      RandomGraphPairs(kNumImages, /*num_neighbors=*/4);
  // Biases below the fixed angular threshold.
  constexpr int kNumBiased = 3;
  FlatHashSet<image_pair_t> biased_pair_ids;
  for (size_t i = 0; i < pairs.size(); ++i) {
    const auto [image_id1, image_id2] = pairs[i];
    Eigen::Vector3d bias = Eigen::Vector3d::Zero();
    if (i % (pairs.size() / kNumBiased) == 1 &&
        biased_pair_ids.size() < kNumBiased) {
      bias = DegToRad(4.0) * RandomGaussianVector().normalized();
      biased_pair_ids.insert(ImagePairToPairId(image_id1, image_id2));
    }
    AddNoisyEdge(rotations,
                 image_id1,
                 image_id2,
                 IsotropicCovariance(0.2),
                 pose_graph,
                 bias);
  }
  ASSERT_EQ(biased_pair_ids.size(), kNumBiased);
  SolveRotationAveraging(options, pose_graph, reconstruction);

  const std::optional<RotationAveragingStatistics> result =
      EstimateRotationAveragingStatistics(options, pose_graph, reconstruction);
  ASSERT_TRUE(result.has_value());
  for (const auto& [pair_id, edge] : result->edges) {
    if (biased_pair_ids.count(pair_id)) {
      EXPECT_LT(edge.p_value, 1e-6);
    } else {
      EXPECT_GT(edge.p_value, 1e-6);
    }
  }

  PoseGraph filtered_pose_graph = pose_graph;
  EXPECT_EQ(FilterEdgesByRelativeRotationStatistics(
                *result, 1e-6, filtered_pose_graph),
            kNumBiased);
  for (const auto& [pair_id, edge] : filtered_pose_graph.Edges()) {
    EXPECT_EQ(edge.valid, biased_pair_ids.count(pair_id) == 0);
  }

  // The fixed angular threshold does not detect the biases.
  filtered_pose_graph = pose_graph;
  FilterEdgesByRelativeRotation(
      filtered_pose_graph, reconstruction, /*max_angle_deg=*/10);
  for (const auto& [pair_id, edge] : filtered_pose_graph.Edges()) {
    EXPECT_TRUE(edge.valid);
  }
}

TEST(EstimateRotationAveragingStatistics, AccountsForUncertainEdges) {
  SetPRNGSeed(3);
  RotationEstimatorOptions options = CreateCovarianceOptions();
  constexpr int kNumImages = 20;
  std::vector<Eigen::Quaterniond> rotations;
  Reconstruction reconstruction = CreateReconstruction(kNumImages, rotations);
  PoseGraph pose_graph;
  for (const auto& [image_id1, image_id2] :
       RandomGraphPairs(kNumImages, /*num_neighbors=*/4)) {
    AddNoisyEdge(
        rotations, image_id1, image_id2, IsotropicCovariance(0.2), pose_graph);
  }
  // An uncertain edge with an error above the fixed angular threshold that is
  // consistent with its covariance.
  constexpr image_t kImageId1 = 1;
  image_t image_id2 = kNumImages;
  while (pose_graph.HasEdge(kImageId1, image_id2)) {
    --image_id2;
  }
  ASSERT_GT(image_id2, kImageId1);
  AddNoisyEdge(rotations,
               kImageId1,
               image_id2,
               /*cov=*/IsotropicCovariance(5),
               pose_graph,
               /*bias=*/Eigen::Vector3d(DegToRad(11.0), 0, 0));
  // Remove the noise to control the error.
  pose_graph.EdgeRef(kImageId1, image_id2).first.cam2_from_cam1.rotation() =
      rotations[image_id2 - 1] * rotations[kImageId1 - 1].inverse() *
      ExpMap(Eigen::Vector3d(DegToRad(11.0), 0, 0));
  SolveRotationAveraging(options, pose_graph, reconstruction);

  const std::optional<RotationAveragingStatistics> result =
      EstimateRotationAveragingStatistics(options, pose_graph, reconstruction);
  ASSERT_TRUE(result.has_value());
  const RelativeRotationStatistics& edge =
      result->edges.at(ImagePairToPairId(kImageId1, image_id2));
  EXPECT_EQ(edge.num_dofs, 3);
  EXPECT_GT(edge.min_redundancy, 0.9);
  EXPECT_GT(edge.p_value, 0.1);
}

TEST(EstimateRotationAveragingStatistics, DoesNotTestBridges) {
  SetPRNGSeed(4);
  RotationEstimatorOptions options = CreateCovarianceOptions();
  constexpr int kClusterSize = 6;
  std::vector<Eigen::Quaterniond> rotations;
  Reconstruction reconstruction =
      CreateReconstruction(2 * kClusterSize, rotations);
  PoseGraph pose_graph;
  for (int cluster = 0; cluster < 2; ++cluster) {
    for (int i = 1; i <= kClusterSize; ++i) {
      for (int j = i + 1; j <= kClusterSize; ++j) {
        AddNoisyEdge(rotations,
                     cluster * kClusterSize + i,
                     cluster * kClusterSize + j,
                     IsotropicCovariance(0.2),
                     pose_graph);
      }
    }
  }
  // A biased bridge edge between the clusters.
  constexpr image_t kBridgeImageId1 = 1;
  constexpr image_t kBridgeImageId2 = kClusterSize + 1;
  AddNoisyEdge(rotations,
               kBridgeImageId1,
               kBridgeImageId2,
               IsotropicCovariance(0.2),
               pose_graph,
               /*bias=*/Eigen::Vector3d(0, DegToRad(20.0), 0));
  SolveRotationAveraging(options, pose_graph, reconstruction);

  const std::optional<RotationAveragingStatistics> result =
      EstimateRotationAveragingStatistics(options, pose_graph, reconstruction);
  ASSERT_TRUE(result.has_value());
  const RelativeRotationStatistics& bridge =
      result->edges.at(ImagePairToPairId(kBridgeImageId1, kBridgeImageId2));
  EXPECT_EQ(bridge.num_dofs, 0);
  EXPECT_NEAR(bridge.min_redundancy, 0, 1e-6);
  EXPECT_EQ(bridge.p_value, 1);
  for (const auto& [pair_id, edge] : result->edges) {
    if (pair_id != ImagePairToPairId(kBridgeImageId1, kBridgeImageId2)) {
      EXPECT_EQ(edge.num_dofs, 3);
    }
  }
}

TEST(EstimateRotationAveragingStatistics, IgnoresImagesWithoutPoses) {
  SetPRNGSeed(5);
  RotationEstimatorOptions options = CreateCovarianceOptions();
  constexpr int kNumImages = 10;
  std::vector<Eigen::Quaterniond> rotations;
  Reconstruction reconstruction = CreateReconstruction(kNumImages, rotations);
  PoseGraph pose_graph;
  for (const auto& [image_id1, image_id2] :
       RandomGraphPairs(kNumImages, /*num_neighbors=*/3)) {
    AddNoisyEdge(
        rotations, image_id1, image_id2, IsotropicCovariance(1), pose_graph);
  }
  SolveRotationAveraging(options, pose_graph, reconstruction);
  constexpr image_t kUnposedImageId = 3;
  reconstruction.DeRegisterFrame(
      reconstruction.Image(kUnposedImageId).FrameId());
  reconstruction.Frame(reconstruction.Image(kUnposedImageId).FrameId())
      .ResetPose();

  const std::optional<RotationAveragingStatistics> result =
      EstimateRotationAveragingStatistics(options, pose_graph, reconstruction);
  ASSERT_TRUE(result.has_value());
  for (const auto& [pair_id, edge] : pose_graph.Edges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    EXPECT_EQ(result->edges.count(pair_id) > 0,
              image_id1 != kUnposedImageId && image_id2 != kUnposedImageId);
  }
}

TEST(RunRotationAveraging, StatisticalFiltering) {
  SetPRNGSeed(6);
  RotationEstimatorOptions options = CreateCovarianceOptions();
  options.rotation_outlier_significance = 1e-6;
  options.max_rotation_error_deg = 0;
  constexpr int kNumImages = 20;
  std::vector<Eigen::Quaterniond> rotations;
  Reconstruction reconstruction = CreateReconstruction(kNumImages, rotations);
  PoseGraph pose_graph;
  for (const auto& [image_id1, image_id2] :
       RandomGraphPairs(kNumImages, /*num_neighbors=*/4)) {
    AddNoisyEdge(
        rotations, image_id1, image_id2, IsotropicCovariance(0.2), pose_graph);
  }
  constexpr image_t kImageId1 = 1;
  constexpr image_t kImageId2 = 2;
  ASSERT_TRUE(pose_graph.HasEdge(kImageId1, kImageId2));
  pose_graph.EdgeRef(kImageId1, kImageId2).first.cam2_from_cam1.rotation() *=
      ExpMap(Eigen::Vector3d(0, 0, DegToRad(5.0)));

  ASSERT_TRUE(RunRotationAveraging(options, pose_graph, reconstruction, {}));
  EXPECT_FALSE(pose_graph.IsValid(ImagePairToPairId(kImageId1, kImageId2)));
  EXPECT_EQ(reconstruction.NumRegFrames(), kNumImages);
}

}  // namespace
}  // namespace colmap
