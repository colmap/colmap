// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/manifold.h"
#include "colmap/math/random.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

void GenerateReconstruction(Reconstruction* reconstruction) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras = 3;
  synthetic_dataset_options.num_images = 8;
  synthetic_dataset_options.num_points3D = 1000;
  synthetic_dataset_options.point2D_stddev = 0.01;
  SynthesizeDataset(synthetic_dataset_options, reconstruction);
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
      options.params == BACovarianceOptions::Params::kOnlyPoints ||
      options.params == BACovarianceOptions::Params::kPosesAndPoints ||
      options.params == BACovarianceOptions::Params::kAll;
  const bool estimate_pose_covs =
      options.params == BACovarianceOptions::Params::kOnlyPoses ||
      options.params == BACovarianceOptions::Params::kPosesAndPoints ||
      options.params == BACovarianceOptions::Params::kAll;
  const bool estimate_other_covs =
      options.params == BACovarianceOptions::Params::kAll;

  Reconstruction reconstruction;
  GenerateReconstruction(&reconstruction);
  BundleAdjustmentConfig config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    config.AddImage(image_id);
    if (test_options.fixed_cam_poses) {
      config.SetConstantCamPose(image_id);
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

  auto bundle_adjuster = CreateDefaultBundleAdjuster(
      BundleAdjustmentOptions(), std::move(config), reconstruction);
  auto problem = bundle_adjuster->Problem();

  const std::optional<BACovariance> ba_cov =
      EstimateBACovariance(options, reconstruction, *bundle_adjuster);
  ASSERT_TRUE(ba_cov.has_value());

  const std::vector<detail::PointParam> points =
      detail::GetPointParams(reconstruction, *problem);
  const std::vector<detail::PoseParam> poses =
      detail::GetPoseParams(reconstruction, *problem);
  const std::vector<const double*> others =
      GetOtherParams(*problem, poses, points);

  if (!test_options.fixed_cam_poses && estimate_pose_covs) {
    LOG(INFO) << "Comparing pose covariances";

    std::vector<std::pair<const double*, const double*>> cov_param_pairs;
    for (const auto& pose : poses) {
      if (pose.qvec != nullptr) {
        cov_param_pairs.emplace_back(pose.qvec, pose.qvec);
      }
      if (pose.tvec != nullptr) {
        cov_param_pairs.emplace_back(pose.tvec, pose.tvec);
      }
      if (pose.qvec != nullptr && pose.tvec != nullptr) {
        cov_param_pairs.emplace_back(pose.qvec, pose.tvec);
      }
    }

    ceres::Covariance::Options ceres_cov_options;
    ceres::Covariance ceres_cov_computer(ceres_cov_options);
    ASSERT_TRUE(ceres_cov_computer.Compute(cov_param_pairs, problem.get()));

    for (const auto& pose : poses) {
      int tangent_size = 0;
      std::vector<const double*> param_blocks;
      if (pose.qvec != nullptr) {
        tangent_size += ParameterBlockTangentSize(*problem, pose.qvec);
        param_blocks.push_back(pose.qvec);
      }
      if (pose.tvec != nullptr) {
        tangent_size += ParameterBlockTangentSize(*problem, pose.tvec);
        param_blocks.push_back(pose.tvec);
      }

      Eigen::MatrixXd ceres_cov(tangent_size, tangent_size);
      ceres_cov_computer.GetCovarianceMatrixInTangentSpace(param_blocks,
                                                           ceres_cov.data());

      const std::optional<Eigen::MatrixXd> cov =
          ba_cov->GetCamFromWorldCov(pose.image_id);
      ASSERT_TRUE(cov.has_value());
      ExpectNearEigenMatrixXd(ceres_cov, *cov, /*tol=*/1e-8);
    }

    ASSERT_FALSE(ba_cov->GetCamFromWorldCov(kInvalidImageId).has_value());
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

    for (const auto& pose : poses) {
      if (pose.qvec != nullptr) {
        problem->SetParameterBlockConstant(pose.qvec);
      }
      if (pose.tvec != nullptr) {
        problem->SetParameterBlockConstant(pose.tvec);
      }
    }
    for (const double* other : others) {
      if (other != nullptr) {
        problem->SetParameterBlockConstant(other);
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

      const std::optional<Eigen::Matrix3d> cov =
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
          options.params = BACovarianceOptions::Params::kAll;
          BACovarianceTestOptions test_options;
          test_options.fixed_points = true;
          return std::make_pair(options, test_options);
        }(),
        []() {
          BACovarianceOptions options;
          options.params = BACovarianceOptions::Params::kAll;
          BACovarianceTestOptions test_options;
          test_options.fixed_cam_intrinsics = true;
          return std::make_pair(options, test_options);
        }(),
        []() {
          BACovarianceOptions options;
          options.params = BACovarianceOptions::Params::kAll;
          BACovarianceTestOptions test_options;
          test_options.fixed_cam_poses = true;
          return std::make_pair(options, test_options);
        }(),
        []() {
          BACovarianceOptions options;
          options.params = BACovarianceOptions::Params::kOnlyPoints;
          BACovarianceTestOptions test_options;
          return std::make_pair(options, test_options);
        }(),
        []() {
          BACovarianceOptions options;
          options.params = BACovarianceOptions::Params::kOnlyPoses;
          BACovarianceTestOptions test_options;
          return std::make_pair(options, test_options);
        }(),
        []() {
          BACovarianceOptions options;
          options.params = BACovarianceOptions::Params::kPosesAndPoints;
          BACovarianceTestOptions test_options;
          return std::make_pair(options, test_options);
        }()));

// TODO

// TEST(EstimatePointCovariances, RankDeficientPoints) {
//   Reconstruction reconstruction;
//   GenerateReconstruction(&reconstruction);

//   // Add poorly conditioned points to the reconstruction.
//   const std::set<image_t> reg_image_ids = reconstruction.RegImageIds();
//   for (auto it = ++reg_image_ids.begin(); it != reg_image_ids.end(); ++it) {
//     const image_t image_id1 = *std::prev(it);
//     Image& image1 = reconstruction.Image(image_id1);
//     const Camera& camera1 = reconstruction.Camera(image1.CameraId());
//     const Eigen::Vector3d point_xyz1 =
//         Inverse(image1.CamFromWorld()).translation;
//     const image_t image_id2 = *it;
//     Image& image2 = reconstruction.Image(image_id2);
//     const Camera& camera2 = reconstruction.Camera(image1.CameraId());
//     const Eigen::Vector3d point_xyz2 =
//         Inverse(image2.CamFromWorld()).translation;
//     for (const double& val : {0.2, 0.4, 0.6, 0.8}) {
//       const Eigen::Vector3d point = val * point_xyz1 + (1 - val) *
//       point_xyz2; Track track;

//       Point2D point2D1;
//       point2D1.xy =
//           camera1.ImgFromCam((image1.CamFromWorld() * point).hnormalized());
//       const point2D_t point2D_idx1 = image1.NumPoints2D();
//       image1.Points2D().push_back(point2D1);
//       track.AddElement(image_id1, point2D_idx1);

//       Point2D point2D2;
//       point2D2.xy =
//           camera2.ImgFromCam((image2.CamFromWorld() * point).hnormalized());
//       const point2D_t point2D_idx2 = image2.NumPoints2D();
//       image2.Points2D().push_back(point2D2);
//       track.AddElement(image_id2, point2D_idx2);

//       const point3D_t point3D_id = reconstruction.AddPoint3D(point, track);
//       image1.SetPoint3DForPoint2D(point2D_idx1, point3D_id);
//       image2.SetPoint3DForPoint2D(point2D_idx2, point3D_id);
//     }
//   }

//   BundleAdjustmentConfig config;
//   for (const auto& [image_id, image] : reconstruction.Images()) {
//     config.AddImage(image_id);
//     config.SetConstantCamPose(image_id);
//     config.SetConstantCamIntrinsics(image.CameraId());
//   }
//   for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
//     config.AddVariablePoint(point3D_id);
//   }
//   BundleAdjustmentOptions options;
//   options.solver_options.max_num_iterations = 0;
//   BundleAdjuster bundle_adjuster(options, config);
//   bundle_adjuster.Solve(&reconstruction);
//   std::shared_ptr<ceres::Problem> problem = bundle_adjuster.Problem();

//   EXPECT_EQ(EstimatePointCovariances(reconstruction, *problem,
//   /*damping=*/1e-8)
//                 .size(),
//             reconstruction.NumPoints3D());
//   EXPECT_LT(
//       EstimatePointCovariances(reconstruction, *problem,
//       /*damping=*/0).size(), reconstruction.NumPoints3D());
// }

}  // namespace
}  // namespace colmap
