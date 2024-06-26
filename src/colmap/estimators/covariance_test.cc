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

std::shared_ptr<BundleAdjuster> BuildBundleAdjuster(
    Reconstruction* reconstruction) {
  BundleAdjustmentConfig config;
  std::vector<image_t> image_ids;
  for (const auto& image : reconstruction->Images()) {
    image_ids.push_back(image.first);
    config.AddImage(image.first);
  }
  config.SetConstantCamPose(image_ids[0]);
  config.SetConstantCamPositions(image_ids[1], {0});
  BundleAdjustmentOptions options;
  return std::make_shared<BundleAdjuster>(options, config);
}

void ExpectNearEigenMatrixXd(const Eigen::MatrixXd& mat1,
                             const Eigen::MatrixXd& mat2,
                             double tolerance) {
  ASSERT_EQ(mat1.rows(), mat2.rows());
  ASSERT_EQ(mat1.cols(), mat2.cols());
  for (int i = 0; i < mat1.rows(); ++i) {
    for (int j = 0; j < mat1.cols(); ++j) {
      EXPECT_NEAR(mat1(i, j), mat2(i, j), tolerance);
    }
  }
}

}  // namespace

TEST(Covariance, PoseCovarianceInterface) {
  Reconstruction reconstruction;
  GenerateReconstruction(&reconstruction);
  std::shared_ptr<BundleAdjuster> bundle_adjuster =
      BuildBundleAdjuster(&reconstruction);
  bundle_adjuster->Solve(&reconstruction);
  std::shared_ptr<ceres::Problem> problem = bundle_adjuster->Problem();

  std::map<image_t, Eigen::MatrixXd> image_id_to_covar_ceres;
  if (!EstimatePoseCovarianceCeresBackend(
          problem.get(), &reconstruction, image_id_to_covar_ceres)) {
    LOG(INFO) << "Skipping due to failure of ceres covariance computation.";
    return;
  }
  std::map<image_t, Eigen::MatrixXd> image_id_to_covar;
  ASSERT_TRUE(EstimatePoseCovariance(
      problem.get(), &reconstruction, image_id_to_covar));
  for (auto it = image_id_to_covar.begin(); it != image_id_to_covar.end();
       ++it) {
    ASSERT_TRUE(image_id_to_covar_ceres.find(it->first) !=
                image_id_to_covar_ceres.end());
    Eigen::MatrixXd covar = it->second;
    Eigen::MatrixXd covar_ceres = image_id_to_covar_ceres.at(it->first);
    ExpectNearEigenMatrixXd(covar, covar_ceres, 1e-6);
  }
}

TEST(Covariance, Compute) {
  Reconstruction reconstruction;
  GenerateReconstruction(&reconstruction);
  std::shared_ptr<BundleAdjuster> bundle_adjuster =
      BuildBundleAdjuster(&reconstruction);
  bundle_adjuster->Solve(&reconstruction);
  std::shared_ptr<ceres::Problem> problem = bundle_adjuster->Problem();

  BundleAdjustmentCovarianceEstimatorCeresBackend estimator_ceres(
      problem.get(), &reconstruction);
  if (!estimator_ceres.Compute()) {
    LOG(INFO) << "Skipping due to failure of ceres covariance computation.";
    return;
  }
  BundleAdjustmentCovarianceEstimator estimator(problem.get(), &reconstruction);
  ASSERT_TRUE(estimator.Compute());

  // covariance for each image
  std::vector<image_t> image_ids;
  for (const auto& image : reconstruction.Images()) {
    image_ids.push_back(image.first);
  }
  Eigen::MatrixXd covar, covar_ceres;
  size_t n_images = image_ids.size();
  for (size_t i = 0; i < n_images; ++i) {
    image_t image_id = image_ids[i];
    if (!estimator.HasPose(image_id)) continue;
    covar = estimator.GetPoseCovariance(image_id);
    covar_ceres = estimator_ceres.GetPoseCovariance(image_id);
    ExpectNearEigenMatrixXd(covar, covar_ceres, 1e-6);
  }

  // cross image covariance
  for (size_t i = 0; i < n_images - 1; ++i) {
    image_t image_id1 = image_ids[i];
    image_t image_id2 = image_ids[i + 1];
    if (!estimator.HasPose(image_id1) || !estimator.HasPose(image_id2))
      continue;
    covar = estimator.GetPoseCovariance(image_id1, image_id2);
    covar_ceres = estimator_ceres.GetPoseCovariance(image_id1, image_id2);
    ExpectNearEigenMatrixXd(covar, covar_ceres, 1e-6);
  }

  // multiple images
  std::vector<image_t> test_image_ids;
  for (size_t i = 0; i < n_images; ++i) {
    if (i % 2 != 0) continue;
    image_t image_id = image_ids[i];
    if (!estimator.HasPose(image_id)) continue;
    test_image_ids.push_back(image_id);
  }
  covar = estimator.GetPoseCovariance(test_image_ids);
  covar_ceres = estimator_ceres.GetPoseCovariance(test_image_ids);
  ExpectNearEigenMatrixXd(covar, covar_ceres, 1e-6);
}

TEST(Covariance, ComputeFull) {
  Reconstruction reconstruction;
  GenerateReconstruction(&reconstruction);
  std::shared_ptr<BundleAdjuster> bundle_adjuster =
      BuildBundleAdjuster(&reconstruction);
  bundle_adjuster->Solve(&reconstruction);
  std::shared_ptr<ceres::Problem> problem = bundle_adjuster->Problem();

  BundleAdjustmentCovarianceEstimatorCeresBackend estimator_ceres(
      problem.get(), &reconstruction);
  if (!estimator_ceres.ComputeFull()) {
    LOG(INFO) << "Skipping due to failure of ceres covariance computation.";
    return;
  }
  BundleAdjustmentCovarianceEstimator estimator(problem.get(), &reconstruction);
  ASSERT_TRUE(estimator.ComputeFull());
  std::vector<double*> parameter_blocks;
  for (const auto& camera : reconstruction.Cameras()) {
    const double* ptr = camera.second.params.data();
    if (!estimator.HasBlock(const_cast<double*>(ptr))) continue;
    parameter_blocks.push_back(const_cast<double*>(ptr));
  }
  Eigen::MatrixXd covar = estimator.GetCovariance(parameter_blocks);
  Eigen::MatrixXd covar_ceres = estimator_ceres.GetCovariance(parameter_blocks);
  ExpectNearEigenMatrixXd(covar, covar_ceres, 1e-6);
}

TEST(Covariance, RankDeficientPoints) {
  Reconstruction reconstruction;
  GenerateReconstruction(&reconstruction);

  // add poorly conditioned points into reconstruction
  std::vector<image_t> image_ids;
  for (const auto& image : reconstruction.Images()) {
    image_ids.push_back(image.first);
  }
  size_t n_images = image_ids.size();
  for (size_t i = 0; i < n_images - 1; ++i) {
    image_t image_id1 = image_ids[i];
    Image& image1 = reconstruction.Image(image_id1);
    const Camera& camera1 = reconstruction.Camera(image1.CameraId());
    Eigen::Vector3d position_1 = Inverse(image1.CamFromWorld()).translation;
    image_t image_id2 = image_ids[i + 1];
    Image& image2 = reconstruction.Image(image_id2);
    const Camera& camera2 = reconstruction.Camera(image1.CameraId());
    Eigen::Vector3d position_2 = Inverse(image2.CamFromWorld()).translation;
    std::vector<double> values = {0.2, 0.4, 0.6, 0.8};
    for (const double& val : values) {
      Eigen::Vector3d point = val * position_1 + (1 - val) * position_2;
      Track track;
      // image 1
      Point2D point2D_1;
      point2D_1.xy =
          camera1.ImgFromCam((image1.CamFromWorld() * point).hnormalized());
      point2D_t point2D_idx_1 = image1.NumPoints2D();
      auto& point2Ds_1 = image1.Points2D();
      point2Ds_1.push_back(point2D_1);
      track.AddElement(image_id1, point2D_idx_1);
      // image 2
      Point2D point2D_2;
      point2D_2.xy =
          camera2.ImgFromCam((image2.CamFromWorld() * point).hnormalized());
      point2D_t point2D_idx_2 = image2.NumPoints2D();
      auto& point2Ds_2 = image2.Points2D();
      point2Ds_2.push_back(point2D_2);
      track.AddElement(image_id2, point2D_idx_2);
      // insert 3d point
      point3D_t point3D_id = reconstruction.AddPoint3D(point, track);
      // inverse index
      image1.SetPoint3DForPoint2D(point2D_idx_1, point3D_id);
      image2.SetPoint3DForPoint2D(point2D_idx_2, point3D_id);
    }
  }

  // bundle adjustment
  std::shared_ptr<BundleAdjuster> bundle_adjuster =
      BuildBundleAdjuster(&reconstruction);
  bundle_adjuster->Solve(&reconstruction);
  std::shared_ptr<ceres::Problem> problem = bundle_adjuster->Problem();

  // covariance computation
  BundleAdjustmentCovarianceEstimator estimator(problem.get(), &reconstruction);
  ASSERT_TRUE(estimator.Compute());
}

}  // namespace colmap
