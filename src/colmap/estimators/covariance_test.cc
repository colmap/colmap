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
  options.solver_options.num_threads = 1;
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

TEST(Covariance, Poses) {
  Reconstruction reconstruction;
  GenerateReconstruction(&reconstruction);
  std::shared_ptr<BundleAdjuster> bundle_adjuster =
      BuildBundleAdjuster(&reconstruction);
  bundle_adjuster->Solve(&reconstruction);
  std::shared_ptr<ceres::Problem> problem = bundle_adjuster->Problem();

  const auto cov_ceres = EstimateCeresBACovariance(
      reconstruction, problem.get(), BACovarianceType::kOnlyPoses);
  EXPECT_EQ(cov_ceres.pose_covs.size(), reconstruction.NumImages() - 1);
  EXPECT_EQ(cov_ceres.point_covs.size(), 0);
  const auto cov_schur = EstimateSchurBACovariance(
      reconstruction, problem.get(), BACovarianceType::kOnlyPoses);
  EXPECT_EQ(cov_schur.pose_covs.size(), reconstruction.NumImages() - 1);
  EXPECT_EQ(cov_schur.point_covs.size(), 0);

  for (const auto& [image_id, _] : reconstruction.Images()) {
    if (cov_ceres.pose_covs.count(image_id) == 0) {
      EXPECT_EQ(cov_schur.pose_covs.count(image_id), 0);
    } else {
      ExpectNearEigenMatrixXd(cov_ceres.pose_covs.at(image_id),
                              cov_schur.pose_covs.at(image_id),
                              1e-6);
    }
  }
}

TEST(Covariance, RankDeficientPoints) {
  Reconstruction reconstruction;
  GenerateReconstruction(&reconstruction);

  // add poorly conditioned points into reconstruction
  const std::vector<image_t> reg_image_ids = reconstruction.RegImageIds();
  const size_t num_reg_images = reg_image_ids.size();
  for (size_t i = 0; i < num_reg_images - 1; ++i) {
    const image_t image_id1 = reg_image_ids[i];
    Image& image1 = reconstruction.Image(image_id1);
    const Camera& camera1 = reconstruction.Camera(image1.CameraId());
    const Eigen::Vector3d position_1 =
        Inverse(image1.CamFromWorld()).translation;
    const image_t image_id2 = reg_image_ids[i + 1];
    Image& image2 = reconstruction.Image(image_id2);
    const Camera& camera2 = reconstruction.Camera(image1.CameraId());
    const Eigen::Vector3d position_2 =
        Inverse(image2.CamFromWorld()).translation;
    for (const double& val : {0.2, 0.4, 0.6, 0.8}) {
      const Eigen::Vector3d point = val * position_1 + (1 - val) * position_2;
      Track track;

      Point2D point2D1;
      point2D1.xy =
          camera1.ImgFromCam((image1.CamFromWorld() * point).hnormalized());
      const point2D_t point2D_idx_1 = image1.NumPoints2D();
      image1.Points2D().push_back(point2D1);
      track.AddElement(image_id1, point2D_idx_1);

      Point2D point2D2;
      point2D2.xy =
          camera2.ImgFromCam((image2.CamFromWorld() * point).hnormalized());
      const point2D_t point2D_idx_2 = image2.NumPoints2D();
      image2.Points2D().push_back(point2D2);
      track.AddElement(image_id2, point2D_idx_2);

      const point3D_t point3D_id = reconstruction.AddPoint3D(point, track);
      image1.SetPoint3DForPoint2D(point2D_idx_1, point3D_id);
      image2.SetPoint3DForPoint2D(point2D_idx_2, point3D_id);
    }
  }

  std::shared_ptr<BundleAdjuster> bundle_adjuster =
      BuildBundleAdjuster(&reconstruction);
  bundle_adjuster->Solve(&reconstruction);
  std::shared_ptr<ceres::Problem> problem = bundle_adjuster->Problem();

  EXPECT_TRUE(EstimateCeresBACovariance(
                  reconstruction, problem.get(), BACovarianceType::kOnlyPoses)
                  .success);

  EXPECT_TRUE(EstimateSchurBACovariance(
                  reconstruction, problem.get(), BACovarianceType::kOnlyPoses)
                  .success);
}

}  // namespace colmap
