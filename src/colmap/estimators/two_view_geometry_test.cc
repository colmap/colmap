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

#include "colmap/estimators/two_view_geometry.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/homography_matrix.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void ExtractPointsAndMatches(const Reconstruction& reconstruction,
                             const Image& image1,
                             const Image& image2,
                             std::vector<Eigen::Vector2d>& points1,
                             std::vector<Eigen::Vector2d>& points2,
                             std::vector<Eigen::Vector3d>& points3D,
                             FeatureMatches& matches) {
  points1.clear();
  points2.clear();
  matches.clear();

  for (const Point2D& point2D : image1.Points2D()) {
    points1.emplace_back(point2D.xy);
  }

  for (const Point2D& point2D : image2.Points2D()) {
    points2.emplace_back(point2D.xy);
  }

  for (const auto& [_, point3D] : reconstruction.Points3D()) {
    const Track& track = point3D.track;
    CHECK_EQ(track.Length(), 2);

    points3D.emplace_back(point3D.xyz);

    const auto& elem1 = track.Element(0);
    const auto& elem2 = track.Element(1);

    int idx1 = -1, idx2 = -1;
    if (elem1.image_id == image1.ImageId() &&
        elem2.image_id == image2.ImageId()) {
      idx1 = elem1.point2D_idx;
      idx2 = elem2.point2D_idx;
    } else if (elem1.image_id == image2.ImageId() &&
               elem2.image_id == image1.ImageId()) {
      idx1 = elem2.point2D_idx;
      idx2 = elem1.point2D_idx;
    } else {
      LOG(FATAL) << "Invalid track element.";
    }

    matches.emplace_back(idx1, idx2);
  }
}

struct TwoViewGeometryPoseTestData {
  Camera camera1;
  Camera camera2;
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  TwoViewGeometry geometry;
};

TwoViewGeometryPoseTestData CreateTwoViewGeometryPoseTestData(
    TwoViewGeometry::ConfigurationType config) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const Image& image1 = reconstruction.Image(1);
  const Image& image2 = reconstruction.Image(2);

  TwoViewGeometryPoseTestData data;
  data.camera1 = reconstruction.Camera(image1.CameraId());
  data.camera2 = reconstruction.Camera(image2.CameraId());
  data.geometry.config = config;
  data.geometry.cam2_from_cam1 =
      image2.CamFromWorld() * Inverse(image1.CamFromWorld());

  if (config == TwoViewGeometry::ConfigurationType::CALIBRATED) {
    data.geometry.E = EssentialMatrixFromPose(data.geometry.cam2_from_cam1);
  } else if (config == TwoViewGeometry::ConfigurationType::UNCALIBRATED) {
    data.geometry.F = FundamentalFromEssentialMatrix(
        data.camera2.CalibrationMatrix(),
        EssentialMatrixFromPose(data.geometry.cam2_from_cam1),
        data.camera1.CalibrationMatrix());
  } else if (config == TwoViewGeometry::ConfigurationType::PLANAR) {
    const Eigen::Vector3d homography_plane_normal =
        image1.CamFromWorld().rotation *
        -(image1.ViewingDirection() + image2.ViewingDirection()).normalized();
    constexpr double kHomographyPlaneDistance = 1;
    data.geometry.H =
        HomographyMatrixFromPose(data.camera1.CalibrationMatrix(),
                                 data.camera2.CalibrationMatrix(),
                                 data.geometry.cam2_from_cam1.rotation.matrix(),
                                 data.geometry.cam2_from_cam1.translation,
                                 homography_plane_normal,
                                 kHomographyPlaneDistance);
  } else if (config == TwoViewGeometry::ConfigurationType::PANORAMIC) {
    data.geometry.cam2_from_cam1.translation = Eigen::Vector3d::Zero();
    data.geometry.H =
        HomographyMatrixFromPose(data.camera1.CalibrationMatrix(),
                                 data.camera2.CalibrationMatrix(),
                                 data.geometry.cam2_from_cam1.rotation.matrix(),
                                 data.geometry.cam2_from_cam1.translation,
                                 Eigen::Vector3d::UnitZ(),
                                 1);
  } else {
    LOG(FATAL) << "Invalid configuration.";
  }

  std::vector<Eigen::Vector3d> points3D;
  ExtractPointsAndMatches(reconstruction,
                          image1,
                          image2,
                          data.points1,
                          data.points2,
                          points3D,
                          data.geometry.inlier_matches);

  if (config == TwoViewGeometry::ConfigurationType::PANORAMIC) {
    data.geometry.tri_angle = 0;
  } else {
    data.geometry.tri_angle = Median(CalculateTriangulationAngles(
        image1.ProjectionCenter(), image2.ProjectionCenter(), points3D));
  }

  return data;
}

bool CheckEqualTwoViewGeometry(const TwoViewGeometry& geometry,
                               const TwoViewGeometry& expected_geometry,
                               double tri_angle_tol,
                               double rotation_tol,
                               double translation_tol,
                               bool normalized_translation) {
  const double tri_angle_error =
      std::abs(geometry.tri_angle - expected_geometry.tri_angle);
  const double rotation_error =
      geometry.cam2_from_cam1.rotation.angularDistance(
          expected_geometry.cam2_from_cam1.rotation);
  const double translation_error =
      (geometry.cam2_from_cam1.translation -
       (normalized_translation
            ? expected_geometry.cam2_from_cam1.translation.normalized()
            : expected_geometry.cam2_from_cam1.translation))
          .norm();
  if (tri_angle_error > tri_angle_tol || rotation_error > rotation_tol ||
      translation_error > translation_tol) {
    LOG(ERROR) << "Two view geometries do not match with errors: tri_angle="
               << tri_angle_error << ", rotation=" << rotation_error
               << ", translation=" << translation_error;
    return false;
  }
  return true;
}

TEST(EstimateTwoViewGeometryPose, Calibrated) {
  constexpr int kNumTests = 100;
  int num_failures = 0;
  for (int seed = 0; seed < kNumTests; ++seed) {
    SetPRNGSeed(seed);
    const TwoViewGeometryPoseTestData test_data =
        CreateTwoViewGeometryPoseTestData(
            TwoViewGeometry::ConfigurationType::CALIBRATED);

    TwoViewGeometry geometry;
    geometry.config = test_data.geometry.config;
    geometry.E = test_data.geometry.E;
    geometry.inlier_matches = test_data.geometry.inlier_matches;
    EXPECT_TRUE(EstimateTwoViewGeometryPose(test_data.camera1,
                                            test_data.points1,
                                            test_data.camera2,
                                            test_data.points2,
                                            &geometry));
    if (!CheckEqualTwoViewGeometry(geometry,
                                   test_data.geometry,
                                   /*tri_angle_tol=*/1e-6,
                                   /*rotation_tol=*/1e-6,
                                   /*translation_tol=*/1e-6,
                                   /*normalized_translation=*/true)) {
      num_failures++;
    }
  }
  EXPECT_EQ(num_failures, 0);
}

TEST(EstimateTwoViewGeometryPose, FailureDueToInsufficientMatches) {
  SetPRNGSeed(0);
  for (const auto config : {TwoViewGeometry::ConfigurationType::CALIBRATED,
                            TwoViewGeometry::ConfigurationType::UNCALIBRATED,
                            TwoViewGeometry::ConfigurationType::PLANAR,
                            TwoViewGeometry::ConfigurationType::PANORAMIC}) {
    TwoViewGeometryPoseTestData test_data =
        CreateTwoViewGeometryPoseTestData(config);
    test_data.geometry.inlier_matches.clear();

    TwoViewGeometry geometry;
    geometry.config = test_data.geometry.config;
    geometry.E = test_data.geometry.E;
    geometry.inlier_matches = test_data.geometry.inlier_matches;
    EXPECT_FALSE(EstimateTwoViewGeometryPose(test_data.camera1,
                                             test_data.points1,
                                             test_data.camera2,
                                             test_data.points2,
                                             &geometry))
        << config;
  }
}

TEST(EstimateTwoViewGeometryPose, Uncalibrated) {
  constexpr int kNumTests = 100;
  int num_failures = 0;
  for (int seed = 0; seed < kNumTests; ++seed) {
    SetPRNGSeed(seed);
    const TwoViewGeometryPoseTestData test_data =
        CreateTwoViewGeometryPoseTestData(
            TwoViewGeometry::ConfigurationType::UNCALIBRATED);

    TwoViewGeometry geometry;
    geometry.config = test_data.geometry.config;
    geometry.F = test_data.geometry.F;
    geometry.inlier_matches = test_data.geometry.inlier_matches;
    EXPECT_TRUE(EstimateTwoViewGeometryPose(test_data.camera1,
                                            test_data.points1,
                                            test_data.camera2,
                                            test_data.points2,
                                            &geometry));
    if (!CheckEqualTwoViewGeometry(geometry,
                                   test_data.geometry,
                                   /*tri_angle_tol=*/1e-6,
                                   /*rotation_tol=*/1e-6,
                                   /*translation_tol=*/1e-6,
                                   /*normalized_translation=*/true)) {
      num_failures++;
    }
  }
  EXPECT_EQ(num_failures, 0);
}

TEST(EstimateTwoViewGeometryPose, Planar) {
  constexpr int kNumTests = 100;
  int num_failures = 0;
  for (int seed = 0; seed < kNumTests; ++seed) {
    SetPRNGSeed(seed);
    const TwoViewGeometryPoseTestData test_data =
        CreateTwoViewGeometryPoseTestData(
            TwoViewGeometry::ConfigurationType::PLANAR);

    TwoViewGeometry geometry;
    geometry.config = test_data.geometry.config;
    geometry.H = test_data.geometry.H;
    geometry.inlier_matches = test_data.geometry.inlier_matches;
    EXPECT_TRUE(EstimateTwoViewGeometryPose(test_data.camera1,
                                            test_data.points1,
                                            test_data.camera2,
                                            test_data.points2,
                                            &geometry));
    if (!CheckEqualTwoViewGeometry(geometry,
                                   test_data.geometry,
                                   /*tri_angle_tol=*/1e-3,
                                   /*rotation_tol=*/1e-6,
                                   /*translation_tol=*/1e-5,
                                   /*normalized_translation=*/false)) {
      num_failures++;
    }
  }
  EXPECT_EQ(num_failures, 0);
}

TEST(EstimateTwoViewGeometryPose, PlanarOrPanoramic) {
  constexpr int kNumTests = 100;
  int num_failures = 0;
  for (int seed = 0; seed < kNumTests; ++seed) {
    SetPRNGSeed(seed);
    for (const auto config : {TwoViewGeometry::ConfigurationType::PLANAR,
                              TwoViewGeometry::ConfigurationType::PANORAMIC}) {
      const TwoViewGeometryPoseTestData test_data =
          CreateTwoViewGeometryPoseTestData(config);

      TwoViewGeometry geometry;
      geometry.config = TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
      geometry.H = test_data.geometry.H;
      geometry.inlier_matches = test_data.geometry.inlier_matches;
      EXPECT_TRUE(EstimateTwoViewGeometryPose(test_data.camera1,
                                              test_data.points1,
                                              test_data.camera2,
                                              test_data.points2,
                                              &geometry));
      EXPECT_EQ(geometry.config, config);
      if (!CheckEqualTwoViewGeometry(geometry,
                                     test_data.geometry,
                                     /*tri_angle_tol=*/1e-3,
                                     /*rotation_tol=*/1e-6,
                                     /*translation_tol=*/1e-6,
                                     /*normalized_translation=*/false)) {
        num_failures++;
      }
    }
  }
  EXPECT_EQ(num_failures, 0);
}

struct TwoViewGeometryTestData {
  Camera camera1;
  Camera camera2;
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  FeatureMatches matches;
};

TwoViewGeometryTestData CreateTwoViewGeometryTestData(
    const SyntheticDatasetOptions& synthetic_dataset_options,
    const SyntheticNoiseOptions& synthetic_noise_options = {}) {
  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  CHECK_EQ(reconstruction.NumImages(), 2);
  const Image& image1 = reconstruction.Image(1);
  const Image& image2 = reconstruction.Image(2);

  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  std::vector<Eigen::Vector3d> points3D;
  FeatureMatches matches;

  TwoViewGeometryTestData data;
  data.camera1 = reconstruction.Camera(image1.CameraId());
  data.camera2 = reconstruction.Camera(image2.CameraId());

  ExtractPointsAndMatches(reconstruction,
                          image1,
                          image2,
                          data.points1,
                          data.points2,
                          points3D,
                          data.matches);

  return data;
}

TEST(EstimateTwoViewGeometry, DetectWatermark) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  TwoViewGeometryTestData test_data =
      CreateTwoViewGeometryTestData(synthetic_dataset_options);

  TwoViewGeometryOptions two_view_geometry_options;
  two_view_geometry_options.detect_watermark = true;
  EXPECT_NE(EstimateTwoViewGeometry(test_data.camera1,
                                    test_data.points1,
                                    test_data.camera2,
                                    test_data.points2,
                                    test_data.matches,
                                    two_view_geometry_options)
                .config,
            TwoViewGeometry::ConfigurationType::WATERMARK);

  // Place the points on the left and right side of the images.
  for (size_t i = 0; i < test_data.matches.size(); ++i) {
    const double y = static_cast<double>(i) / test_data.matches.size() *
                     test_data.camera1.height;
    test_data.points1[test_data.matches[i].point2D_idx1] =
        Eigen::Vector2d(0, y);
    test_data.points2[test_data.matches[i].point2D_idx2] =
        Eigen::Vector2d(test_data.camera2.width - 1, y);
  }
  EXPECT_EQ(EstimateTwoViewGeometry(test_data.camera1,
                                    test_data.points1,
                                    test_data.camera2,
                                    test_data.points2,
                                    test_data.matches,
                                    two_view_geometry_options)
                .config,
            TwoViewGeometry::ConfigurationType::WATERMARK);

  // Place the points on the top and bottom side of the images.
  for (size_t i = 0; i < test_data.matches.size(); ++i) {
    const double x = static_cast<double>(i) / test_data.matches.size() *
                     test_data.camera1.width;
    test_data.points1[test_data.matches[i].point2D_idx1] =
        Eigen::Vector2d(x, 0);
    test_data.points2[test_data.matches[i].point2D_idx2] =
        Eigen::Vector2d(x, test_data.camera2.height - 1);
  }
  EXPECT_EQ(EstimateTwoViewGeometry(test_data.camera1,
                                    test_data.points1,
                                    test_data.camera2,
                                    test_data.points2,
                                    test_data.matches,
                                    two_view_geometry_options)
                .config,
            TwoViewGeometry::ConfigurationType::WATERMARK);

  // With disabled detection, expect a normal config.
  two_view_geometry_options.detect_watermark = false;
  EXPECT_NE(EstimateTwoViewGeometry(test_data.camera1,
                                    test_data.points1,
                                    test_data.camera2,
                                    test_data.points2,
                                    test_data.matches,
                                    two_view_geometry_options)
                .config,
            TwoViewGeometry::ConfigurationType::WATERMARK);
}

TEST(EstimateTwoViewGeometry, IgnoreStationaryMatches) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 500;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  TwoViewGeometryTestData test_data =
      CreateTwoViewGeometryTestData(synthetic_dataset_options);

  for (auto& match : test_data.matches) {
    test_data.points1[match.point2D_idx1] =
        test_data.points2[match.point2D_idx2];
  }

  TwoViewGeometryOptions two_view_geometry_options;
  TwoViewGeometry geometry1 =
      EstimateTwoViewGeometry(test_data.camera1,
                              test_data.points1,
                              test_data.camera2,
                              test_data.points2,
                              test_data.matches,
                              two_view_geometry_options);
  EXPECT_EQ(geometry1.config,
            TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC);
  EXPECT_EQ(geometry1.inlier_matches.size(),
            synthetic_dataset_options.num_points3D);

  two_view_geometry_options.filter_stationary_matches = true;
  TwoViewGeometry geometry2 =
      EstimateTwoViewGeometry(test_data.camera1,
                              test_data.points1,
                              test_data.camera2,
                              test_data.points2,
                              test_data.matches,
                              two_view_geometry_options);
  EXPECT_EQ(geometry2.config, TwoViewGeometry::ConfigurationType::DEGENERATE);
  EXPECT_EQ(geometry2.inlier_matches.size(), 0);
}

TEST(EstimateTwoViewGeometry, CalibratedDeterministic) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 500;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 5;
  const TwoViewGeometryTestData test_data = CreateTwoViewGeometryTestData(
      synthetic_dataset_options, synthetic_noise_options);

  TwoViewGeometryOptions two_view_geometry_options;
  two_view_geometry_options.ransac_options.random_seed = 42;
  const TwoViewGeometry geometry1 =
      EstimateTwoViewGeometry(test_data.camera1,
                              test_data.points1,
                              test_data.camera2,
                              test_data.points2,
                              test_data.matches,
                              two_view_geometry_options);
  EXPECT_EQ(geometry1.config, TwoViewGeometry::ConfigurationType::CALIBRATED);

  two_view_geometry_options.ransac_options.random_seed = 42;
  const TwoViewGeometry geometry2 =
      EstimateTwoViewGeometry(test_data.camera1,
                              test_data.points1,
                              test_data.camera2,
                              test_data.points2,
                              test_data.matches,
                              two_view_geometry_options);
  EXPECT_EQ(geometry2.config, TwoViewGeometry::ConfigurationType::CALIBRATED);

  // Using the same random seed should produce identical results.
  EXPECT_EQ(geometry1.E, geometry2.E);

  two_view_geometry_options.ransac_options.random_seed = 123;
  const TwoViewGeometry geometry3 =
      EstimateTwoViewGeometry(test_data.camera1,
                              test_data.points1,
                              test_data.camera2,
                              test_data.points2,
                              test_data.matches,
                              two_view_geometry_options);
  EXPECT_EQ(geometry3.config, TwoViewGeometry::ConfigurationType::CALIBRATED);

  // Using a different random seed may produce different results.
  EXPECT_NE(geometry1.E, geometry3.E);
}

TEST(EstimateTwoViewGeometry, UncalibratedDeterministic) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 500;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 5;
  const TwoViewGeometryTestData test_data = CreateTwoViewGeometryTestData(
      synthetic_dataset_options, synthetic_noise_options);

  TwoViewGeometryOptions two_view_geometry_options;
  two_view_geometry_options.ransac_options.random_seed = 42;
  const TwoViewGeometry geometry1 =
      EstimateTwoViewGeometry(test_data.camera1,
                              test_data.points1,
                              test_data.camera2,
                              test_data.points2,
                              test_data.matches,
                              two_view_geometry_options);
  EXPECT_EQ(geometry1.config, TwoViewGeometry::ConfigurationType::UNCALIBRATED);

  two_view_geometry_options.ransac_options.random_seed = 42;
  const TwoViewGeometry geometry2 =
      EstimateTwoViewGeometry(test_data.camera1,
                              test_data.points1,
                              test_data.camera2,
                              test_data.points2,
                              test_data.matches,
                              two_view_geometry_options);
  EXPECT_EQ(geometry2.config, TwoViewGeometry::ConfigurationType::UNCALIBRATED);

  // Using the same random seed should produce identical results.
  EXPECT_EQ(geometry1.F, geometry2.F);

  two_view_geometry_options.ransac_options.random_seed = 123;
  const TwoViewGeometry geometry3 =
      EstimateTwoViewGeometry(test_data.camera1,
                              test_data.points1,
                              test_data.camera2,
                              test_data.points2,
                              test_data.matches,
                              two_view_geometry_options);
  EXPECT_EQ(geometry3.config, TwoViewGeometry::ConfigurationType::UNCALIBRATED);

  // Using a different random seed may produce different results.
  EXPECT_NE(geometry1.F, geometry3.F);
}

TEST(EstimateTwoViewGeometry, PlanarOrPanoramicDeterministic) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 500;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0;
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 5;
  const TwoViewGeometryTestData test_data = CreateTwoViewGeometryTestData(
      synthetic_dataset_options, synthetic_noise_options);

  TwoViewGeometryOptions two_view_geometry_options;
  two_view_geometry_options.force_H_use = true;
  two_view_geometry_options.ransac_options.random_seed = 42;
  const TwoViewGeometry geometry1 =
      EstimateTwoViewGeometry(test_data.camera1,
                              test_data.points1,
                              test_data.camera2,
                              test_data.points2,
                              test_data.matches,
                              two_view_geometry_options);
  EXPECT_EQ(geometry1.config,
            TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC);

  two_view_geometry_options.ransac_options.random_seed = 42;
  const TwoViewGeometry geometry2 =
      EstimateTwoViewGeometry(test_data.camera1,
                              test_data.points1,
                              test_data.camera2,
                              test_data.points2,
                              test_data.matches,
                              two_view_geometry_options);
  EXPECT_EQ(geometry2.config,
            TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC);

  // Using the same random seed should produce identical results.
  EXPECT_EQ(geometry1.H, geometry2.H);

  two_view_geometry_options.ransac_options.random_seed = 123;
  const TwoViewGeometry geometry3 =
      EstimateTwoViewGeometry(test_data.camera1,
                              test_data.points1,
                              test_data.camera2,
                              test_data.points2,
                              test_data.matches,
                              two_view_geometry_options);
  EXPECT_EQ(geometry2.config,
            TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC);

  // Using a different random seed may produce different results.
  EXPECT_NE(geometry1.H, geometry3.H);
}

TEST(TwoViewGeometryFromKnownRelativePose, Nominal) {
  constexpr int kNumTests = 100;
  for (int seed = 0; seed < kNumTests; ++seed) {
    SetPRNGSeed(seed);
    const TwoViewGeometryPoseTestData test_data =
        CreateTwoViewGeometryPoseTestData(
            TwoViewGeometry::ConfigurationType::CALIBRATED);

    TwoViewGeometry geometry =
        TwoViewGeometryFromKnownRelativePose(test_data.camera1,
                                             test_data.points1,
                                             test_data.camera2,
                                             test_data.points2,
                                             test_data.geometry.cam2_from_cam1,
                                             test_data.geometry.inlier_matches,
                                             /*min_num_inliers=*/15,
                                             /*max_error=*/4.0);

    EXPECT_EQ(geometry.cam2_from_cam1, test_data.geometry.cam2_from_cam1);
    EXPECT_EQ(geometry.E, test_data.geometry.E);
    EXPECT_EQ(geometry.inlier_matches, test_data.geometry.inlier_matches);
  }
}

struct RigTwoViewGeometryTestData {
  Rig rig1;
  Rig rig2;
  std::vector<std::pair<std::pair<image_t, image_t>, FeatureMatches>> matches;
  Reconstruction reconstruction;
};

RigTwoViewGeometryTestData CreateRigTwoViewGeometryTestData(
    const SyntheticDatasetOptions& synthetic_dataset_options) {
  RigTwoViewGeometryTestData data;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  SynthesizeDataset(
      synthetic_dataset_options, &data.reconstruction, database.get());

  CHECK_EQ(data.reconstruction.NumRigs(), 2);

  data.rig1 = data.reconstruction.Rig(1);
  data.rig2 = data.reconstruction.Rig(2);
  for (auto& [pair_id, matches] : database->ReadAllMatches()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const auto& camera1 = data.reconstruction.Camera(
        data.reconstruction.Image(image_id1).CameraId());
    const auto& camera2 = data.reconstruction.Camera(
        data.reconstruction.Image(image_id2).CameraId());
    if (data.rig1.HasSensor(camera1.SensorId()) &&
        data.rig2.HasSensor(camera2.SensorId())) {
      data.matches.emplace_back(std::make_pair(image_id1, image_id2), matches);
    } else if (data.rig1.HasSensor(camera2.SensorId()) &&
               data.rig2.HasSensor(camera1.SensorId())) {
      data.matches.emplace_back(std::make_pair(image_id2, image_id1), matches);
    }
    // else: Ignore matches between sensors in the same rig.
  }

  return data;
}

TEST(EstimateRigTwoViewGeometries, Nominal) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  const RigTwoViewGeometryTestData test_data =
      CreateRigTwoViewGeometryTestData(synthetic_dataset_options);

  TwoViewGeometryOptions two_view_geometry_options;
  two_view_geometry_options.ransac_options.random_seed = 42;
  const auto geometries =
      EstimateRigTwoViewGeometries(test_data.rig1,
                                   test_data.rig2,
                                   test_data.reconstruction.Images(),
                                   test_data.reconstruction.Cameras(),
                                   test_data.matches,
                                   two_view_geometry_options);
  EXPECT_EQ(geometries.size(), test_data.matches.size());
  for (const auto& [image_pair, geometry] : geometries) {
    EXPECT_EQ(geometry.config,
              TwoViewGeometry::ConfigurationType::CALIBRATED_RIG);
    EXPECT_THAT(
        geometry.cam2_from_cam1,
        Rigid3dNear(
            test_data.reconstruction.Image(image_pair.second).CamFromWorld() *
                Inverse(test_data.reconstruction.Image(image_pair.first)
                            .CamFromWorld()),
            /*rtol=*/1e-2,
            /*ttol=*/1e-3));
    EXPECT_GT(geometry.inlier_matches.size(), 0);
  }
}

TEST(EstimateMultipleTwoViewGeometries, SingleGeometry) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  const TwoViewGeometryTestData test_data =
      CreateTwoViewGeometryTestData(synthetic_dataset_options);

  TwoViewGeometryOptions options;
  options.multiple_models = true;
  options.ransac_options.random_seed = 42;

  const TwoViewGeometry geometry = EstimateTwoViewGeometry(test_data.camera1,
                                                           test_data.points1,
                                                           test_data.camera2,
                                                           test_data.points2,
                                                           test_data.matches,
                                                           options);

  EXPECT_EQ(geometry.config, TwoViewGeometry::ConfigurationType::CALIBRATED);
  EXPECT_GT(geometry.inlier_matches.size(), 0);
}

TEST(EstimateMultipleTwoViewGeometries, NoGeometry) {
  SetPRNGSeed(1);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 5;  // Too few points
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  const TwoViewGeometryTestData test_data =
      CreateTwoViewGeometryTestData(synthetic_dataset_options);

  TwoViewGeometryOptions options;
  options.multiple_models = true;
  options.min_num_inliers = 100;  // Require too many inliers
  options.ransac_options.random_seed = 42;

  const TwoViewGeometry geometry = EstimateTwoViewGeometry(test_data.camera1,
                                                           test_data.points1,
                                                           test_data.camera2,
                                                           test_data.points2,
                                                           test_data.matches,
                                                           options);

  EXPECT_EQ(geometry.config, TwoViewGeometry::ConfigurationType::DEGENERATE);
  EXPECT_EQ(geometry.inlier_matches.size(), 0);
}

TEST(EstimateMultipleTwoViewGeometries, MultipleGeometries) {
  SetPRNGSeed(1);

  // Create two separate synthetic datasets with different poses.

  Reconstruction reconstruction1;
  SyntheticDatasetOptions options1;
  options1.num_rigs = 2;
  options1.num_cameras_per_rig = 1;
  options1.num_frames_per_rig = 1;
  options1.num_points3D = 100;
  options1.camera_has_prior_focal_length = true;
  SynthesizeDataset(options1, &reconstruction1);

  Reconstruction reconstruction2;
  SyntheticDatasetOptions options2;
  options2.num_rigs = 2;
  options2.num_cameras_per_rig = 1;
  options2.num_frames_per_rig = 1;
  options2.num_points3D = 100;
  options2.camera_has_prior_focal_length = true;
  SynthesizeDataset(options2, &reconstruction2);

  const Image& image1 = reconstruction1.Image(1);
  const Image& image2 = reconstruction1.Image(2);
  const Image& image3 = reconstruction2.Image(1);
  const Image& image4 = reconstruction2.Image(2);

  const Camera camera1 = reconstruction1.Camera(image1.CameraId());
  const Camera camera2 = reconstruction1.Camera(image2.CameraId());
  EXPECT_EQ(camera1, reconstruction2.Camera(image3.CameraId()));
  EXPECT_EQ(camera2, reconstruction2.Camera(image4.CameraId()));

  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  std::vector<Eigen::Vector3d> points3D1;
  FeatureMatches matches1;
  ExtractPointsAndMatches(
      reconstruction1, image1, image2, points1, points2, points3D1, matches1);

  std::vector<Eigen::Vector2d> points3;
  std::vector<Eigen::Vector2d> points4;
  std::vector<Eigen::Vector3d> points3D2;
  FeatureMatches matches2;
  ExtractPointsAndMatches(
      reconstruction2, image3, image4, points3, points4, points3D2, matches2);

  std::vector<Eigen::Vector2d> all_points1;
  std::vector<Eigen::Vector2d> all_points2;
  const size_t matches_offset = points1.size();
  all_points1.insert(all_points1.end(), points1.begin(), points1.end());
  all_points1.insert(all_points1.end(), points3.begin(), points3.end());
  all_points2.insert(all_points2.end(), points2.begin(), points2.end());
  all_points2.insert(all_points2.end(), points4.begin(), points4.end());

  FeatureMatches all_matches = matches1;
  for (const auto& match : matches2) {
    all_matches.emplace_back(match.point2D_idx1 + matches_offset,
                             match.point2D_idx2 + matches_offset);
  }

  TwoViewGeometryOptions two_view_options;
  two_view_options.multiple_models = true;
  two_view_options.ransac_options.random_seed = 42;

  const TwoViewGeometry geometry = EstimateTwoViewGeometry(camera1,
                                                           all_points1,
                                                           camera2,
                                                           all_points2,
                                                           all_matches,
                                                           two_view_options);

  EXPECT_EQ(geometry.config, TwoViewGeometry::ConfigurationType::MULTIPLE);
  EXPECT_EQ(geometry.inlier_matches.size(), matches1.size() + matches2.size());
}

}  // namespace
}  // namespace colmap
