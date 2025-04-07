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
#include "colmap/geometry/triangulation.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

struct TwoViewGeometryTestData {
  Camera camera1;
  Camera camera2;
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  TwoViewGeometry geometry;
};

TwoViewGeometryTestData CreateTwoViewGeometryTestData(
    TwoViewGeometry::ConfigurationType config) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras = 2;
  synthetic_dataset_options.num_images = 2;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.point2D_stddev = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const Image& image1 = reconstruction.Image(1);
  const Image& image2 = reconstruction.Image(2);

  TwoViewGeometryTestData data;
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

  for (const Point2D& point2D : image1.Points2D()) {
    data.points1.emplace_back(point2D.xy);
  }
  for (const Point2D& point2D : image2.Points2D()) {
    data.points2.emplace_back(point2D.xy);
  }

  std::vector<Eigen::Vector3d> points3D;
  for (const auto& [_, point3D] : reconstruction.Points3D()) {
    points3D.push_back(point3D.xyz);
    CHECK_EQ(point3D.track.Length(), 2);
    const TrackElement& elem1 = point3D.track.Element(0);
    const TrackElement& elem2 = point3D.track.Element(1);
    if (elem1.image_id == image1.ImageId() &&
        elem2.image_id == image2.ImageId()) {
      data.geometry.inlier_matches.emplace_back(elem1.point2D_idx,
                                                elem2.point2D_idx);
    } else if (elem1.image_id == image2.ImageId() &&
               elem2.image_id == image1.ImageId()) {
      data.geometry.inlier_matches.emplace_back(elem2.point2D_idx,
                                                elem1.point2D_idx);
    } else {
      LOG(FATAL) << "Invalid track element.";
    }
  }

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
    const TwoViewGeometryTestData test_data = CreateTwoViewGeometryTestData(
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
    TwoViewGeometryTestData test_data = CreateTwoViewGeometryTestData(config);
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
    const TwoViewGeometryTestData test_data = CreateTwoViewGeometryTestData(
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
    const TwoViewGeometryTestData test_data = CreateTwoViewGeometryTestData(
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
      const TwoViewGeometryTestData test_data =
          CreateTwoViewGeometryTestData(config);

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

}  // namespace
}  // namespace colmap
