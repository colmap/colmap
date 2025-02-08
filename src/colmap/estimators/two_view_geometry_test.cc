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
#include "colmap/geometry/triangulation.h"
#include "colmap/math/math.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(EstimateTwoViewGeometryPose, Calibrated) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras = 2;
  synthetic_dataset_options.num_images = 2;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.point2D_stddev = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto& image1 = reconstruction.Image(1);
  const auto& camera1 = reconstruction.Camera(image1.CameraId());
  const auto& image2 = reconstruction.Image(2);
  const auto& camera2 = reconstruction.Camera(image2.CameraId());
  const Rigid3d cam2_from_cam1 =
      image2.CamFromWorld() * Inverse(image1.CamFromWorld());

  TwoViewGeometry geometry;
  geometry.config = TwoViewGeometry::ConfigurationType::CALIBRATED;
  geometry.E = EssentialMatrixFromPose(cam2_from_cam1);

  std::vector<Eigen::Vector2d> points1;
  for (const auto& point2D : image1.Points2D()) {
    points1.emplace_back(point2D.xy);
  }

  std::vector<Eigen::Vector2d> points2;
  for (const auto& point2D : image2.Points2D()) {
    points2.emplace_back(point2D.xy);
  }

  std::vector<Eigen::Vector3d> points3D;
  for (const auto& point3D : reconstruction.Points3D()) {
    points3D.push_back(point3D.second.xyz);
    CHECK_EQ(point3D.second.track.Length(), 2);
    const auto& element1 = point3D.second.track.Element(0);
    const auto& element2 = point3D.second.track.Element(1);
    if (element1.image_id == image1.ImageId() &&
        element2.image_id == image2.ImageId()) {
      geometry.inlier_matches.emplace_back(element1.point2D_idx,
                                           element2.point2D_idx);
    } else if (element1.image_id == image2.ImageId() &&
               element2.image_id == image1.ImageId()) {
      geometry.inlier_matches.emplace_back(element2.point2D_idx,
                                           element1.point2D_idx);
    } else {
      LOG(FATAL) << "Invalid track element.";
    }
  }

  const double tri_angle = Median(CalculateTriangulationAngles(
      image1.ProjectionCenter(), image2.ProjectionCenter(), points3D));

  EXPECT_TRUE(EstimateTwoViewGeometryPose(
      camera1, points1, camera2, points2, &geometry));
  EXPECT_NEAR(geometry.tri_angle, tri_angle, 1e-6);
  EXPECT_THAT(geometry.cam2_from_cam1.rotation.coeffs(),
              EigenMatrixNear(cam2_from_cam1.rotation.coeffs(), 1e-6));
  EXPECT_THAT(geometry.cam2_from_cam1.translation,
              EigenMatrixNear(cam2_from_cam1.translation.normalized(), 1e-6));
}

// TODO: Add test for uncalibrated, panoramic, planar cases.

}  // namespace
}  // namespace colmap
