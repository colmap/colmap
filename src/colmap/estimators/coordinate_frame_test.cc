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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/estimators/coordinate_frame.h"

#include "colmap/geometry/gps.h"
#include "colmap/scene/projection.h"

#include <gtest/gtest.h>

namespace colmap {

TEST(CoordinateFrame, EstimateGravityVectorFromImageOrientation) {
  Reconstruction reconstruction;
  EXPECT_EQ(EstimateGravityVectorFromImageOrientation(reconstruction),
            Eigen::Vector3d::Zero());
}

TEST(CoordinateFrame, EstimateManhattanWorldFrame) {
  Reconstruction reconstruction;
  std::string image_path;
  EXPECT_EQ(
      EstimateManhattanWorldFrame(
          ManhattanWorldFrameEstimationOptions(), reconstruction, image_path),
      Eigen::Matrix3d::Zero());
}

TEST(CoordinateFrame, AlignToPrincipalPlane) {
  // Start with reconstruction containing points on the Y-Z plane and cameras
  // "above" the plane on the positive X axis. After alignment the points should
  // be on the X-Y plane and the cameras "above" the plane on the positive Z
  // axis.
  Sim3d tform;
  Reconstruction reconstruction;
  // Setup image with projection center at (1, 0, 0)
  Image image;
  image.SetImageId(1);
  image.SetRegistered(true);
  image.CamFromWorld() =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(-1, 0, 0));
  reconstruction.AddImage(image);
  // Setup 4 points on the Y-Z plane
  const point3D_t p1 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, -1, 0), Track());
  const point3D_t p2 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 1, 0), Track());
  const point3D_t p3 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, -1), Track());
  const point3D_t p4 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0, 0, 1), Track());
  AlignToPrincipalPlane(&reconstruction, &tform);
  // Note that the final X and Y axes may be inverted after alignment, so we
  // need to account for both cases when checking for correctness
  const bool inverted = tform.rotation.y() < 0;

  // Verify that points lie on the correct locations of the X-Y plane
  EXPECT_LE((reconstruction.Point3D(p1).XYZ() -
             Eigen::Vector3d(inverted ? 1 : -1, 0, 0))
                .norm(),
            1e-6);
  EXPECT_LE((reconstruction.Point3D(p2).XYZ() -
             Eigen::Vector3d(inverted ? -1 : 1, 0, 0))
                .norm(),
            1e-6);
  EXPECT_LE((reconstruction.Point3D(p3).XYZ() -
             Eigen::Vector3d(0, inverted ? 1 : -1, 0))
                .norm(),
            1e-6);
  EXPECT_LE((reconstruction.Point3D(p4).XYZ() -
             Eigen::Vector3d(0, inverted ? -1 : 1, 0))
                .norm(),
            1e-6);
  // Verify that projection center is at (0, 0, 1)
  EXPECT_LE(
      (reconstruction.Image(1).ProjectionCenter() - Eigen::Vector3d(0, 0, 1))
          .norm(),
      1e-6);
  // Verify that transform matrix does shuffling of axes
  Eigen::Matrix3x4d expected;
  if (inverted) {
    expected << 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0;
  } else {
    expected << 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0;
  }
  EXPECT_LT((tform.ToMatrix() - expected).norm(), 1e-6);
}

TEST(CoordinateFrame, AlignToENUPlane) {
  // Create reconstruction with 4 points with known LLA coordinates. After the
  // ENU transform all 4 points should land approximately on the X-Y plane.
  GPSTransform gps;
  auto points = gps.EllToXYZ({Eigen::Vector3d(50, 10.1, 100),
                              Eigen::Vector3d(50.1, 10, 100),
                              Eigen::Vector3d(50.1, 10.1, 100),
                              Eigen::Vector3d(50, 10, 100)});
  Sim3d tform;
  Reconstruction reconstruction;
  std::vector<point3D_t> point_ids;
  for (size_t i = 0; i < points.size(); ++i) {
    point_ids.push_back(reconstruction.AddPoint3D(points[i], Track()));
    std::cout << points[i].transpose() << std::endl;
  }
  AlignToENUPlane(&reconstruction, &tform, false);
  // Verify final locations of points
  EXPECT_LE((reconstruction.Point3D(point_ids[0]).XYZ() -
             Eigen::Vector3d(3584.8565215, -5561.5336506, 0.0742643))
                .norm(),
            1e-6);
  EXPECT_LE((reconstruction.Point3D(point_ids[1]).XYZ() -
             Eigen::Vector3d(-3577.3888622, 5561.6397107, 0.0783761))
                .norm(),
            1e-6);
  EXPECT_LE((reconstruction.Point3D(point_ids[2]).XYZ() -
             Eigen::Vector3d(3577.4152111, 5561.6397283, 0.0783613))
                .norm(),
            1e-6);
  EXPECT_LE((reconstruction.Point3D(point_ids[3]).XYZ() -
             Eigen::Vector3d(-3584.8301178, -5561.5336683, 0.0742791))
                .norm(),
            1e-6);

  // Verify that straight line distance between points is preserved
  for (size_t i = 1; i < points.size(); ++i) {
    const double dist_orig = (points[i] - points[i - 1]).norm();
    const double dist_tform = (reconstruction.Point3D(point_ids[i]).XYZ() -
                               reconstruction.Point3D(point_ids[i - 1]).XYZ())
                                  .norm();
    EXPECT_LE(std::abs(dist_orig - dist_tform), 1e-6);
  }
}

}  // namespace colmap
