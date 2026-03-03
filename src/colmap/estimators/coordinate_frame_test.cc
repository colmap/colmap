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

#include "colmap/estimators/coordinate_frame.h"

#include "colmap/geometry/gps.h"
#include "colmap/math/random.h"
#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CoordinateFrame, EstimateGravityVectorFromImageOrientation) {
  Reconstruction reconstruction;
  EXPECT_EQ(EstimateGravityVectorFromImageOrientation(reconstruction),
            Eigen::Vector3d::Zero());
}

TEST(CoordinateFrame, EstimateGravityVectorFromUprightImages) {
  // Create a reconstruction with multiple upright images.
  // Upright images have gravity aligned with the camera Y axis, so
  // row(1) of the rotation matrix should point downward (along +Y world).
  Reconstruction reconstruction;
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera);

  // Add 5 registered images with random rotation around +Y (upright, gravity
  // ~ +Y)
  for (int i = 0; i < 5; ++i) {
    Image image;
    image.SetImageId(i);
    image.SetCameraId(camera.camera_id);
    const Rigid3d cam_from_world(
        Eigen::Quaterniond(
            Eigen::AngleAxisd(RandomUniformReal<double>(-EIGEN_PI, EIGEN_PI),
                              Eigen::Vector3d::UnitY())),
        Eigen::Vector3d(i, 0, 0));
    reconstruction.AddImageWithTrivialFrame(std::move(image), cam_from_world);
  }

  // 1 outlier image rotated 90 degrees around Z
  {
    Image image;
    image.SetImageId(6);
    image.SetCameraId(camera.camera_id);
    const Rigid3d cam_from_world(Eigen::Quaterniond(Eigen::AngleAxisd(
                                     EIGEN_PI / 2, Eigen::Vector3d::UnitZ())),
                                 Eigen::Vector3d(6, 0, 0));
    reconstruction.AddImageWithTrivialFrame(std::move(image), cam_from_world);
  }

  const Eigen::Vector3d gravity =
      EstimateGravityVectorFromImageOrientation(reconstruction);
  // Consensus should find gravity ~ (0, 1, 0) despite the outlier
  EXPECT_NEAR(gravity.norm(), 1.0, 1e-6);
  EXPECT_NEAR(std::abs(gravity.dot(Eigen::Vector3d(0, 1, 0))), 1.0, 1e-6);
}

#ifdef COLMAP_LSD_ENABLED
TEST(CoordinateFrame, EstimateManhattanWorldFrame) {
  Reconstruction reconstruction;
  std::filesystem::path image_path;
  EXPECT_EQ(
      EstimateManhattanWorldFrame(
          ManhattanWorldFrameEstimationOptions(), reconstruction, image_path),
      Eigen::Matrix3d::Zero());
}
#endif

TEST(CoordinateFrame, AlignToPrincipalPlane) {
  // Start with reconstruction containing points on the Y-Z plane and cameras
  // "above" the plane on the positive X axis. After alignment the points should
  // be on the X-Y plane and the cameras "above" the plane on the positive Z
  // axis.
  Sim3d tform;
  Reconstruction reconstruction;
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  reconstruction.AddCamera(camera);
  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, 1));
  reconstruction.AddRig(rig);
  Frame frame;
  frame.SetFrameId(1);
  frame.SetRigId(rig.RigId());
  frame.AddDataId(data_t(camera.SensorId(), 1));
  frame.SetRigFromWorld(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(-1, 0, 0)));
  reconstruction.AddFrame(frame);
  // Setup image with projection center at (1, 0, 0)
  Image image;
  image.SetCameraId(camera.camera_id);
  image.SetImageId(1);
  image.SetFrameId(1);
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
  const bool inverted = tform.rotation().y() < 0;

  // Verify that points lie on the correct locations of the X-Y plane
  EXPECT_LE((reconstruction.Point3D(p1).xyz -
             Eigen::Vector3d(inverted ? 1 : -1, 0, 0))
                .norm(),
            1e-6);
  EXPECT_LE((reconstruction.Point3D(p2).xyz -
             Eigen::Vector3d(inverted ? -1 : 1, 0, 0))
                .norm(),
            1e-6);
  EXPECT_LE((reconstruction.Point3D(p3).xyz -
             Eigen::Vector3d(0, inverted ? 1 : -1, 0))
                .norm(),
            1e-6);
  EXPECT_LE((reconstruction.Point3D(p4).xyz -
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
  EXPECT_THAT(tform.ToMatrix(), EigenMatrixNear(expected, 1e-6));
}

TEST(CoordinateFrame, AlignToENUPlane) {
  // Create reconstruction with 4 points with known LLA coordinates. After the
  // ENU transform all 4 points should land approximately on the X-Y plane.
  GPSTransform gps;
  auto points = gps.EllipsoidToECEF({Eigen::Vector3d(50, 10.1, 100),
                                     Eigen::Vector3d(50.1, 10, 100),
                                     Eigen::Vector3d(50.1, 10.1, 100),
                                     Eigen::Vector3d(50, 10, 100)});
  Sim3d tform;
  Reconstruction reconstruction;
  std::vector<point3D_t> point_ids;
  for (size_t i = 0; i < points.size(); ++i) {
    point_ids.push_back(reconstruction.AddPoint3D(points[i], Track()));
    LOG(INFO) << points[i].transpose();
  }
  AlignToENUPlane(&reconstruction, &tform, false);
  // Verify final locations of points
  EXPECT_THAT(reconstruction.Point3D(point_ids[0]).xyz,
              EigenMatrixNear(Eigen::Vector3d(3584.8433196335045,
                                              -5561.5866894473402,
                                              -0.0020947810262441635),
                              1e-6));
  EXPECT_THAT(reconstruction.Point3D(point_ids[1]).xyz,
              EigenMatrixNear(Eigen::Vector3d(-3577.4020366631503,
                                              5561.5866894469982,
                                              0.0020947791635990143),
                              1e-6));
  EXPECT_THAT(reconstruction.Point3D(point_ids[2]).xyz,
              EigenMatrixNear(Eigen::Vector3d(3577.4020366640707,
                                              5561.5866894467654,
                                              0.0020947791635990143),
                              1e-6));
  EXPECT_THAT(reconstruction.Point3D(point_ids[3]).xyz,
              EigenMatrixNear(Eigen::Vector3d(-3584.8433196330498,
                                              -5561.586689447573,
                                              -0.0020947810262441635),
                              1e-6));

  // Verify that straight line distance between points is preserved
  for (size_t i = 1; i < points.size(); ++i) {
    const double dist_orig = (points[i] - points[i - 1]).norm();
    const double dist_tform = (reconstruction.Point3D(point_ids[i]).xyz -
                               reconstruction.Point3D(point_ids[i - 1]).xyz)
                                  .norm();
    EXPECT_LE(std::abs(dist_orig - dist_tform), 1e-6);
  }
}

TEST(CoordinateFrame, AlignToENUPlaneUnscaled) {
  // Test unscaled variant: starting from a model with non-unit scale, the
  // alignment should undo the scale so that original distances are preserved.
  GPSTransform gps;
  auto points = gps.EllipsoidToECEF({Eigen::Vector3d(50, 10.1, 100),
                                     Eigen::Vector3d(50.1, 10, 100),
                                     Eigen::Vector3d(50.1, 10.1, 100),
                                     Eigen::Vector3d(50, 10, 100)});

  Reconstruction reconstruction;
  std::vector<point3D_t> point3D_ids;
  point3D_ids.reserve(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    point3D_ids.push_back(reconstruction.AddPoint3D(points[i], Track()));
  }

  // Apply a non-unit scale to simulate a scaled model
  const double model_scale = 2.0;
  Sim3d pre_scale(
      model_scale, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  reconstruction.Transform(pre_scale);

  // Record scaled distances for comparison
  std::vector<double> scaled_dists;
  for (size_t i = 1; i < points.size(); ++i) {
    scaled_dists.push_back((reconstruction.Point3D(point3D_ids[i]).xyz -
                            reconstruction.Point3D(point3D_ids[i - 1]).xyz)
                               .norm());
  }

  // Align with unscaled=true, passing the pre_scale as the current transform
  Sim3d tform = pre_scale;
  AlignToENUPlane(&reconstruction, &tform, /*unscaled=*/true);

  // The applied transform should have inverse scale to undo pre_scale
  EXPECT_NEAR(tform.scale(), 1.0 / model_scale, 1e-6);

  // Original (unscaled) distances between ECEF points should be preserved
  for (size_t i = 1; i < points.size(); ++i) {
    const double dist_orig = (points[i] - points[i - 1]).norm();
    const double dist_tform = (reconstruction.Point3D(point3D_ids[i]).xyz -
                               reconstruction.Point3D(point3D_ids[i - 1]).xyz)
                                  .norm();
    EXPECT_NEAR(dist_orig, dist_tform, 1e-4);
  }
}

}  // namespace
}  // namespace colmap
