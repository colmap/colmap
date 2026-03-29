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
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/eigen_matchers.h"
#include "colmap/util/testing.h"

#include <array>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(EstimateGravityVectorFromImageOrientation, Nominal) {
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

TEST(EstimateGravityVectorFromImageOrientation, Empty) {
  Reconstruction reconstruction;
  EXPECT_EQ(EstimateGravityVectorFromImageOrientation(reconstruction),
            Eigen::Vector3d::Zero());
}

#ifdef COLMAP_LSD_ENABLED

constexpr int kWidth = 512;
constexpr int kHeight = 512;
constexpr double kFocal = 400.0;

struct Line3D {
  Eigen::Vector3d beg;
  Eigen::Vector3d end;
};

struct ManhattanScene {
  Eigen::Matrix3d manhattan_from_world;
  std::vector<Line3D> vertical_lines;
  std::vector<Line3D> horizontal_lines;
};

ManhattanScene CreateManhattanScene() {
  // Rotate the Manhattan frame so the solution is not trivially axis-aligned.
  const Eigen::Matrix3d manhattan_from_world =
      (Eigen::AngleAxisd(DegToRad(25.0), Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(DegToRad(11.0), Eigen::Vector3d::UnitX()))
          .toRotationMatrix();

  // Define axis-aligned 3D lines, then rotate them into the tilted frame.
  // Vertical lines are parallel to Y; horizontal lines to X.
  std::vector<Line3D> vertical_lines = {
      {{-1.5, -2, 5}, {-1.5, 2, 5}},
      {{-0.5, -2, 5}, {-0.5, 2, 5}},
      {{0.5, -2, 5}, {0.5, 2, 5}},
      {{1.5, -2, 5}, {1.5, 2, 5}},
      {{-1, -2, 7}, {-1, 2, 7}},
      {{0, -2, 7}, {0, 2, 7}},
      {{1, -2, 7}, {1, 2, 7}},
  };

  std::vector<Line3D> horizontal_lines = {
      {{-2, -1.5, 5}, {2, -1.5, 5}},
      {{-2, -0.5, 5}, {2, -0.5, 5}},
      {{-2, 0.5, 5}, {2, 0.5, 5}},
      {{-2, 1.5, 5}, {2, 1.5, 5}},
      {{-2, -1, 7}, {2, -1, 7}},
      {{-2, 0, 7}, {2, 0, 7}},
      {{-2, 1, 7}, {2, 1, 7}},
  };

  for (auto& l : vertical_lines) {
    l.beg = manhattan_from_world * l.beg;
    l.end = manhattan_from_world * l.end;
  }
  for (auto& l : horizontal_lines) {
    l.beg = manhattan_from_world * l.beg;
    l.end = manhattan_from_world * l.end;
  }

  return {manhattan_from_world,
          std::move(vertical_lines),
          std::move(horizontal_lines)};
}

void DrawLine(Bitmap& bitmap,
              const Eigen::Vector2d& a,
              const Eigen::Vector2d& b,
              int radius) {
  const int steps =
      std::max(1, static_cast<int>(std::ceil((b - a).norm() * 2)));
  for (int s = 0; s <= steps; ++s) {
    const double t = static_cast<double>(s) / steps;
    const double x = a.x() + t * (b.x() - a.x());
    const double y = a.y() + t * (b.y() - a.y());
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        if (dx * dx + dy * dy > radius * radius) continue;
        const int px = static_cast<int>(std::round(x + dx));
        const int py = static_cast<int>(std::round(y + dy));
        if (px >= 0 && px < kWidth && py >= 0 && py < kHeight) {
          bitmap.SetPixel(px, py, BitmapColor<uint8_t>(255));
        }
      }
    }
  }
}

void RenderLineImages(Reconstruction& reconstruction,
                      const ManhattanScene& scene,
                      const std::filesystem::path& test_dir) {
  // 3 cameras with combined pitch (around X) and yaw (around Y) so that both
  // the horizontal and vertical vanishing points are at finite image locations.
  const std::array<double, 3> pitch_deg = {10.0, 10.0, 8.0};
  const std::array<double, 3> yaw_deg = {15.0, -15.0, 5.0};

  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, kFocal, kWidth, kHeight);
  reconstruction.AddCameraWithTrivialRig(camera);

  for (size_t cam_idx = 0; cam_idx < pitch_deg.size(); ++cam_idx) {
    const Eigen::Quaterniond cam_from_world_rot(
        Eigen::AngleAxisd(DegToRad(pitch_deg[cam_idx]),
                          Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(DegToRad(yaw_deg[cam_idx]),
                          Eigen::Vector3d::UnitY()));
    const Rigid3d cam_from_world(cam_from_world_rot, Eigen::Vector3d::Zero());

    Image image;
    image.SetImageId(cam_idx);
    image.SetCameraId(camera.camera_id);
    image.SetName("image" + std::to_string(cam_idx) + ".png");
    reconstruction.AddImageWithTrivialFrame(std::move(image), cam_from_world);

    const auto& reg_image = reconstruction.Image(cam_idx);

    // Project 3D lines and draw thick white stripes on a gray background.
    Bitmap bitmap(kWidth, kHeight, /*as_rgb=*/true);
    bitmap.Fill(BitmapColor<uint8_t>(128));

    constexpr int kRadius = 3;

    for (const auto& line : scene.vertical_lines) {
      const auto p1 = reg_image.ProjectPoint(line.beg);
      const auto p2 = reg_image.ProjectPoint(line.end);
      if (p1 && p2) DrawLine(bitmap, *p1, *p2, kRadius);
    }
    for (const auto& line : scene.horizontal_lines) {
      const auto p1 = reg_image.ProjectPoint(line.beg);
      const auto p2 = reg_image.ProjectPoint(line.end);
      if (p1 && p2) DrawLine(bitmap, *p1, *p2, kRadius);
    }

    ASSERT_TRUE(bitmap.Write(test_dir / reg_image.Name()));
  }
}

TEST(EstimateManhattanWorldFrame, Synthetic) {
  SetPRNGSeed(0);

  const auto scene = CreateManhattanScene();
  const auto test_dir = CreateTestDir();

  Reconstruction reconstruction;
  ASSERT_NO_FATAL_FAILURE(RenderLineImages(reconstruction, scene, test_dir));

  // Run Manhattan world frame estimation.
  ManhattanWorldFrameEstimationOptions options;
  const Eigen::Matrix3d frame =
      EstimateManhattanWorldFrame(options, reconstruction, test_dir);

  // The estimated frame should recover the tilted Manhattan axes.
  const Eigen::Vector3d expected_rightward = scene.manhattan_from_world.col(0);
  const Eigen::Vector3d expected_downward = scene.manhattan_from_world.col(1);
  const Eigen::Vector3d expected_forward = scene.manhattan_from_world.col(2);

  // Rightward direction (col 0) must align with the rotated X axis.
  // The sign of the rightward and forward axes is ambiguous, so check absolute
  // dot product.
  EXPECT_LT(std::abs(std::abs(frame.col(0).dot(expected_rightward)) - 1), 1e-3);
  // Gravity direction (col 1) must align with the rotated Y axis.
  // The sign is deterministic (flipped to match +Y in the implementation).
  EXPECT_LT(std::abs(frame.col(1).dot(expected_downward) - 1), 1e-3);
  // Forward direction (col 2) must align with the rotated Z axis.
  EXPECT_LT(std::abs(std::abs(frame.col(2).dot(expected_forward)) - 1), 1e-3);

  // Verify orthonormality.
  EXPECT_NEAR(frame.col(0).norm(), 1.0, 1e-6);
  EXPECT_NEAR(frame.col(1).norm(), 1.0, 1e-6);
  EXPECT_NEAR(frame.col(2).norm(), 1.0, 1e-6);
  EXPECT_NEAR(std::abs(frame.col(0).dot(frame.col(1))), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(frame.col(0).dot(frame.col(2))), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(frame.col(1).dot(frame.col(2))), 0.0, 1e-6);
}

TEST(EstimateManhattanWorldFrame, Empty) {
  Reconstruction reconstruction;
  std::filesystem::path image_path;
  EXPECT_EQ(
      EstimateManhattanWorldFrame(
          ManhattanWorldFrameEstimationOptions(), reconstruction, image_path),
      Eigen::Matrix3d::Zero());
}

#endif

TEST(AlignToPrincipalPlane, Nominal) {
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

TEST(AlignToENUPlane, Scaled) {
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

TEST(AlignToENUPlane, Unscaled) {
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
