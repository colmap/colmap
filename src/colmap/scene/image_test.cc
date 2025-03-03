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

#include "colmap/scene/image.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Image, Default) {
  Image image;
  EXPECT_EQ(image.ImageId(), kInvalidImageId);
  EXPECT_EQ(image.Name(), "");
  EXPECT_EQ(image.CameraId(), kInvalidCameraId);
  EXPECT_FALSE(image.HasCameraId());
  EXPECT_FALSE(image.HasCameraPtr());
  EXPECT_FALSE(image.HasPose());
  EXPECT_EQ(image.NumPoints2D(), 0);
  EXPECT_EQ(image.NumPoints3D(), 0);
  EXPECT_EQ(image.Points2D().size(), 0);
}

TEST(Image, Equals) {
  Image image;
  Image other = image;
  EXPECT_EQ(image, other);
  image.SetName("test");
  EXPECT_NE(image, other);
  other.SetName("test");
  EXPECT_EQ(image, other);
}

TEST(Image, Print) {
  Image image;
  image.SetImageId(1);
  image.SetCameraId(2);
  image.SetName("test");
  std::ostringstream stream;
  stream << image;
  EXPECT_EQ(stream.str(),
            "Image(image_id=1, camera_id=2, name=\"test\", "
            "has_pose=0, triangulated=0/0)");
}

TEST(Image, ImageId) {
  Image image;
  EXPECT_EQ(image.ImageId(), kInvalidImageId);
  image.SetImageId(1);
  EXPECT_EQ(image.ImageId(), 1);
}

TEST(Image, Name) {
  Image image;
  EXPECT_EQ(image.Name(), "");
  image.SetName("test1");
  EXPECT_EQ(image.Name(), "test1");
  image.Name() = "test2";
  EXPECT_EQ(image.Name(), "test2");
}

TEST(Image, CameraId) {
  Image image;
  EXPECT_EQ(image.CameraId(), kInvalidCameraId);
  image.SetCameraId(1);
  EXPECT_EQ(image.CameraId(), 1);
}

TEST(Image, CameraPtr) {
  Image image;
  EXPECT_FALSE(image.HasCameraPtr());
  EXPECT_ANY_THROW(image.CameraPtr());
  Camera camera;
  camera.camera_id = 1;
  EXPECT_ANY_THROW(image.SetCameraPtr(&camera));
  image.SetCameraId(2);
  EXPECT_ANY_THROW(image.SetCameraPtr(&camera));
  image.SetCameraId(1);
  image.SetCameraPtr(&camera);
  EXPECT_TRUE(image.HasCameraPtr());
  EXPECT_EQ(image.CameraPtr(), &camera);
  image.ResetCameraPtr();
  EXPECT_FALSE(image.HasCameraPtr());
  EXPECT_ANY_THROW(image.CameraPtr());
}

TEST(Image, SetResetPose) {
  Image image;
  EXPECT_FALSE(image.HasPose());
  EXPECT_ANY_THROW(image.CamFromWorld());
  image.SetCamFromWorld(Rigid3d());
  EXPECT_TRUE(image.HasPose());
  EXPECT_EQ(image.CamFromWorld().rotation.coeffs(),
            Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(image.CamFromWorld().translation, Eigen::Vector3d::Zero());
  image.ResetPose();
  EXPECT_FALSE(image.HasPose());
  EXPECT_ANY_THROW(image.CamFromWorld());
}

TEST(Image, ConstructCopy) {
  Image image;
  image.SetCamFromWorld(Rigid3d());
  Image image_copy = Image(image);
  EXPECT_EQ(image, image_copy);
  EXPECT_EQ(Rigid3d(), image_copy.CamFromWorld());
  image_copy.ResetPose();
  EXPECT_TRUE(image.HasPose());
  EXPECT_FALSE(image_copy.HasPose());
}

TEST(Image, AssignCopy) {
  Image image;
  image.SetCamFromWorld(Rigid3d());
  Image image_copy = image;
  EXPECT_EQ(image, image_copy);
  EXPECT_EQ(Rigid3d(), image_copy.CamFromWorld());
  image_copy.ResetPose();
  EXPECT_TRUE(image.HasPose());
  EXPECT_FALSE(image_copy.HasPose());
}

TEST(Image, NumPoints2D) {
  Image image;
  EXPECT_EQ(image.NumPoints2D(), 0);
  image.SetPoints2D(std::vector<Eigen::Vector2d>(10));
  EXPECT_EQ(image.NumPoints2D(), 10);
}

TEST(Image, NumPoints3D) {
  Image image;
  image.SetPoints2D(std::vector<Eigen::Vector2d>(10));
  EXPECT_EQ(image.NumPoints3D(), 0);
  image.SetPoint3DForPoint2D(0, 0);
  EXPECT_EQ(image.NumPoints3D(), 1);
  image.SetPoint3DForPoint2D(0, 1);
  image.SetPoint3DForPoint2D(1, 2);
  EXPECT_EQ(image.NumPoints3D(), 2);
}

TEST(Image, Points2D) {
  Image image;
  EXPECT_EQ(image.Points2D().size(), 0);
  std::vector<Eigen::Vector2d> points2D(10);
  points2D[0] = Eigen::Vector2d(1.0, 2.0);
  image.SetPoints2D(points2D);
  EXPECT_EQ(image.Points2D().size(), 10);
  EXPECT_EQ(image.Point2D(0).xy(0), 1.0);
  EXPECT_EQ(image.Point2D(0).xy(1), 2.0);
  EXPECT_EQ(image.NumPoints3D(), 0);
}

TEST(Image, Points2DWith3D) {
  Image image;
  EXPECT_EQ(image.Points2D().size(), 0);
  std::vector<Point2D> points2D(10);
  points2D[0].xy = Eigen::Vector2d(1.0, 2.0);
  points2D[0].point3D_id = 1;
  image.SetPoints2D(points2D);
  EXPECT_EQ(image.Points2D().size(), 10);
  EXPECT_EQ(image.Point2D(0).xy(0), 1.0);
  EXPECT_EQ(image.Point2D(0).xy(1), 2.0);
  EXPECT_EQ(image.NumPoints3D(), 1);
}

TEST(Image, Points3D) {
  Image image;
  image.SetPoints2D(std::vector<Eigen::Vector2d>(2));
  EXPECT_FALSE(image.Point2D(0).HasPoint3D());
  EXPECT_FALSE(image.Point2D(1).HasPoint3D());
  EXPECT_EQ(image.NumPoints3D(), 0);
  image.SetPoint3DForPoint2D(0, 0);
  EXPECT_TRUE(image.Point2D(0).HasPoint3D());
  EXPECT_FALSE(image.Point2D(1).HasPoint3D());
  EXPECT_EQ(image.NumPoints3D(), 1);
  EXPECT_TRUE(image.HasPoint3D(0));
  image.SetPoint3DForPoint2D(0, 1);
  EXPECT_TRUE(image.Point2D(0).HasPoint3D());
  EXPECT_FALSE(image.Point2D(1).HasPoint3D());
  EXPECT_EQ(image.NumPoints3D(), 1);
  EXPECT_FALSE(image.HasPoint3D(0));
  EXPECT_TRUE(image.HasPoint3D(1));
  image.SetPoint3DForPoint2D(1, 0);
  EXPECT_TRUE(image.Point2D(0).HasPoint3D());
  EXPECT_TRUE(image.Point2D(1).HasPoint3D());
  EXPECT_EQ(image.NumPoints3D(), 2);
  EXPECT_TRUE(image.HasPoint3D(0));
  EXPECT_TRUE(image.HasPoint3D(1));
  image.ResetPoint3DForPoint2D(0);
  EXPECT_FALSE(image.Point2D(0).HasPoint3D());
  EXPECT_TRUE(image.Point2D(1).HasPoint3D());
  EXPECT_EQ(image.NumPoints3D(), 1);
  EXPECT_TRUE(image.HasPoint3D(0));
  EXPECT_FALSE(image.HasPoint3D(1));
  image.ResetPoint3DForPoint2D(1);
  EXPECT_FALSE(image.Point2D(0).HasPoint3D());
  EXPECT_FALSE(image.Point2D(1).HasPoint3D());
  EXPECT_EQ(image.NumPoints3D(), 0);
  EXPECT_FALSE(image.HasPoint3D(0));
  EXPECT_FALSE(image.HasPoint3D(1));
  image.ResetPoint3DForPoint2D(0);
  EXPECT_FALSE(image.Point2D(0).HasPoint3D());
  EXPECT_FALSE(image.Point2D(1).HasPoint3D());
  EXPECT_EQ(image.NumPoints3D(), 0);
  EXPECT_FALSE(image.HasPoint3D(0));
  EXPECT_FALSE(image.HasPoint3D(1));
}

TEST(Image, ProjectionCenter) {
  Image image;
  image.SetCamFromWorld(Rigid3d());
  EXPECT_EQ(image.ProjectionCenter(), Eigen::Vector3d::Zero());
}

TEST(Image, ViewingDirection) {
  Image image;
  image.SetCamFromWorld(Rigid3d());
  EXPECT_EQ(image.ViewingDirection(), Eigen::Vector3d(0, 0, 1));
}

TEST(Image, ProjectPoint) {
  Image image;
  image.SetCamFromWorld(Rigid3d());
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1, 1, 1);
  image.SetCameraId(camera.camera_id);
  image.SetCameraPtr(&camera);
  const std::optional<Eigen::Vector2d> result =
      image.ProjectPoint(Eigen::Vector3d(2, 0, 1));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), Eigen::Vector2d(2.5, 0.5));
  EXPECT_FALSE(image.ProjectPoint(Eigen::Vector3d(2, 0, 0)).has_value());
  EXPECT_FALSE(image.ProjectPoint(Eigen::Vector3d(2, 0, -1)).has_value());
}

}  // namespace
}  // namespace colmap
