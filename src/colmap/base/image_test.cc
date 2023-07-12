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

#include "colmap/base/image.h"

#include <gtest/gtest.h>

namespace colmap {

TEST(Image, Default) {
  Image image;
  EXPECT_EQ(image.ImageId(), kInvalidImageId);
  EXPECT_EQ(image.Name(), "");
  EXPECT_EQ(image.CameraId(), kInvalidCameraId);
  EXPECT_FALSE(image.HasCamera());
  EXPECT_FALSE(image.IsRegistered());
  EXPECT_EQ(image.NumPoints2D(), 0);
  EXPECT_EQ(image.NumPoints3D(), 0);
  EXPECT_EQ(image.NumObservations(), 0);
  EXPECT_EQ(image.NumCorrespondences(), 0);
  EXPECT_EQ(image.NumVisiblePoints3D(), 0);
  EXPECT_EQ(image.Point3DVisibilityScore(), 0);
  EXPECT_EQ(image.Qvec(0), 1.0);
  EXPECT_EQ(image.Qvec(1), 0.0);
  EXPECT_EQ(image.Qvec(2), 0.0);
  EXPECT_EQ(image.Qvec(3), 0.0);
  EXPECT_TRUE(image.QvecPrior().array().isNaN().all());
  EXPECT_EQ(image.Tvec(0), 0.0);
  EXPECT_EQ(image.Tvec(1), 0.0);
  EXPECT_EQ(image.Tvec(2), 0.0);
  EXPECT_TRUE(image.TvecPrior().array().isNaN().all());
  EXPECT_FALSE(image.HasQvecPrior());
  EXPECT_FALSE(image.HasTvecPrior());
  EXPECT_EQ(image.Points2D().size(), 0);
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

TEST(Image, Registered) {
  Image image;
  EXPECT_FALSE(image.IsRegistered());
  image.SetRegistered(true);
  EXPECT_TRUE(image.IsRegistered());
  image.SetRegistered(false);
  EXPECT_FALSE(image.IsRegistered());
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

TEST(Image, NumObservations) {
  Image image;
  EXPECT_EQ(image.NumObservations(), 0);
  image.SetNumObservations(10);
  EXPECT_EQ(image.NumObservations(), 10);
}

TEST(Image, NumCorrespondences) {
  Image image;
  EXPECT_EQ(image.NumCorrespondences(), 0);
  image.SetNumCorrespondences(10);
  EXPECT_EQ(image.NumCorrespondences(), 10);
}

TEST(Image, NumVisiblePoints3D) {
  Image image;
  image.SetPoints2D(std::vector<Eigen::Vector2d>(10));
  image.SetNumObservations(10);
  Camera camera;
  camera.SetWidth(10);
  camera.SetHeight(10);
  image.SetUp(camera);
  EXPECT_EQ(image.NumVisiblePoints3D(), 0);
  image.IncrementCorrespondenceHasPoint3D(0);
  EXPECT_EQ(image.NumVisiblePoints3D(), 1);
  image.IncrementCorrespondenceHasPoint3D(0);
  image.IncrementCorrespondenceHasPoint3D(1);
  EXPECT_EQ(image.NumVisiblePoints3D(), 2);
  image.DecrementCorrespondenceHasPoint3D(0);
  EXPECT_EQ(image.NumVisiblePoints3D(), 2);
  image.DecrementCorrespondenceHasPoint3D(0);
  EXPECT_EQ(image.NumVisiblePoints3D(), 1);
  image.DecrementCorrespondenceHasPoint3D(1);
  EXPECT_EQ(image.NumVisiblePoints3D(), 0);
}

TEST(Image, Point3DVisibilityScore) {
  Image image;
  std::vector<Eigen::Vector2d> points2D;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      points2D.emplace_back(i, j);
    }
  }
  image.SetPoints2D(points2D);
  image.SetNumObservations(16);
  Camera camera;
  camera.SetWidth(4);
  camera.SetHeight(4);
  image.SetUp(camera);
  Eigen::Matrix<size_t, Eigen::Dynamic, 1> scores(
      image.kNumPoint3DVisibilityPyramidLevels, 1);
  for (int i = 1; i <= image.kNumPoint3DVisibilityPyramidLevels; ++i) {
    scores(i - 1) = (1 << i) * (1 << i);
  }
  EXPECT_EQ(image.Point3DVisibilityScore(), 0);
  image.IncrementCorrespondenceHasPoint3D(0);
  EXPECT_EQ(image.Point3DVisibilityScore(), scores.sum());
  image.IncrementCorrespondenceHasPoint3D(0);
  EXPECT_EQ(image.Point3DVisibilityScore(), scores.sum());
  image.IncrementCorrespondenceHasPoint3D(1);
  EXPECT_EQ(image.Point3DVisibilityScore(),
            scores.sum() + scores.bottomRows(scores.size() - 1).sum());
  image.IncrementCorrespondenceHasPoint3D(1);
  image.IncrementCorrespondenceHasPoint3D(1);
  image.IncrementCorrespondenceHasPoint3D(4);
  EXPECT_EQ(image.Point3DVisibilityScore(),
            scores.sum() + 2 * scores.bottomRows(scores.size() - 1).sum());
  image.IncrementCorrespondenceHasPoint3D(4);
  image.IncrementCorrespondenceHasPoint3D(5);
  EXPECT_EQ(image.Point3DVisibilityScore(),
            scores.sum() + 3 * scores.bottomRows(scores.size() - 1).sum());
  image.DecrementCorrespondenceHasPoint3D(0);
  EXPECT_EQ(image.Point3DVisibilityScore(),
            scores.sum() + 3 * scores.bottomRows(scores.size() - 1).sum());
  image.DecrementCorrespondenceHasPoint3D(0);
  EXPECT_EQ(image.Point3DVisibilityScore(),
            scores.sum() + 2 * scores.bottomRows(scores.size() - 1).sum());
  image.IncrementCorrespondenceHasPoint3D(2);
  EXPECT_EQ(image.Point3DVisibilityScore(),
            2 * scores.sum() + 2 * scores.bottomRows(scores.size() - 1).sum());
}

TEST(Image, Qvec) {
  Image image;
  EXPECT_EQ(image.Qvec(0), 1.0);
  EXPECT_EQ(image.Qvec(1), 0.0);
  EXPECT_EQ(image.Qvec(2), 0.0);
  EXPECT_EQ(image.Qvec(3), 0.0);
  image.Qvec(0) = 2.0;
  EXPECT_EQ(image.Qvec(0), 2.0);
  image.SetQvec(Eigen::Vector4d(3.0, 0.0, 0.0, 0.0));
  EXPECT_EQ(image.Qvec(0), 3.0);
  image.Qvec() = Eigen::Vector4d(4.0, 0.0, 0.0, 0.0);
  EXPECT_EQ(image.Qvec(0), 4.0);
}

TEST(Image, QvecPrior) {
  Image image;
  EXPECT_TRUE(image.QvecPrior().array().isNaN().all());
  EXPECT_FALSE(image.HasQvecPrior());
  image.QvecPrior(0) = 2.0;
  EXPECT_FALSE(image.HasQvecPrior());
  image.QvecPrior(1) = 2.0;
  EXPECT_FALSE(image.HasQvecPrior());
  image.QvecPrior(2) = 2.0;
  EXPECT_FALSE(image.HasQvecPrior());
  image.QvecPrior(3) = 2.0;
  EXPECT_TRUE(image.HasQvecPrior());
  EXPECT_EQ(image.QvecPrior(0), 2.0);
  EXPECT_EQ(image.QvecPrior(1), 2.0);
  EXPECT_EQ(image.QvecPrior(2), 2.0);
  EXPECT_EQ(image.QvecPrior(3), 2.0);
  image.SetQvecPrior(Eigen::Vector4d(3.0, 0.0, 0.0, 0.0));
  EXPECT_EQ(image.QvecPrior(0), 3.0);
  image.QvecPrior() = Eigen::Vector4d(4.0, 0.0, 0.0, 0.0);
  EXPECT_EQ(image.QvecPrior(0), 4.0);
}

TEST(Image, Tvec) {
  Image image;
  EXPECT_EQ(image.Tvec(0), 0.0);
  EXPECT_EQ(image.Tvec(1), 0.0);
  EXPECT_EQ(image.Tvec(2), 0.0);
  image.Tvec(0) = 2.0;
  EXPECT_EQ(image.Tvec(0), 2.0);
  image.SetTvec(Eigen::Vector3d(3.0, 0.0, 0.0));
  EXPECT_EQ(image.Tvec(0), 3.0);
  image.Tvec() = Eigen::Vector3d(4.0, 0.0, 0.0);
  EXPECT_EQ(image.Tvec(0), 4.0);
}

TEST(Image, TvecPrior) {
  Image image;
  EXPECT_TRUE(image.TvecPrior().array().isNaN().all());
  EXPECT_FALSE(image.HasTvecPrior());
  image.TvecPrior(0) = 2.0;
  EXPECT_FALSE(image.HasTvecPrior());
  image.TvecPrior(1) = 2.0;
  EXPECT_FALSE(image.HasTvecPrior());
  image.TvecPrior(2) = 2.0;
  EXPECT_TRUE(image.HasTvecPrior());
  EXPECT_EQ(image.TvecPrior(0), 2.0);
  EXPECT_EQ(image.TvecPrior(1), 2.0);
  EXPECT_EQ(image.TvecPrior(2), 2.0);
  image.SetTvecPrior(Eigen::Vector3d(3.0, 0.0, 0.0));
  EXPECT_EQ(image.TvecPrior(0), 3.0);
  image.TvecPrior() = Eigen::Vector3d(4.0, 0.0, 0.0);
  EXPECT_EQ(image.TvecPrior(0), 4.0);
}

TEST(Image, Points2D) {
  Image image;
  EXPECT_EQ(image.Points2D().size(), 0);
  std::vector<Eigen::Vector2d> points2D(10);
  points2D[0] = Eigen::Vector2d(1.0, 2.0);
  image.SetPoints2D(points2D);
  EXPECT_EQ(image.Points2D().size(), 10);
  EXPECT_EQ(image.Point2D(0).X(), 1.0);
  EXPECT_EQ(image.Point2D(0).Y(), 2.0);
  EXPECT_EQ(image.NumPoints3D(), 0);
}

TEST(Image, Points2DWith3D) {
  Image image;
  EXPECT_EQ(image.Points2D().size(), 0);
  std::vector<Point2D> points2D(10);
  points2D[0].XY() = Eigen::Vector2d(1.0, 2.0);
  points2D[0].SetPoint3DId(1);
  image.SetPoints2D(points2D);
  EXPECT_EQ(image.Points2D().size(), 10);
  EXPECT_EQ(image.Point2D(0).X(), 1.0);
  EXPECT_EQ(image.Point2D(0).Y(), 2.0);
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

TEST(Image, NormalizeQvec) {
  Image image;
  EXPECT_LT(std::abs(image.Qvec().norm() - 1.0), 1e-10);
  image.Qvec(0) = 2.0;
  EXPECT_LT(std::abs(image.Qvec().norm() - 2.0), 1e-10);
  image.NormalizeQvec();
  EXPECT_LT(std::abs(image.Qvec().norm() - 1.0), 1e-10);
}

TEST(Image, ProjectionMatrix) {
  Image image;
  EXPECT_TRUE(image.ProjectionMatrix().isApprox(Eigen::Matrix3x4d::Identity()));
}

TEST(Image, InverseProjectionMatrix) {
  Image image;
  EXPECT_TRUE(
      image.InverseProjectionMatrix().isApprox(Eigen::Matrix3x4d::Identity()));
}

TEST(Image, RotationMatrix) {
  Image image;
  EXPECT_TRUE(image.RotationMatrix().isApprox(Eigen::Matrix3d::Identity()));
}

TEST(Image, ProjectionCenter) {
  Image image;
  EXPECT_TRUE(image.ProjectionCenter().isApprox(Eigen::Vector3d::Zero()));
}

TEST(Image, ViewingDirection) {
  Image image;
  EXPECT_TRUE(image.ViewingDirection().isApprox(Eigen::Vector3d(0, 0, 1)));
}

}  // namespace colmap
