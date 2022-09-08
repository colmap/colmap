// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

#define TEST_NAME "base/image"
#include "util/testing.h"

#include "base/image.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestDefault) {
  Image image;
  BOOST_CHECK_EQUAL(image.ImageId(), kInvalidImageId);
  BOOST_CHECK_EQUAL(image.Name(), "");
  BOOST_CHECK_EQUAL(image.CameraId(), kInvalidCameraId);
  BOOST_CHECK_EQUAL(image.HasCamera(), false);
  BOOST_CHECK_EQUAL(image.IsRegistered(), false);
  BOOST_CHECK_EQUAL(image.NumPoints2D(), 0);
  BOOST_CHECK_EQUAL(image.NumPoints3D(), 0);
  BOOST_CHECK_EQUAL(image.NumObservations(), 0);
  BOOST_CHECK_EQUAL(image.NumCorrespondences(), 0);
  BOOST_CHECK_EQUAL(image.NumVisiblePoints3D(), 0);
  BOOST_CHECK_EQUAL(image.Point3DVisibilityScore(), 0);
  BOOST_CHECK_EQUAL(image.Qvec(0), 1.0);
  BOOST_CHECK_EQUAL(image.Qvec(1), 0.0);
  BOOST_CHECK_EQUAL(image.Qvec(2), 0.0);
  BOOST_CHECK_EQUAL(image.Qvec(3), 0.0);
  BOOST_CHECK(IsNaN(image.QvecPrior(0)));
  BOOST_CHECK(IsNaN(image.QvecPrior(1)));
  BOOST_CHECK(IsNaN(image.QvecPrior(2)));
  BOOST_CHECK(IsNaN(image.QvecPrior(3)));
  BOOST_CHECK_EQUAL(image.Tvec(0), 0.0);
  BOOST_CHECK_EQUAL(image.Tvec(1), 0.0);
  BOOST_CHECK_EQUAL(image.Tvec(2), 0.0);
  BOOST_CHECK(IsNaN(image.TvecPrior(0)));
  BOOST_CHECK(IsNaN(image.TvecPrior(1)));
  BOOST_CHECK(IsNaN(image.TvecPrior(2)));
  BOOST_CHECK_EQUAL(image.HasQvecPrior(), false);
  BOOST_CHECK_EQUAL(image.HasTvecPrior(), false);
  BOOST_CHECK_EQUAL(image.Points2D().size(), 0);
}

BOOST_AUTO_TEST_CASE(TestImageId) {
  Image image;
  BOOST_CHECK_EQUAL(image.ImageId(), kInvalidImageId);
  image.SetImageId(1);
  BOOST_CHECK_EQUAL(image.ImageId(), 1);
}

BOOST_AUTO_TEST_CASE(TestName) {
  Image image;
  BOOST_CHECK_EQUAL(image.Name(), "");
  image.SetName("test1");
  BOOST_CHECK_EQUAL(image.Name(), "test1");
  image.Name() = "test2";
  BOOST_CHECK_EQUAL(image.Name(), "test2");
}

BOOST_AUTO_TEST_CASE(TestCameraId) {
  Image image;
  BOOST_CHECK_EQUAL(image.CameraId(), kInvalidCameraId);
  image.SetCameraId(1);
  BOOST_CHECK_EQUAL(image.CameraId(), 1);
}

BOOST_AUTO_TEST_CASE(TestRegistered) {
  Image image;
  BOOST_CHECK_EQUAL(image.IsRegistered(), false);
  image.SetRegistered(true);
  BOOST_CHECK_EQUAL(image.IsRegistered(), true);
  image.SetRegistered(false);
  BOOST_CHECK_EQUAL(image.IsRegistered(), false);
}

BOOST_AUTO_TEST_CASE(TestNumPoints2D) {
  Image image;
  BOOST_CHECK_EQUAL(image.NumPoints2D(), 0);
  image.SetPoints2D(std::vector<Eigen::Vector2d>(10));
  BOOST_CHECK_EQUAL(image.NumPoints2D(), 10);
}

BOOST_AUTO_TEST_CASE(TestNumPoints3D) {
  Image image;
  image.SetPoints2D(std::vector<Eigen::Vector2d>(10));
  BOOST_CHECK_EQUAL(image.NumPoints3D(), 0);
  image.SetPoint3DForPoint2D(0, 0);
  BOOST_CHECK_EQUAL(image.NumPoints3D(), 1);
  image.SetPoint3DForPoint2D(0, 1);
  image.SetPoint3DForPoint2D(1, 2);
  BOOST_CHECK_EQUAL(image.NumPoints3D(), 2);
}

BOOST_AUTO_TEST_CASE(TestNumObservations) {
  Image image;
  BOOST_CHECK_EQUAL(image.NumObservations(), 0);
  image.SetNumObservations(10);
  BOOST_CHECK_EQUAL(image.NumObservations(), 10);
}

BOOST_AUTO_TEST_CASE(TestNumCorrespondences) {
  Image image;
  BOOST_CHECK_EQUAL(image.NumCorrespondences(), 0);
  image.SetNumCorrespondences(10);
  BOOST_CHECK_EQUAL(image.NumCorrespondences(), 10);
}

BOOST_AUTO_TEST_CASE(TestNumVisiblePoints3D) {
  Image image;
  image.SetPoints2D(std::vector<Eigen::Vector2d>(10));
  image.SetNumObservations(10);
  Camera camera;
  camera.SetWidth(10);
  camera.SetHeight(10);
  image.SetUp(camera);
  BOOST_CHECK_EQUAL(image.NumVisiblePoints3D(), 0);
  image.IncrementCorrespondenceHasPoint3D(0);
  BOOST_CHECK_EQUAL(image.NumVisiblePoints3D(), 1);
  image.IncrementCorrespondenceHasPoint3D(0);
  image.IncrementCorrespondenceHasPoint3D(1);
  BOOST_CHECK_EQUAL(image.NumVisiblePoints3D(), 2);
  image.DecrementCorrespondenceHasPoint3D(0);
  BOOST_CHECK_EQUAL(image.NumVisiblePoints3D(), 2);
  image.DecrementCorrespondenceHasPoint3D(0);
  BOOST_CHECK_EQUAL(image.NumVisiblePoints3D(), 1);
  image.DecrementCorrespondenceHasPoint3D(1);
  BOOST_CHECK_EQUAL(image.NumVisiblePoints3D(), 0);
}

BOOST_AUTO_TEST_CASE(TestPoint3DVisibilityScore) {
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
  BOOST_CHECK_EQUAL(image.Point3DVisibilityScore(), 0);
  image.IncrementCorrespondenceHasPoint3D(0);
  BOOST_CHECK_EQUAL(image.Point3DVisibilityScore(), scores.sum());
  image.IncrementCorrespondenceHasPoint3D(0);
  BOOST_CHECK_EQUAL(image.Point3DVisibilityScore(), scores.sum());
  image.IncrementCorrespondenceHasPoint3D(1);
  BOOST_CHECK_EQUAL(image.Point3DVisibilityScore(),
                    scores.sum() + scores.bottomRows(scores.size() - 1).sum());
  image.IncrementCorrespondenceHasPoint3D(1);
  image.IncrementCorrespondenceHasPoint3D(1);
  image.IncrementCorrespondenceHasPoint3D(4);
  BOOST_CHECK_EQUAL(
      image.Point3DVisibilityScore(),
      scores.sum() + 2 * scores.bottomRows(scores.size() - 1).sum());
  image.IncrementCorrespondenceHasPoint3D(4);
  image.IncrementCorrespondenceHasPoint3D(5);
  BOOST_CHECK_EQUAL(
      image.Point3DVisibilityScore(),
      scores.sum() + 3 * scores.bottomRows(scores.size() - 1).sum());
  image.DecrementCorrespondenceHasPoint3D(0);
  BOOST_CHECK_EQUAL(
      image.Point3DVisibilityScore(),
      scores.sum() + 3 * scores.bottomRows(scores.size() - 1).sum());
  image.DecrementCorrespondenceHasPoint3D(0);
  BOOST_CHECK_EQUAL(
      image.Point3DVisibilityScore(),
      scores.sum() + 2 * scores.bottomRows(scores.size() - 1).sum());
  image.IncrementCorrespondenceHasPoint3D(2);
  BOOST_CHECK_EQUAL(
      image.Point3DVisibilityScore(),
      2 * scores.sum() + 2 * scores.bottomRows(scores.size() - 1).sum());
}

BOOST_AUTO_TEST_CASE(TestQvec) {
  Image image;
  BOOST_CHECK_EQUAL(image.Qvec(0), 1.0);
  BOOST_CHECK_EQUAL(image.Qvec(1), 0.0);
  BOOST_CHECK_EQUAL(image.Qvec(2), 0.0);
  BOOST_CHECK_EQUAL(image.Qvec(3), 0.0);
  image.Qvec(0) = 2.0;
  BOOST_CHECK_EQUAL(image.Qvec(0), 2.0);
  image.SetQvec(Eigen::Vector4d(3.0, 0.0, 0.0, 0.0));
  BOOST_CHECK_EQUAL(image.Qvec(0), 3.0);
  image.Qvec() = Eigen::Vector4d(4.0, 0.0, 0.0, 0.0);
  BOOST_CHECK_EQUAL(image.Qvec(0), 4.0);
}

BOOST_AUTO_TEST_CASE(TestQvecPrior) {
  Image image;
  BOOST_CHECK(IsNaN(image.QvecPrior(0)));
  BOOST_CHECK(IsNaN(image.QvecPrior(1)));
  BOOST_CHECK(IsNaN(image.QvecPrior(2)));
  BOOST_CHECK(IsNaN(image.QvecPrior(3)));
  BOOST_CHECK_EQUAL(image.HasQvecPrior(), false);
  image.QvecPrior(0) = 2.0;
  BOOST_CHECK_EQUAL(image.HasQvecPrior(), false);
  image.QvecPrior(1) = 2.0;
  BOOST_CHECK_EQUAL(image.HasQvecPrior(), false);
  image.QvecPrior(2) = 2.0;
  BOOST_CHECK_EQUAL(image.HasQvecPrior(), false);
  image.QvecPrior(3) = 2.0;
  BOOST_CHECK_EQUAL(image.HasQvecPrior(), true);
  BOOST_CHECK_EQUAL(image.QvecPrior(0), 2.0);
  BOOST_CHECK_EQUAL(image.QvecPrior(1), 2.0);
  BOOST_CHECK_EQUAL(image.QvecPrior(2), 2.0);
  BOOST_CHECK_EQUAL(image.QvecPrior(3), 2.0);
  image.SetQvecPrior(Eigen::Vector4d(3.0, 0.0, 0.0, 0.0));
  BOOST_CHECK_EQUAL(image.QvecPrior(0), 3.0);
  image.QvecPrior() = Eigen::Vector4d(4.0, 0.0, 0.0, 0.0);
  BOOST_CHECK_EQUAL(image.QvecPrior(0), 4.0);
}

BOOST_AUTO_TEST_CASE(TestTvec) {
  Image image;
  BOOST_CHECK_EQUAL(image.Tvec(0), 0.0);
  BOOST_CHECK_EQUAL(image.Tvec(1), 0.0);
  BOOST_CHECK_EQUAL(image.Tvec(2), 0.0);
  image.Tvec(0) = 2.0;
  BOOST_CHECK_EQUAL(image.Tvec(0), 2.0);
  image.SetTvec(Eigen::Vector3d(3.0, 0.0, 0.0));
  BOOST_CHECK_EQUAL(image.Tvec(0), 3.0);
  image.Tvec() = Eigen::Vector3d(4.0, 0.0, 0.0);
  BOOST_CHECK_EQUAL(image.Tvec(0), 4.0);
}

BOOST_AUTO_TEST_CASE(TestTvecPrior) {
  Image image;
  BOOST_CHECK(IsNaN(image.TvecPrior(0)));
  BOOST_CHECK(IsNaN(image.TvecPrior(1)));
  BOOST_CHECK(IsNaN(image.TvecPrior(2)));
  BOOST_CHECK_EQUAL(image.HasTvecPrior(), false);
  image.TvecPrior(0) = 2.0;
  BOOST_CHECK_EQUAL(image.HasTvecPrior(), false);
  image.TvecPrior(1) = 2.0;
  BOOST_CHECK_EQUAL(image.HasTvecPrior(), false);
  image.TvecPrior(2) = 2.0;
  BOOST_CHECK_EQUAL(image.HasTvecPrior(), true);
  BOOST_CHECK_EQUAL(image.TvecPrior(0), 2.0);
  BOOST_CHECK_EQUAL(image.TvecPrior(1), 2.0);
  BOOST_CHECK_EQUAL(image.TvecPrior(2), 2.0);
  image.SetTvecPrior(Eigen::Vector3d(3.0, 0.0, 0.0));
  BOOST_CHECK_EQUAL(image.TvecPrior(0), 3.0);
  image.TvecPrior() = Eigen::Vector3d(4.0, 0.0, 0.0);
  BOOST_CHECK_EQUAL(image.TvecPrior(0), 4.0);
}

BOOST_AUTO_TEST_CASE(TestPoints2D) {
  Image image;
  BOOST_CHECK_EQUAL(image.Points2D().size(), 0);
  std::vector<Eigen::Vector2d> points2D(10);
  points2D[0] = Eigen::Vector2d(1.0, 2.0);
  image.SetPoints2D(points2D);
  BOOST_CHECK_EQUAL(image.Points2D().size(), 10);
  BOOST_CHECK_EQUAL(image.Point2D(0).X(), 1.0);
  BOOST_CHECK_EQUAL(image.Point2D(0).Y(), 2.0);
}

BOOST_AUTO_TEST_CASE(TestPoint3D) {
  Image image;
  image.SetPoints2D(std::vector<Eigen::Vector2d>(2));
  BOOST_CHECK(!image.Point2D(0).HasPoint3D());
  BOOST_CHECK(!image.Point2D(1).HasPoint3D());
  BOOST_CHECK_EQUAL(image.NumPoints3D(), 0);
  image.SetPoint3DForPoint2D(0, 0);
  BOOST_CHECK(image.Point2D(0).HasPoint3D());
  BOOST_CHECK(!image.Point2D(1).HasPoint3D());
  BOOST_CHECK_EQUAL(image.NumPoints3D(), 1);
  BOOST_CHECK(image.HasPoint3D(0));
  image.SetPoint3DForPoint2D(0, 1);
  BOOST_CHECK(image.Point2D(0).HasPoint3D());
  BOOST_CHECK(!image.Point2D(1).HasPoint3D());
  BOOST_CHECK_EQUAL(image.NumPoints3D(), 1);
  BOOST_CHECK(!image.HasPoint3D(0));
  BOOST_CHECK(image.HasPoint3D(1));
  image.SetPoint3DForPoint2D(1, 0);
  BOOST_CHECK(image.Point2D(0).HasPoint3D());
  BOOST_CHECK(image.Point2D(1).HasPoint3D());
  BOOST_CHECK_EQUAL(image.NumPoints3D(), 2);
  BOOST_CHECK(image.HasPoint3D(0));
  BOOST_CHECK(image.HasPoint3D(1));
  image.ResetPoint3DForPoint2D(0);
  BOOST_CHECK(!image.Point2D(0).HasPoint3D());
  BOOST_CHECK(image.Point2D(1).HasPoint3D());
  BOOST_CHECK_EQUAL(image.NumPoints3D(), 1);
  BOOST_CHECK(image.HasPoint3D(0));
  BOOST_CHECK(!image.HasPoint3D(1));
  image.ResetPoint3DForPoint2D(1);
  BOOST_CHECK(!image.Point2D(0).HasPoint3D());
  BOOST_CHECK(!image.Point2D(1).HasPoint3D());
  BOOST_CHECK_EQUAL(image.NumPoints3D(), 0);
  BOOST_CHECK(!image.HasPoint3D(0));
  BOOST_CHECK(!image.HasPoint3D(1));
  image.ResetPoint3DForPoint2D(0);
  BOOST_CHECK(!image.Point2D(0).HasPoint3D());
  BOOST_CHECK(!image.Point2D(1).HasPoint3D());
  BOOST_CHECK_EQUAL(image.NumPoints3D(), 0);
  BOOST_CHECK(!image.HasPoint3D(0));
  BOOST_CHECK(!image.HasPoint3D(1));
}

BOOST_AUTO_TEST_CASE(TestNormalizeQvec) {
  Image image;
  BOOST_CHECK_LT(std::abs(image.Qvec().norm() - 1.0), 1e-10);
  image.Qvec(0) = 2.0;
  BOOST_CHECK_LT(std::abs(image.Qvec().norm() - 2.0), 1e-10);
  image.NormalizeQvec();
  BOOST_CHECK_LT(std::abs(image.Qvec().norm() - 1.0), 1e-10);
}

BOOST_AUTO_TEST_CASE(TestProjectionMatrix) {
  Image image;
  BOOST_CHECK(image.ProjectionMatrix().isApprox(Eigen::Matrix3x4d::Identity()));
}

BOOST_AUTO_TEST_CASE(TestInverseProjectionMatrix) {
  Image image;
  BOOST_CHECK(
      image.InverseProjectionMatrix().isApprox(Eigen::Matrix3x4d::Identity()));
}

BOOST_AUTO_TEST_CASE(TestRotationMatrix) {
  Image image;
  BOOST_CHECK(image.RotationMatrix().isApprox(Eigen::Matrix3d::Identity()));
}

BOOST_AUTO_TEST_CASE(TestProjectionCenter) {
  Image image;
  BOOST_CHECK(image.ProjectionCenter().isApprox(Eigen::Vector3d::Zero()));
}

BOOST_AUTO_TEST_CASE(TestViewingDirection) {
  Image image;
  BOOST_CHECK(image.ViewingDirection().isApprox(Eigen::Vector3d(0, 0, 1)));
}
