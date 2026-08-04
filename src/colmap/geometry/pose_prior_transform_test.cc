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

#include "colmap/geometry/pose_prior_transform.h"

#include "colmap/util/eigen_matchers.h"

#include <algorithm>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

PosePrior MakePrior(data_t data_id,
                    double lat,
                    double lon,
                    double alt,
                    PosePrior::CoordinateSystem coordinate_system =
                        PosePrior::CoordinateSystem::WGS84) {
  PosePrior prior;
  prior.corr_data_id = data_id;
  prior.coordinate_system = coordinate_system;
  prior.position = Eigen::Vector3d(lat, lon, alt);
  return prior;
}

data_t CameraData(uint32_t id) {
  return data_t(sensor_t(SensorType::CAMERA, 1), id);
}

TEST(PosePriorEnuFrame, NoUsablePriorsYieldsNoFrame) {
  EXPECT_FALSE(PosePriorEnuFrame::Derive({}).has_value());

  // A Cartesian prior carries no datum, and a WGS84 row with no latitude or
  // longitude carries no position. Neither can define an Earth frame, and
  // returning a default-constructed one would silently place the scene at
  // (0, 0) off the coast of Africa.
  std::vector<PosePrior> priors;
  priors.push_back(MakePrior(
      CameraData(1), 10.0, 20.0, 30.0, PosePrior::CoordinateSystem::CARTESIAN));
  priors.push_back(
      MakePrior(CameraData(2), PosePrior::kNaN, PosePrior::kNaN, 30.0));
  EXPECT_FALSE(PosePriorEnuFrame::Derive(priors).has_value());
}

TEST(PosePriorEnuFrame, IsUsableMatchesWhatDeriveAccepts) {
  EXPECT_TRUE(PosePriorEnuFrame::IsUsable(MakePrior(CameraData(1), 1, 2, 3)));
  // Altitude is optional: the row still anchors a latitude and longitude.
  EXPECT_TRUE(PosePriorEnuFrame::IsUsable(
      MakePrior(CameraData(1), 1, 2, PosePrior::kNaN)));
  EXPECT_FALSE(PosePriorEnuFrame::IsUsable(
      MakePrior(CameraData(1), PosePrior::kNaN, 2, 3)));
  EXPECT_FALSE(PosePriorEnuFrame::IsUsable(
      MakePrior(CameraData(1), 1, PosePrior::kNaN, 3)));
  EXPECT_FALSE(PosePriorEnuFrame::IsUsable(MakePrior(
      CameraData(1), 1, 2, 3, PosePrior::CoordinateSystem::CARTESIAN)));
}

TEST(PosePriorEnuFrame, SinglePriorIsItsOwnOrigin) {
  const std::vector<PosePrior> priors = {
      MakePrior(CameraData(1), 45.5, -73.6, 40.0)};
  const auto frame = PosePriorEnuFrame::Derive(priors);
  ASSERT_TRUE(frame.has_value());
  EXPECT_NEAR(frame->OriginWgs84().x(), 45.5, 1e-9);
  EXPECT_NEAR(frame->OriginWgs84().y(), -73.6, 1e-9);
  EXPECT_NEAR(frame->OriginWgs84().z(), 40.0, 1e-9);
  EXPECT_TRUE(frame->HasRealAltitude());
  EXPECT_THAT(frame->PositionInEnu(priors[0]),
              EigenMatrixNear(Eigen::Vector3d::Zero().eval(), 1e-6));
}

TEST(PosePriorEnuFrame, HorizontalOnlyRowsStillAnchorTheOrigin) {
  // Three coincident latitude/longitude rows, only one of which declares an
  // altitude. If rows without a height were dropped from the origin
  // computation, the frame would depend on which rows happened to carry one --
  // a field unrelated to where the capture is.
  std::vector<PosePrior> priors = {
      MakePrior(CameraData(1), 10.0, 20.0, PosePrior::kNaN),
      MakePrior(CameraData(2), 10.0, 20.0, 100.0),
      MakePrior(CameraData(3), 10.0, 20.0, PosePrior::kNaN)};
  const auto frame = PosePriorEnuFrame::Derive(priors);
  ASSERT_TRUE(frame.has_value());
  EXPECT_NEAR(frame->OriginWgs84().x(), 10.0, 1e-9);
  EXPECT_NEAR(frame->OriginWgs84().y(), 20.0, 1e-9);
  EXPECT_NEAR(frame->OriginWgs84().z(), 100.0, 1e-9);
  EXPECT_TRUE(frame->HasRealAltitude());
}

TEST(PosePriorEnuFrame, NoAltitudeAnywhereIsReportedNotFabricated) {
  const std::vector<PosePrior> priors = {
      MakePrior(CameraData(1), 10.0, 20.0, PosePrior::kNaN)};
  const auto frame = PosePriorEnuFrame::Derive(priors);
  ASSERT_TRUE(frame.has_value());
  // The origin needs *an* altitude to be a tangent plane at all, so it uses
  // 0.0 -- but a caller that publishes an ellipsoidal height must be able to
  // tell that apart from a measured sea-level capture.
  EXPECT_NEAR(frame->OriginWgs84().z(), 0.0, 1e-9);
  EXPECT_FALSE(frame->HasRealAltitude());
}

TEST(PosePriorEnuFrame, PositionInEnuLeavesAbsentAltitudeAbsent) {
  const std::vector<PosePrior> priors = {
      MakePrior(CameraData(1), 10.0, 20.0, 100.0),
      MakePrior(CameraData(2), 10.0, 20.0, PosePrior::kNaN)};
  const auto frame = PosePriorEnuFrame::Derive(priors);
  ASSERT_TRUE(frame.has_value());

  const Eigen::Vector3d full = frame->PositionInEnu(priors[0]);
  EXPECT_THAT(full, EigenMatrixNear(Eigen::Vector3d::Zero().eval(), 1e-6));

  const Eigen::Vector3d horizontal = frame->PositionInEnu(priors[1]);
  EXPECT_NEAR(horizontal.x(), 0.0, 1e-6);
  EXPECT_NEAR(horizontal.y(), 0.0, 1e-6);
  EXPECT_FALSE(std::isfinite(horizontal.z()))
      << "an absent height must not come back as the origin's height";
}

TEST(PosePriorEnuFrame, EastNorthUpAxesPointTheRightWay) {
  const std::vector<PosePrior> priors = {
      MakePrior(CameraData(1), 0.0, 0.0, 0.0)};
  const auto frame = PosePriorEnuFrame::Derive(priors);
  ASSERT_TRUE(frame.has_value());

  // A point slightly east, north, and above the origin must land on the
  // positive East, North and Up axes respectively. This catches an axis swap
  // or sign flip that every "is it self-consistent" test would miss.
  const Eigen::Vector3d east =
      frame->PositionInEnu(MakePrior(CameraData(2), 0.0, 0.001, 0.0));
  EXPECT_GT(east.x(), 1.0);
  EXPECT_NEAR(east.y(), 0.0, 1e-3);

  const Eigen::Vector3d north =
      frame->PositionInEnu(MakePrior(CameraData(3), 0.001, 0.0, 0.0));
  EXPECT_NEAR(north.x(), 0.0, 1e-3);
  EXPECT_GT(north.y(), 1.0);

  const Eigen::Vector3d up =
      frame->PositionInEnu(MakePrior(CameraData(4), 0.0, 0.0, 25.0));
  EXPECT_NEAR(up.x(), 0.0, 1e-6);
  EXPECT_NEAR(up.y(), 0.0, 1e-6);
  EXPECT_NEAR(up.z(), 25.0, 1e-6);
}

TEST(PosePriorEnuFrame, SharedFromLocalEnuRotatesADistantRowsAxes) {
  // At lat=0, lon=90, GPSTransform::ENUFromECEF reduces to the signed
  // permutation R90 = [[-1,0,0],[0,0,1],[0,1,0]] (its own transpose and
  // inverse), and at lat=0, lon=0 to R0 = [[0,1,0],[0,0,1],[1,0,0]]. The
  // local-to-shared rotation for a row at (0,0) in a frame originating at
  // (0,90) is therefore S = R90 * R0^T = [[0,0,-1],[0,1,0],[1,0,0]], derived
  // by hand rather than by calling GPSTransform, so this test would catch a
  // change in GPSTransform's own convention.
  const std::vector<PosePrior> priors = {
      MakePrior(CameraData(1), 0.0, 90.0, 0.0)};
  const auto frame = PosePriorEnuFrame::Derive(priors);
  ASSERT_TRUE(frame.has_value());

  Eigen::Matrix3d expected_s;
  // clang-format off
  expected_s << 0, 0, -1,
                0, 1,  0,
                1, 0,  0;
  // clang-format on
  EXPECT_THAT(frame->SharedFromLocalEnu(Eigen::Vector3d(0.0, 0.0, 0.0)),
              EigenMatrixNear(expected_s, 1e-9));

  // At the origin's own latitude/longitude it must be exactly identity.
  EXPECT_THAT(frame->SharedFromLocalEnu(Eigen::Vector3d(0.0, 90.0, 0.0)),
              EigenMatrixNear(Eigen::Matrix3d::Identity().eval(), 1e-9));
}

TEST(PosePriorEnuFrame, CovarianceIsRotatedIntoTheSharedFrame) {
  // Same geometry as above: S conjugates diag(1,4,9) into diag(9,4,1),
  // swapping axes 0 and 2. Skipping the rotation would leave diag(1,4,9) and
  // quietly point the anisotropic uncertainty the wrong way.
  const std::vector<PosePrior> priors = {
      MakePrior(CameraData(1), 0.0, 90.0, 0.0)};
  const auto frame = PosePriorEnuFrame::Derive(priors);
  ASSERT_TRUE(frame.has_value());

  PosePrior distant = MakePrior(CameraData(2), 0.0, 0.0, 0.0);
  distant.position_covariance = Eigen::Vector3d(1.0, 4.0, 9.0).asDiagonal();
  const Eigen::Matrix3d expected = Eigen::Vector3d(9.0, 4.0, 1.0).asDiagonal();
  EXPECT_THAT(frame->CovarianceInEnu(distant), EigenMatrixNear(expected, 1e-6));

  // A covariance at the origin's own position is unchanged.
  PosePrior local = MakePrior(CameraData(3), 0.0, 90.0, 0.0);
  local.position_covariance = Eigen::Vector3d(1.0, 4.0, 9.0).asDiagonal();
  const Eigen::Matrix3d unchanged = Eigen::Vector3d(1.0, 4.0, 9.0).asDiagonal();
  EXPECT_THAT(frame->CovarianceInEnu(local), EigenMatrixNear(unchanged, 1e-6));
}

TEST(PosePriorEnuFrame, OriginIsIdenticalUnderInputPermutation) {
  // The mapper and the report read the same priors but not necessarily in the
  // same order. GeometricMedian accumulates in input order, so without the
  // deterministic sort the two could differ in the last bits of the origin --
  // and every transform either one publishes is expressed against it.
  std::vector<PosePrior> priors;
  for (uint32_t i = 0; i < 24; ++i) {
    priors.push_back(MakePrior(CameraData(i + 1),
                               45.5 + 0.0007 * i,
                               -73.6 - 0.0011 * i,
                               40.0 + 0.3 * i));
  }
  const auto reference = PosePriorEnuFrame::Derive(priors);
  ASSERT_TRUE(reference.has_value());

  std::mt19937 rng(42);
  for (int trial = 0; trial < 8; ++trial) {
    std::vector<PosePrior> shuffled = priors;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    const auto frame = PosePriorEnuFrame::Derive(shuffled);
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->OriginWgs84().x(), reference->OriginWgs84().x());
    EXPECT_EQ(frame->OriginWgs84().y(), reference->OriginWgs84().y());
    EXPECT_EQ(frame->OriginWgs84().z(), reference->OriginWgs84().z());
    EXPECT_EQ(frame->OriginEcef().x(), reference->OriginEcef().x());
    EXPECT_EQ(frame->OriginEcef().y(), reference->OriginEcef().y());
    EXPECT_EQ(frame->OriginEcef().z(), reference->OriginEcef().z());
  }
}

TEST(PosePriorEnuFrame, GrossOutlierDoesNotMoveTheOrigin) {
  // A single wildly wrong GPS row is a valid archive row -- robust fitting,
  // not the parser, rejects it. It must not drag the frame that every
  // transform is expressed against.
  std::vector<PosePrior> clustered;
  for (uint32_t i = 0; i < 20; ++i) {
    clustered.push_back(MakePrior(
        CameraData(i + 1), 45.5 + 0.0001 * i, -73.6 + 0.0001 * i, 40.0));
  }
  const auto clean = PosePriorEnuFrame::Derive(clustered);
  ASSERT_TRUE(clean.has_value());

  std::vector<PosePrior> with_outlier = clustered;
  with_outlier.push_back(MakePrior(CameraData(999), -20.0, 150.0, 40.0));
  const auto polluted = PosePriorEnuFrame::Derive(with_outlier);
  ASSERT_TRUE(polluted.has_value());

  // Judge the movement against the spread of the real data, not an absolute
  // metre count: the estimator's job is to stay inside the capture, and what
  // counts as "inside" is set by how large the capture is.
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);
  std::vector<Eigen::Vector3d> clustered_lla;
  for (const PosePrior& prior : clustered) {
    clustered_lla.push_back(prior.position);
  }
  const std::vector<Eigen::Vector3d> clustered_ecef =
      gps_transform.EllipsoidToECEF(clustered_lla);
  const double cluster_extent_m =
      (clustered_ecef.front() - clustered_ecef.back()).norm();
  ASSERT_GT(cluster_extent_m, 100.0) << "fixture should span a real capture";

  const double median_shift_m =
      (polluted->OriginEcef() - clean->OriginEcef()).norm();
  EXPECT_LT(median_shift_m, 0.05 * cluster_extent_m);

  // The mean is what this estimator replaced. Asserting that it *is* dragged
  // keeps the test honest: if GeometricMedian silently degenerated into an
  // average, the check above alone could still pass on a kinder fixture.
  const auto mean_of = [](const std::vector<Eigen::Vector3d>& points) {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& point : points) {
      sum += point;
    }
    return (sum / static_cast<double>(points.size())).eval();
  };
  std::vector<Eigen::Vector3d> with_outlier_ecef = clustered_ecef;
  with_outlier_ecef.push_back(
      gps_transform.EllipsoidToECEF({with_outlier.back().position})[0]);
  const double mean_shift_m =
      (mean_of(with_outlier_ecef) - mean_of(clustered_ecef)).norm();
  EXPECT_GT(mean_shift_m, 100000.0);
  EXPECT_LT(median_shift_m, 1e-4 * mean_shift_m);
}

TEST(PosePriorEnuFrame, EnuAndEcefRotationsAreInverses) {
  const std::vector<PosePrior> priors = {
      MakePrior(CameraData(1), 45.5, -73.6, 40.0)};
  const auto frame = PosePriorEnuFrame::Derive(priors);
  ASSERT_TRUE(frame.has_value());
  EXPECT_THAT((frame->EnuFromEcef() * frame->EcefFromEnu()).eval(),
              EigenMatrixNear(Eigen::Matrix3d::Identity().eval(), 1e-12));
  EXPECT_NEAR(frame->EnuFromEcef().determinant(), 1.0, 1e-12);
}

}  // namespace
}  // namespace colmap
