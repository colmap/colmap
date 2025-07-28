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

#include "colmap/geometry/pose_prior.h"

#include "colmap/math/math.h"
#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(PosePrior, Equals) {
  PosePrior prior;
  prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  prior.SetPosition(Eigen::Vector3d::Zero());
  prior.SetPositionCovariance(Eigen::Matrix3d::Identity());
  prior.SetRotation(Eigen::Quaterniond::Identity());
  prior.SetRotationCovariance(Eigen::Matrix3d::Identity());

  PosePrior other = prior;
  EXPECT_EQ(prior, other);

  prior.SetPosition(Eigen::Vector3d(1, 2, 3));
  EXPECT_NE(prior, other);
  other.SetPosition(Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(prior, other);

  prior.SetRotation(Eigen::Quaterniond(
      Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ())));
  EXPECT_NE(prior, other);
  other.SetRotation(Eigen::Quaterniond(
      Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ())));
  EXPECT_EQ(prior, other);

  prior.SetRotationCovariance(Eigen::Vector3d(1, 2, 3).asDiagonal());
  EXPECT_NE(prior, other);
  other.SetRotationCovariance(Eigen::Vector3d(1, 2, 3).asDiagonal());
  EXPECT_EQ(prior, other);
}

TEST(PosePrior, GetAndSetCovariance) {
  const Eigen::Vector3d position = {1, 2, 3};
  const Eigen::Quaterniond rotation =
      Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));

  PosePrior prior1;
  prior1.SetPosition(position);
  prior1.SetRotation(rotation);

  Eigen::Matrix3d covariance = Eigen::Vector3d(0.1, 0.2, 0.3).asDiagonal();
  prior1.SetPositionCovariance(covariance);
  EXPECT_EQ(prior1.PositionCovariance(), covariance);

  prior1.SetRotationCovariance(covariance);
  EXPECT_EQ(prior1.RotationCovariance(), covariance);

  prior1.SetTranslationCovariance(covariance);
  EXPECT_THAT(
      prior1.TranslationCovariance(),
      EigenMatrixNear(
          Eigen::Matrix3d(prior1.PoseCovariance().block<3, 3>(0, 0)), 1e-14));
  EXPECT_THAT(
      prior1.RotationCovariance(),
      EigenMatrixNear(
          Eigen::Matrix3d(prior1.PoseCovariance().block<3, 3>(3, 3)), 1e-14));

  PosePrior prior2(position, rotation);
  prior2.SetPoseCovariance(prior1.PoseCovariance());

  // NOTE: Decomposing pose covariance may introduce slight differences vs.
  // directly setting position/rotation covariance.
  EXPECT_THAT(prior1.PositionCovariance(),
              EigenMatrixNear(prior2.PositionCovariance(), 1e-16));
  EXPECT_THAT(prior1.RotationCovariance(),
              EigenMatrixNear(prior2.RotationCovariance(), 1e-16));
}

TEST(PosePrior, Print) {
  PosePrior prior;
  prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  prior.SetPosition(Eigen::Vector3d::Zero());
  prior.SetPositionCovariance(Eigen::Matrix3d::Identity());
  prior.SetRotation(Eigen::Quaterniond::Identity());
  prior.SetRotationCovariance(Eigen::Matrix3d::Identity());

  std::ostringstream stream;
  stream << prior;

  EXPECT_EQ(stream.str(),
            "PosePrior(\n"
            "  position=[-0, -0, -0],\n"
            "  position_covariance=[1, 0, 0, 0, 1, 0, 0, 0, 1],\n"
            "  rotation=[0, 0, 0, 1],  // [x, y, z, w]\n"
            "  rotation_covariance=[1, 0, 0, 0, 1, 0, 0, 0, 1],\n"
            "  coordinate_system=CARTESIAN\n"
            ")");
}

}  // namespace
}  // namespace colmap
