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
  prior.world_from_cam.translation = Eigen::Vector3d::Zero();
  prior.world_from_cam.rotation = Eigen::Quaterniond::Identity();
  prior.position_covariance = Eigen::Matrix3d::Identity();
  prior.rotation_covariance = Eigen::Matrix3d::Identity();

  PosePrior other = prior;
  EXPECT_EQ(prior, other);

  prior.world_from_cam.translation = Eigen::Vector3d(1, 2, 3);
  EXPECT_NE(prior, other);
  other.world_from_cam.translation = Eigen::Vector3d(1, 2, 3);
  EXPECT_EQ(prior, other);

  prior.world_from_cam.rotation =
      Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));
  EXPECT_NE(prior, other);
  other.world_from_cam.rotation =
      Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));
  EXPECT_EQ(prior, other);

  prior.rotation_covariance = Eigen::Vector3d(1, 2, 3).asDiagonal();
  EXPECT_NE(prior, other);
  other.rotation_covariance = Eigen::Vector3d(1, 2, 3).asDiagonal();
  EXPECT_EQ(prior, other);
}

TEST(PosePrior, Covariance) {
  const Eigen::Vector3d position = {1, 2, 3};
  const Eigen::Quaterniond rotation =
      Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));

  PosePrior prior1;
  prior1.world_from_cam.translation = position;
  prior1.world_from_cam.rotation = rotation;

  Eigen::Matrix3d covariance = Eigen::Vector3d(0.1, 0.2, 0.3).asDiagonal();
  prior1.position_covariance = covariance;
  EXPECT_EQ(prior1.position_covariance, covariance);

  prior1.rotation_covariance = covariance;
  EXPECT_EQ(prior1.rotation_covariance, covariance);

  PosePrior prior2(position, rotation);
  prior2.SetWorldFromCamCovariance(prior1.WorldFromCamCovariance());
  EXPECT_EQ(prior1.position_covariance, prior2.position_covariance);
  EXPECT_EQ(prior1.rotation_covariance, prior2.rotation_covariance);
}

TEST(PosePrior, Print) {
  PosePrior prior;
  prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  prior.world_from_cam.translation = Eigen::Vector3d::Zero();
  prior.position_covariance = Eigen::Matrix3d::Identity();
  prior.world_from_cam.rotation = Eigen::Quaterniond::Identity();
  prior.rotation_covariance = Eigen::Matrix3d::Identity();

  std::ostringstream stream;
  stream << prior;

  EXPECT_EQ(stream.str(),
            "PosePrior(\n"
            "  world_from_cam=[Rigid3d(rotation_xyzw=[0, 0, 0, 1], "
            "translation=[0, 0, 0])],\n"
            "  position_covariance=[1, 0, 0, 0, 1, 0, 0, 0, 1],\n"
            "  rotation_covariance=[1, 0, 0, 0, 1, 0, 0, 0, 1],\n"
            "  coordinate_system=CARTESIAN)");
}

}  // namespace
}  // namespace colmap
