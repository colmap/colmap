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

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(PosePrior, Equals) {
  PosePrior prior;
  prior.position = Eigen::Vector3d::Zero();
  prior.position_covariance = Eigen::Matrix3d::Identity();
  prior.rotation = Eigen::Quaterniond(1, 0, 0, 0);
  prior.rotation_covariance = Eigen::Matrix3d::Identity();
  prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;

  PosePrior other = prior;
  EXPECT_EQ(prior, other);

  prior.position.x() = 1;
  EXPECT_NE(prior, other);
  other.position.x() = 1;
  EXPECT_EQ(prior, other);

  prior.rotation =
      Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));
  EXPECT_NE(prior, other);
  other.rotation = prior.rotation;
  EXPECT_EQ(prior, other);

  prior.rotation_covariance(0, 0) = 10;
  EXPECT_NE(prior, other);
  other.rotation_covariance(0, 0) = 10;
  EXPECT_EQ(prior, other);
}

TEST(PosePrior, Print) {
  PosePrior prior;
  prior.position = Eigen::Vector3d::Zero();
  prior.position_covariance = Eigen::Matrix3d::Identity();
  prior.rotation = Eigen::Quaterniond(1, 0, 0, 0);  // identity
  prior.rotation_covariance = Eigen::Matrix3d::Identity();
  prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;

  std::ostringstream stream;
  stream << prior;

  const std::string expected =
      "PosePrior(\n"
      "  position=[0, 0, 0],\n"
      "  position_covariance=[1, 0, 0, 0, 1, 0, 0, 0, 1],\n"
      "  rotation=[0, 0, 0, 1],  // [x, y, z, w]\n"
      "  rotation_covariance=[1, 0, 0, 0, 1, 0, 0, 0, 1],\n"
      "  coordinate_system=CARTESIAN\n"
      ")";

  EXPECT_EQ(stream.str(), expected);
}

}  // namespace
}  // namespace colmap
