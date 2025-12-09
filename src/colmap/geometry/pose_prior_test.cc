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

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(PosePrior, Equals) {
  PosePrior prior;
  prior.pose_prior_id = 0;
  prior.corr_data_id = data_t(sensor_t(SensorType::CAMERA, 1), 2);
  prior.position = Eigen::Vector3d::Zero();
  prior.position_covariance = Eigen::Matrix3d::Identity();
  prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  prior.gravity = Eigen::Vector3d::UnitZ();
  PosePrior other = prior;
  EXPECT_EQ(prior, other);
  prior.position.x() = 1;
  EXPECT_NE(prior, other);
  other.position.x() = 1;
  EXPECT_EQ(prior, other);
}

TEST(PosePrior, NaNEquals) {
  PosePrior prior;
  PosePrior other = prior;
  EXPECT_EQ(prior, other);
  prior.position =
      Eigen::Vector3d(1, 2, std::numeric_limits<double>::quiet_NaN());
  other.position =
      Eigen::Vector3d(1, 2, std::numeric_limits<double>::quiet_NaN());
  EXPECT_EQ(prior, other);
  prior.position_covariance = Eigen::Matrix3d::Identity();
  prior.position_covariance(0, 0) = std::numeric_limits<double>::quiet_NaN();
  other.position_covariance = Eigen::Matrix3d::Identity();
  other.position_covariance(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(prior, other);
  other.position.z() = 3;
  EXPECT_NE(prior, other);
}

TEST(PosePrior, Print) {
  PosePrior prior;
  prior.pose_prior_id = 0;
  prior.corr_data_id = data_t(sensor_t(SensorType::CAMERA, 1), 2);
  prior.position = Eigen::Vector3d::Zero();
  prior.position_covariance = Eigen::Matrix3d::Identity();
  prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  prior.gravity = Eigen::Vector3d::UnitZ();
  std::ostringstream stream;
  stream << prior;
  EXPECT_EQ(stream.str(),
            "PosePrior(pose_prior_id=0, corr_data_id=(CAMERA, 1, 2), "
            "position=[0, 0, 0], "
            "position_covariance=[1, 0, 0, 0, 1, 0, 0, 0, 1], "
            "coordinate_system=CARTESIAN, gravity=[0, 0, 1])");
}

}  // namespace
}  // namespace colmap
