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

#include "colmap/geometry/triangulation.h"

#include "colmap/geometry/rigid3.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(TriangulatePoint, Nominal) {
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(0, 0.1, 0.1),
      Eigen::Vector3d(0, 1, 3),
      Eigen::Vector3d(0, 1, 2),
      Eigen::Vector3d(0.01, 0.2, 3),
      Eigen::Vector3d(-1, 0.1, 1),
      Eigen::Vector3d(0.1, 0.1, 0.2),
  };

  const Rigid3d cam_from_world1;

  for (int z = 0; z < 5; ++z) {
    const double qz = z / 5.0;
    for (int tx = 0; tx < 10; tx += 2) {
      const Rigid3d cam_from_world2(Eigen::Quaterniond(0.2, 0.3, 0.4, qz),
                                    Eigen::Vector3d(tx, 2, 3));
      for (size_t i = 0; i < points3D.size(); ++i) {
        const Eigen::Vector3d& point3D = points3D[i];
        const Eigen::Vector3d point2D1 = cam_from_world1 * point3D;
        const Eigen::Vector3d point2D2 = cam_from_world2 * point3D;

        const Eigen::Vector3d tri_point3D =
            TriangulatePoint(cam_from_world1.ToMatrix(),
                             cam_from_world2.ToMatrix(),
                             point2D1.hnormalized(),
                             point2D2.hnormalized());

        EXPECT_TRUE((point3D - tri_point3D).norm() < 1e-10);
      }
    }
  }
}

TEST(CalculateTriangulationAngle, Nominal) {
  const Eigen::Vector3d tvec1(0, 0, 0);
  const Eigen::Vector3d tvec2(0, 1, 0);

  EXPECT_NEAR(
      CalculateTriangulationAngle(tvec1, tvec2, Eigen::Vector3d(0, 0, 100)),
      0.009999666687,
      1e-8);
  EXPECT_NEAR(
      CalculateTriangulationAngle(tvec1, tvec2, Eigen::Vector3d(0, 0, 50)),
      0.019997333973,
      1e-8);
  EXPECT_NEAR(CalculateTriangulationAngles(
                  tvec1, tvec2, {Eigen::Vector3d(0, 0, 50)})[0],
              0.019997333973,
              1e-8);
}

}  // namespace
}  // namespace colmap
