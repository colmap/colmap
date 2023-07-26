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

#include "colmap/estimators/generalized_relative_pose.h"

#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/optim/loransac.h"
#include "colmap/scene/projection.h"

#include <array>

#include <gtest/gtest.h>

namespace colmap {

TEST(GeneralizedRelativePose, Estimate) {
  const size_t kNumPoints = 100;

  std::vector<Eigen::Vector3d> points3D;
  for (size_t i = 0; i < kNumPoints; ++i) {
    points3D.emplace_back(Eigen::Vector3d::Random());
  }

  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double qx = 0; qx < 0.4; qx += 0.1) {
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
    for (double tx = 0; tx < 0.5; tx += 0.1) {
      const int kRefCamIdx = 1;
      const int kNumCams = 3;

      const std::array<Rigid3d, kNumCams> cams_from_world = {{
          Rigid3d(Eigen::Quaterniond(1, qx, 0, 0).normalized(),
                  Eigen::Vector3d(tx, 0.1, 0)),
          Rigid3d(Eigen::Quaterniond(1, qx + 0.05, 0, 0).normalized(),
                  Eigen::Vector3d(tx, 0.2, 0)),
          Rigid3d(Eigen::Quaterniond(1, qx + 0.1, 0, 0).normalized(),
                  Eigen::Vector3d(tx, 0.3, 0)),
      }};

      std::array<Rigid3d, kNumCams> cams_from_rig;
      for (size_t i = 0; i < kNumCams; ++i) {
        cams_from_rig[i] =
            cams_from_world[i] * Inverse(cams_from_world[kRefCamIdx]);
      }

      // Project points to cameras.
      std::vector<GR6PEstimator::X_t> points1;
      std::vector<GR6PEstimator::Y_t> points2;
      for (size_t i = 0; i < points3D.size(); ++i) {
        const Eigen::Vector3d point3D_camera1 =
            cams_from_rig[i % kNumCams] * points3D[i];
        const Eigen::Vector3d point3D_camera2 =
            cams_from_world[(i + 1) % kNumCams] * points3D[i];
        if (point3D_camera1.z() < 0 || point3D_camera2.z() < 0) {
          continue;
        }

        points1.emplace_back();
        points1.back().cam_from_rig = cams_from_rig[i % kNumCams];
        points1.back().ray_in_cam = point3D_camera1.normalized();

        points2.emplace_back();
        points2.back().cam_from_rig = cams_from_rig[(i + 1) % kNumCams];
        points2.back().ray_in_cam = point3D_camera2.normalized();
      }

      RANSACOptions options;
      options.max_error = 1e-3;
      LORANSAC<GR6PEstimator, GR6PEstimator> ransac(options);
      const auto report = ransac.Estimate(points1, points2);

      EXPECT_TRUE(report.success);
      EXPECT_LT(
          (cams_from_world[kRefCamIdx].ToMatrix() - report.model.ToMatrix())
              .norm(),
          1e-2);

      std::vector<double> residuals;
      GR6PEstimator::Residuals(points1, points2, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_LE(residuals[i], options.max_error);
      }
    }
  }
}

}  // namespace colmap
