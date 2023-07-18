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

#include "colmap/estimators/generalized_absolute_pose.h"

#include "colmap/geometry/pose.h"
#include "colmap/geometry/projection.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/optim/ransac.h"

#include <array>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {

TEST(GeneralizedAbsolutePose, Estimate) {
  std::vector<Eigen::Vector3d> points3D;
  points3D.emplace_back(1, 1, 1);
  points3D.emplace_back(0, 1, 1);
  points3D.emplace_back(3, 1.0, 4);
  points3D.emplace_back(3, 1.1, 4);
  points3D.emplace_back(3, 1.2, 4);
  points3D.emplace_back(3, 1.3, 4);
  points3D.emplace_back(3, 1.4, 4);
  points3D.emplace_back(2, 1, 7);

  auto points3D_faulty = points3D;
  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D_faulty[i](0) = 20;
  }

  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double qx = 0; qx < 1; qx += 0.2) {
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
    for (double tx = 0; tx < 1; tx += 0.1) {
      const int kRefTform = 1;
      const int kNumTforms = 3;

      const std::array<Rigid3d, kNumTforms> expected_cam_from_worlds = {{
          Rigid3d(Eigen::Quaterniond(1, qx, 0, 0).normalized(),
                  Eigen::Vector3d(tx, -0.1, 0)),
          Rigid3d(Eigen::Quaterniond(1, qx, 0, 0).normalized(),
                  Eigen::Vector3d(tx, 0, 0)),
          Rigid3d(Eigen::Quaterniond(1, qx, 0, 0).normalized(),
                  Eigen::Vector3d(tx, 0.1, 0)),
      }};

      std::array<Eigen::Matrix3x4d, kNumTforms> rel_tforms;
      for (size_t i = 0; i < kNumTforms; ++i) {
        const Rigid3d cam_i_from_ref_cam =
            expected_cam_from_worlds[i] *
            Inverse(expected_cam_from_worlds[kRefTform]);
        rel_tforms[i] = cam_i_from_ref_cam.ToMatrix();
      }

      // Project points to camera coordinate system.
      std::vector<GP3PEstimator::X_t> points2D;
      for (size_t i = 0; i < points3D.size(); ++i) {
        points2D.emplace_back();
        points2D.back().rel_tform = rel_tforms[i % kNumTforms];
        points2D.back().xy =
            (expected_cam_from_worlds[i % kNumTforms] * points3D[i])
                .hnormalized();
      }

      RANSACOptions options;
      options.max_error = 1e-5;
      RANSAC<GP3PEstimator> ransac(options);
      const auto report = ransac.Estimate(points2D, points3D);

      EXPECT_TRUE(report.success);
      EXPECT_LT((expected_cam_from_worlds[kRefTform].ToMatrix() - report.model)
                    .norm(),
                1e-2);

      // Test residuals of exact points.
      std::vector<double> residuals;
      ransac.estimator.Residuals(points2D, points3D, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_TRUE(residuals[i] < 1e-10);
      }

      // Test residuals of faulty points.
      ransac.estimator.Residuals(
          points2D, points3D_faulty, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_TRUE(residuals[i] > 1e-10);
      }
    }
  }
}

}  // namespace colmap
