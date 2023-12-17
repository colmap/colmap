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

#include "colmap/estimators/absolute_pose.h"

#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/optim/ransac.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(AbsolutePose, P3P) {
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(1, 1, 1),
      Eigen::Vector3d(0, 1, 1),
      Eigen::Vector3d(3, 1.0, 4),
      Eigen::Vector3d(3, 1.1, 4),
      Eigen::Vector3d(3, 1.2, 4),
      Eigen::Vector3d(3, 1.3, 4),
      Eigen::Vector3d(3, 1.4, 4),
      Eigen::Vector3d(2, 1, 7),
  };

  auto points3D_faulty = points3D;
  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D_faulty[i](0) = 20;
  }

  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double qx = 0; qx < 1; qx += 0.2) {
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
    for (double tx = 0; tx < 1; tx += 0.1) {
      const Rigid3d expected_cam_from_world(
          Eigen::Quaterniond(1, qx, 0, 0).normalized(),
          Eigen::Vector3d(tx, 0, 0));

      // Project points to camera coordinate system.
      std::vector<Eigen::Vector2d> points2D;
      for (size_t i = 0; i < points3D.size(); ++i) {
        points2D.push_back(
            (expected_cam_from_world * points3D[i]).hnormalized());
      }

      RANSACOptions options;
      options.max_error = 1e-5;
      RANSAC<P3PEstimator> ransac(options);
      const auto report = ransac.Estimate(points2D, points3D);

      EXPECT_TRUE(report.success);
      EXPECT_LT((expected_cam_from_world.ToMatrix() - report.model).norm(),
                1e-3);

      // Test residuals of exact points.
      std::vector<double> residuals;
      P3PEstimator::Residuals(points2D, points3D, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_LT(residuals[i], 1e-3);
      }

      // Test residuals of faulty points.
      P3PEstimator::Residuals(
          points2D, points3D_faulty, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_GT(residuals[i], 0.1);
      }
    }
  }
}

TEST(AbsolutePose, P4PF) {
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(1, 1, 1),
      Eigen::Vector3d(0, 1, 1),
      Eigen::Vector3d(3, 1.0, 4),
      Eigen::Vector3d(3, 1.1, 4),
      Eigen::Vector3d(3, 1.2, 4),
      Eigen::Vector3d(3, 1.3, 4),
      Eigen::Vector3d(3, 1.4, 4),
      Eigen::Vector3d(2, 1, 7),
  };

  auto points3D_faulty = points3D;
  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D_faulty[i](0) = 20;
  }

  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double qx = 0; qx < 1; qx += 0.2) {
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
    for (double tx = 0; tx < 1; tx += 0.1) {
      // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
      for (double f = 0.5; f < 20; f += 2) {
        const Rigid3d expected_cam_from_world(
            Eigen::Quaterniond(1, qx, 0, 0).normalized(),
            Eigen::Vector3d(tx, 0, 0));

        // Project points to camera coordinate system.
        std::vector<Eigen::Vector2d> points2D;
        for (size_t i = 0; i < points3D.size(); ++i) {
          points2D.push_back(
              f * (expected_cam_from_world * points3D[i]).hnormalized());
        }

        RANSACOptions options;
        options.max_error = 1e-5;
        RANSAC<P4PFEstimator> ransac(options);
        const auto report = ransac.Estimate(points2D, points3D);

        EXPECT_TRUE(report.success);
        EXPECT_NEAR(report.model.focal_length, f, 1e-3);
        EXPECT_LT(
            (expected_cam_from_world.ToMatrix() - report.model.cam_from_world)
                .norm(),
            1e-3);

        // Test residuals of exact points.
        std::vector<double> residuals;
        P4PFEstimator::Residuals(points2D, points3D, report.model, &residuals);
        for (size_t i = 0; i < residuals.size(); ++i) {
          EXPECT_LT(residuals[i], 1e-3);
        }

        // Test residuals of faulty points.
        P4PFEstimator::Residuals(
            points2D, points3D_faulty, report.model, &residuals);
        for (size_t i = 0; i < residuals.size(); ++i) {
          EXPECT_GT(residuals[i], 0.1);
        }
      }
    }
  }
}

TEST(AbsolutePose, EPNP) {
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(1, 1, 1),
      Eigen::Vector3d(0, 1, 1),
      Eigen::Vector3d(3, 1.0, 4),
      Eigen::Vector3d(3, 1.1, 4),
      Eigen::Vector3d(3, 1.2, 4),
      Eigen::Vector3d(3, 1.3, 4),
      Eigen::Vector3d(3, 1.4, 4),
      Eigen::Vector3d(2, 1, 7),
  };

  auto points3D_faulty = points3D;
  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D_faulty[i](0) = 20;
  }

  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double qx = 0; qx < 1; qx += 0.2) {
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
    for (double tx = 0; tx < 1; tx += 0.1) {
      const Rigid3d expected_cam_from_world(
          Eigen::Quaterniond(1, qx, 0, 0).normalized(),
          Eigen::Vector3d(tx, 0, 0));

      // Project points to camera coordinate system.
      std::vector<Eigen::Vector2d> points2D;
      for (size_t i = 0; i < points3D.size(); ++i) {
        points2D.push_back(
            (expected_cam_from_world * points3D[i]).hnormalized());
      }

      RANSACOptions options;
      options.max_error = 1e-5;
      RANSAC<EPNPEstimator> ransac(options);
      const auto report = ransac.Estimate(points2D, points3D);

      EXPECT_TRUE(report.success);
      EXPECT_LT((expected_cam_from_world.ToMatrix() - report.model).norm(),
                1e-3);

      // Test residuals of exact points.
      std::vector<double> residuals;
      EPNPEstimator::Residuals(points2D, points3D, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_LT(residuals[i], 1e-3);
      }

      // Test residuals of faulty points.
      EPNPEstimator::Residuals(
          points2D, points3D_faulty, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_GT(residuals[i], 0.1);
      }
    }
  }
}

TEST(AbsolutePose, EPNP_BrokenSolveSignCase) {
  std::vector<Eigen::Vector2d> points2D;
  points2D.emplace_back(-2.6783007931074532e-01, 5.3457197430746251e-01);
  points2D.emplace_back(-4.2629907287470264e-01, 7.5623350319519789e-01);
  points2D.emplace_back(-1.6767413005963930e-01, -1.3387172544910089e-01);
  points2D.emplace_back(-5.6616329720373559e-02, 2.3621156497739373e-01);
  points2D.emplace_back(-1.7721225948969935e-01, 2.3395366792735982e-02);
  points2D.emplace_back(-5.1836259886632222e-02, -4.4380694271927049e-02);
  points2D.emplace_back(-3.5897765845560037e-01, 1.6252721078589397e-01);
  points2D.emplace_back(2.7057324473684058e-01, -1.4067450104631887e-01);
  points2D.emplace_back(-2.5811166424334520e-01, 8.0167171300227366e-02);
  points2D.emplace_back(2.0239567448222310e-02, -3.2845953375344145e-01);
  points2D.emplace_back(4.2571014715170657e-01, -2.8321173570154773e-01);
  points2D.emplace_back(-5.4597596412987237e-01, 9.1431935871671977e-02);

  std::vector<Eigen::Vector3d> points3D;
  points3D.emplace_back(
      4.4276865308679305e+00, -1.3384364366019632e+00, -3.5997423085253892e+00);
  points3D.emplace_back(
      2.7278555252512309e+00, -3.8152996187231392e-01, -2.6558518399902824e+00);
  points3D.emplace_back(
      4.8548566083054894e+00, -1.4756197433631739e+00, -6.8274946022490501e-01);
  points3D.emplace_back(
      3.1523013527998449e+00, -1.3377020437938025e+00, -1.6443269301929087e+00);
  points3D.emplace_back(
      3.8551679771512073e+00, -1.0557700545885551e+00, -1.1695994508851486e+00);
  points3D.emplace_back(
      5.9571373150353812e+00, -2.6120646101684555e+00, -1.0841441206050342e+00);
  points3D.emplace_back(
      6.3287088499358894e+00, -1.1761274755817175e+00, -2.5951879774151583e+00);
  points3D.emplace_back(
      2.3005305990121250e+00, -1.4019796626800123e+00, -4.4485464455072321e-01);
  points3D.emplace_back(
      5.9816859934587354e+00, -1.4211814511691452e+00, -2.0285923889293449e+00);
  points3D.emplace_back(
      5.2543344690665457e+00, -2.3389255564264144e+00, 4.3708173185524052e-01);
  points3D.emplace_back(
      3.2181599245991688e+00, -2.8906671988445098e+00, 2.6825718150064348e-01);
  points3D.emplace_back(
      4.4592895306946758e+00, -9.1235241641579902e-03, -1.6555237117970871e+00);

  std::vector<EPNPEstimator::M_t> models;
  EPNPEstimator::Estimate(points2D, points3D, &models);

  ASSERT_EQ(models.size(), 1);

  double reproj = 0.0;
  for (size_t i = 0; i < points3D.size(); ++i) {
    reproj +=
        ((models[0] * points3D[i].homogeneous()).hnormalized() - points2D[i])
            .norm();
  }

  EXPECT_TRUE(reproj < 0.2);
}

}  // namespace
}  // namespace colmap
