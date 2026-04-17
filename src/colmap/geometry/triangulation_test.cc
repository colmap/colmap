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

#include "colmap/geometry/triangulation.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"

#include <Eigen/Core>
#include <gmock/gmock.h>
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

  const Rigid3d cam1_from_world;

  for (int z = 0; z < 5; ++z) {
    const double qz = z / 5.0;
    for (int tx = 0; tx < 10; tx += 2) {
      const Rigid3d cam2_from_world(Eigen::Quaterniond(0.2, 0.3, 0.4, qz),
                                    Eigen::Vector3d(tx, 2, 3));
      for (size_t i = 0; i < points3D.size(); ++i) {
        const Eigen::Vector3d& point3D = points3D[i];
        const Eigen::Vector2d point1 =
            (cam1_from_world * point3D).hnormalized();
        const Eigen::Vector2d point2 =
            (cam2_from_world * point3D).hnormalized();

        Eigen::Vector3d tri_point3D;
        EXPECT_TRUE(TriangulatePoint(cam1_from_world.ToMatrix(),
                                     cam2_from_world.ToMatrix(),
                                     point1,
                                     point2,
                                     &tri_point3D));

        EXPECT_THAT(point3D, EigenMatrixNear(tri_point3D, 1e-10));
      }
    }
  }
}

TEST(TriangulatePoint, ParallelRays) {
  Eigen::Vector3d point3D;
  EXPECT_FALSE(TriangulatePoint(
      Rigid3d().ToMatrix(),
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0))
          .ToMatrix(),
      Eigen::Vector2d(0, 0),
      Eigen::Vector2d(0, 0),
      &point3D));
}

TEST(TriangulateMidPoint, Nominal) {
  constexpr int kNumTrials = 10;
  for (int i = 0; i < kNumTrials; ++i) {
    const Rigid3d cam1_from_world(Eigen::Quaterniond::UnitRandom(),
                                  Eigen::Vector3d::Random());
    const Rigid3d cam2_from_world(Eigen::Quaterniond::UnitRandom(),
                                  Eigen::Vector3d::Random());
    const Eigen::Vector3d point3D = Eigen::Vector3d::Random();
    const Eigen::Vector3d cam_ray1 = (cam1_from_world * point3D).normalized();
    const Eigen::Vector3d cam_ray2 = (cam2_from_world * point3D).normalized();

    Eigen::Vector3d point3D_in_cam1;
    ASSERT_TRUE(TriangulateMidPoint(cam2_from_world * Inverse(cam1_from_world),
                                    cam_ray1,
                                    cam_ray2,
                                    &point3D_in_cam1));
    const Eigen::Vector3d point3D_in_world =
        Inverse(cam1_from_world) * point3D_in_cam1;
    EXPECT_THAT(point3D, EigenMatrixNear(point3D_in_world, 1e-10));
  }
}

TEST(TriangulateMidPoint, NonPerfectIntersection) {
  const Rigid3d cam1_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(-1, 0, 0));
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0, 0));
  const Eigen::Vector3d expected_point3D(0, 0, 5);
  Eigen::Vector3d point3D_in_cam1;
  EXPECT_TRUE(TriangulateMidPoint(
      cam2_from_world * Inverse(cam1_from_world),
      (cam1_from_world * (expected_point3D + Eigen::Vector3d(0, 0.1, 0)))
          .normalized(),
      (cam2_from_world * (expected_point3D + Eigen::Vector3d(0, -0.1, 0)))
          .normalized(),
      &point3D_in_cam1));
  EXPECT_THAT(point3D_in_cam1,
              EigenMatrixNear(cam1_from_world * expected_point3D, 1e-3));
}

TEST(TriangulateMidPoint, ParallelRays) {
  Eigen::Vector3d point3D_in_cam1;
  EXPECT_FALSE(TriangulateMidPoint(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)),
      Eigen::Vector3d(0, 0, 1),
      Eigen::Vector3d(0, 0, 1),
      &point3D_in_cam1));
}

TEST(TriangulateMidPoint, BehindCameras) {
  Eigen::Vector3d point3D;
  EXPECT_FALSE(TriangulateMidPoint(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)),
      Eigen::Vector3d(0.1, 0, 1).normalized(),
      Eigen::Vector3d(-0.1, 0, 1).normalized(),
      &point3D));
}

TEST(TriangulateMultiViewPoint, Nominal) {
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(0, 0.1, 0.1),
      Eigen::Vector3d(0, 1, 3),
      Eigen::Vector3d(0, 1, 2),
      Eigen::Vector3d(0.01, 0.2, 3),
      Eigen::Vector3d(-1, 0.1, 1),
      Eigen::Vector3d(0.1, 0.1, 0.2),
  };

  const Rigid3d cam1_from_world;

  for (int z = 0; z < 5; ++z) {
    const double qz = z / 5.0;
    for (int tx = 0; tx < 10; tx += 2) {
      const Rigid3d cam2_from_world(Eigen::Quaterniond(0.21, 0.31, 0.41, qz),
                                    Eigen::Vector3d(tx, 2, 3));
      const Rigid3d cam3_from_world(Eigen::Quaterniond(0.2, 0.3, 0.4, qz),
                                    Eigen::Vector3d(tx, 2.1, 3.1));
      for (size_t i = 0; i < points3D.size(); ++i) {
        const Eigen::Vector3d& point3D = points3D[i];
        const Eigen::Vector2d point1 =
            (cam1_from_world * point3D).hnormalized();
        const Eigen::Vector2d point2 =
            (cam2_from_world * point3D).hnormalized();
        const Eigen::Vector2d point3 =
            (cam3_from_world * point3D).hnormalized();

        const std::array<Eigen::Matrix3x4d, 3> cams_from_world = {
            cam1_from_world.ToMatrix(),
            cam2_from_world.ToMatrix(),
            cam3_from_world.ToMatrix()};
        const std::array<Eigen::Vector2d, 3> points = {point1, point2, point3};

        Eigen::Vector3d tri_point3D;
        EXPECT_TRUE(TriangulateMultiViewPoint(
            span<const Eigen::Matrix3x4d>(cams_from_world.data(),
                                          cams_from_world.size()),
            span<const Eigen::Vector2d>(points.data(), points.size()),
            &tri_point3D));

        EXPECT_THAT(point3D, EigenMatrixNear(tri_point3D, 1e-10));
      }
    }
  }
}

TEST(TriangulateMultiViewPointFromBearings, NominalEquivalentTo2D) {
  // For perspective cameras where every 3D point lies in front of every view,
  // the bearing-vector overload must agree with the 2D-normalized overload
  // up to floating-point roundoff.
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(0, 0.1, 0.1),
      Eigen::Vector3d(0, 1, 3),
      Eigen::Vector3d(0, 1, 2),
      Eigen::Vector3d(0.01, 0.2, 3),
      Eigen::Vector3d(-1, 0.1, 1),
      Eigen::Vector3d(0.1, 0.1, 0.2),
  };

  const Rigid3d cam1_from_world;

  for (int z = 0; z < 5; ++z) {
    const double qz = z / 5.0;
    for (int tx = 0; tx < 10; tx += 2) {
      const Rigid3d cam2_from_world(Eigen::Quaterniond(0.21, 0.31, 0.41, qz),
                                    Eigen::Vector3d(tx, 2, 3));
      const Rigid3d cam3_from_world(Eigen::Quaterniond(0.2, 0.3, 0.4, qz),
                                    Eigen::Vector3d(tx, 2.1, 3.1));
      for (size_t i = 0; i < points3D.size(); ++i) {
        const Eigen::Vector3d& point3D = points3D[i];
        const Eigen::Vector3d point1_cam = cam1_from_world * point3D;
        const Eigen::Vector3d point2_cam = cam2_from_world * point3D;
        const Eigen::Vector3d point3_cam = cam3_from_world * point3D;
        const Eigen::Vector2d point1 = point1_cam.hnormalized();
        const Eigen::Vector2d point2 = point2_cam.hnormalized();
        const Eigen::Vector2d point3 = point3_cam.hnormalized();
        const Eigen::Vector3d ray1 = point1_cam.normalized();
        const Eigen::Vector3d ray2 = point2_cam.normalized();
        const Eigen::Vector3d ray3 = point3_cam.normalized();

        const std::array<Eigen::Matrix3x4d, 3> cams_from_world = {
            cam1_from_world.ToMatrix(),
            cam2_from_world.ToMatrix(),
            cam3_from_world.ToMatrix()};
        const std::array<Eigen::Vector2d, 3> points_2d = {point1, point2, point3};
        const std::array<Eigen::Vector3d, 3> rays_3d = {ray1, ray2, ray3};

        Eigen::Vector3d xyz_2d;
        ASSERT_TRUE(TriangulateMultiViewPoint(
            span<const Eigen::Matrix3x4d>(cams_from_world.data(),
                                          cams_from_world.size()),
            span<const Eigen::Vector2d>(points_2d.data(), points_2d.size()),
            &xyz_2d));

        Eigen::Vector3d xyz_3d;
        ASSERT_TRUE(TriangulateMultiViewPoint(
            span<const Eigen::Matrix3x4d>(cams_from_world.data(),
                                          cams_from_world.size()),
            span<const Eigen::Vector3d>(rays_3d.data(), rays_3d.size()),
            &xyz_3d));

        EXPECT_THAT(point3D, EigenMatrixNear(xyz_3d, 1e-10));
        EXPECT_THAT(xyz_2d, EigenMatrixNear(xyz_3d, 1e-10));
      }
    }
  }
}

TEST(TriangulateMultiViewPointFromBearings, BackHemisphereRays) {
  // The bearing-vector path must work for rays that point away from the
  // optical axis (Z <= 0 in camera frame) — the case where the legacy 2D
  // (u, v, 1) representation can't encode the ray. Sets up three cameras
  // looking "outward" from a point, so for each camera one of the other
  // cameras' observations of the landmark is in the back hemisphere.
  //
  // Landmark sits at the origin. Cameras are translated away from it along
  // ±X/±Y and rotated to face outward (i.e., away from the origin) — so from
  // each camera's perspective, the landmark is behind it (back hemisphere).
  //
  // We deliberately use a generic 3-camera configuration: cam1 faces +X,
  // cam2 faces +Y, cam3 faces +Z, all translated 1m along their look direction
  // from the origin so the world origin is "behind" each camera.
  const Eigen::Vector3d world_point(0, 0, 0);

  auto make_cam = [](const Eigen::Vector3d& position_in_world,
                     const Eigen::Matrix3d& world_from_cam) -> Rigid3d {
    // Cam-from-world: R = world_from_cam^T, t = -R * position.
    const Eigen::Matrix3d cam_from_world_rot = world_from_cam.transpose();
    const Eigen::Vector3d cam_from_world_tr =
        -cam_from_world_rot * position_in_world;
    return Rigid3d(Eigen::Quaterniond(cam_from_world_rot).normalized(),
                   cam_from_world_tr);
  };

  // cam1 at (1, 0, 0), looks +X (so world origin is in its back hemisphere).
  Eigen::Matrix3d world_from_cam1;
  world_from_cam1.col(0) = Eigen::Vector3d(0, 0, -1);  // face-cam +X -> world -Z
  world_from_cam1.col(1) = Eigen::Vector3d(0, 1, 0);   // face-cam +Y -> world +Y
  world_from_cam1.col(2) = Eigen::Vector3d(1, 0, 0);   // face-cam +Z -> world +X
  const Rigid3d cam1_from_world =
      make_cam(Eigen::Vector3d(1, 0, 0), world_from_cam1);

  // cam2 at (0, 1, 0), looks +Y.
  Eigen::Matrix3d world_from_cam2;
  world_from_cam2.col(0) = Eigen::Vector3d(1, 0, 0);
  world_from_cam2.col(1) = Eigen::Vector3d(0, 0, -1);
  world_from_cam2.col(2) = Eigen::Vector3d(0, 1, 0);
  const Rigid3d cam2_from_world =
      make_cam(Eigen::Vector3d(0, 1, 0), world_from_cam2);

  // cam3 at (0, 0, 1), looks +Z.
  const Rigid3d cam3_from_world(
      Eigen::Quaterniond::Identity(), Eigen::Vector3d(0, 0, -1));

  // Each camera's ray toward the origin is -world_from_cam.col(2) rotated
  // into camera frame. We compute it via the camera pose directly.
  auto ray_to_origin = [&world_point](const Rigid3d& cam_from_world) {
    const Eigen::Vector3d point_in_cam = cam_from_world * world_point;
    // All three cameras are set up so point_in_cam.z() < 0 (back hemisphere).
    return point_in_cam.normalized();
  };
  const Eigen::Vector3d ray1 = ray_to_origin(cam1_from_world);
  const Eigen::Vector3d ray2 = ray_to_origin(cam2_from_world);
  const Eigen::Vector3d ray3 = ray_to_origin(cam3_from_world);
  ASSERT_LT(ray1.z(), 0) << "cam1 should see origin in back hemisphere";
  ASSERT_LT(ray2.z(), 0) << "cam2 should see origin in back hemisphere";
  ASSERT_LT(ray3.z(), 0) << "cam3 should see origin in back hemisphere";

  const std::array<Eigen::Matrix3x4d, 3> cams_from_world = {
      cam1_from_world.ToMatrix(),
      cam2_from_world.ToMatrix(),
      cam3_from_world.ToMatrix()};
  const std::array<Eigen::Vector3d, 3> rays = {ray1, ray2, ray3};

  Eigen::Vector3d xyz;
  ASSERT_TRUE(TriangulateMultiViewPoint(
      span<const Eigen::Matrix3x4d>(cams_from_world.data(),
                                    cams_from_world.size()),
      span<const Eigen::Vector3d>(rays.data(), rays.size()),
      &xyz));
  EXPECT_THAT(world_point, EigenMatrixNear(xyz, 1e-10));
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
                  tvec1, tvec2, {Eigen::Vector3d(0, 0, 100)})[0],
              0.009999666687,
              1e-8);
  EXPECT_NEAR(CalculateTriangulationAngles(
                  tvec1, tvec2, {Eigen::Vector3d(0, 0, 50)})[0],
              0.019997333973,
              1e-8);
  // Parallel rays.
  EXPECT_THAT(CalculateTriangulationAngles(Eigen::Vector3d::Zero(),
                                           Eigen::Vector3d::Zero(),
                                           {Eigen::Vector3d(0, 0, 0),
                                            Eigen::Vector3d(50, 0, 0),
                                            Eigen::Vector3d(0, 50, 0),
                                            Eigen::Vector3d(0, 0, 50)}),
              testing::Each(testing::DoubleNear(0, 1e-6)));
  // Orthogonal rays.
  EXPECT_THAT(CalculateTriangulationAngles(
                  Eigen::Vector3d::Zero(),
                  Eigen::Vector3d(50, 0, 50),
                  {Eigen::Vector3d(50, 0, 0), Eigen::Vector3d(0, 0, 50)}),
              testing::Each(testing::DoubleNear(EIGEN_PI / 2, 1e-6)));
  // Opposing rays.
  EXPECT_THAT(CalculateTriangulationAngles(Eigen::Vector3d::Zero(),
                                           Eigen::Vector3d(0, 0, 50),
                                           {Eigen::Vector3d(0, 0, 0),
                                            Eigen::Vector3d(0, 0, 50),
                                            Eigen::Vector3d(0, 0, 25),
                                            Eigen::Vector3d(0, 0, -25),
                                            Eigen::Vector3d(0, 0, 75)}),
              testing::Each(testing::DoubleNear(0, 1e-6)));
}

TEST(CalculateAngleBetweenVectors, ParallelVectors) {
  const Eigen::Vector3d v1(1, 0, 0);
  const Eigen::Vector3d v2(2, 0, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), 0.0, 1e-10);
}

TEST(CalculateAngleBetweenVectors, OppositeVectors) {
  const Eigen::Vector3d v1(1, 0, 0);
  const Eigen::Vector3d v2(-1, 0, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), EIGEN_PI, 1e-10);
}

TEST(CalculateAngleBetweenVectors, PerpendicularVectors) {
  const Eigen::Vector3d v1(1, 0, 0);
  const Eigen::Vector3d v2(0, 1, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), EIGEN_PI / 2, 1e-10);
}

TEST(CalculateAngleBetweenVectors, PerpendicularVectorsDifferentMagnitudes) {
  const Eigen::Vector3d v1(3, 0, 0);
  const Eigen::Vector3d v2(0, 5, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), EIGEN_PI / 2, 1e-10);
}

TEST(CalculateAngleBetweenVectors, ZeroVector) {
  const Eigen::Vector3d v1(1, 0, 0);
  const Eigen::Vector3d v2(0, 0, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), 0.0, 1e-10);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v2, v1), 0.0, 1e-10);
}

TEST(CalculateAngleBetweenVectors, BothZeroVectors) {
  const Eigen::Vector3d v1(0, 0, 0);
  const Eigen::Vector3d v2(0, 0, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), 0.0, 1e-10);
}

TEST(CalculateAngleBetweenVectors, FortyFiveDegrees) {
  const Eigen::Vector3d v1(1, 0, 0);
  const Eigen::Vector3d v2(1, 1, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), EIGEN_PI / 4, 1e-10);
}

TEST(CalculateAngleBetweenVectors, IdenticalVectors) {
  const Eigen::Vector3d v1(1, 2, 3);
  const Eigen::Vector3d v2(1, 2, 3);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), 0.0, 1e-10);
}

}  // namespace
}  // namespace colmap
