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

#include "colmap/exe/sphere.h"

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"

#include <cmath>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/LU>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(SphereCubeFaceSpecs, Count) {
  const auto faces = SphereCubeFaceSpecs();
  EXPECT_EQ(faces.size(), 6u);
}

TEST(SphereCubeFaceSpecs, RotationsAreProperRotations) {
  // Every r_sphere_from_face must be an orthogonal matrix with det +1.
  const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
  for (const auto& face : SphereCubeFaceSpecs()) {
    const Eigen::Matrix3d should_be_identity =
        face.r_sphere_from_face.transpose() * face.r_sphere_from_face;
    EXPECT_THAT(should_be_identity, EigenMatrixNear(identity, 1e-12))
        << "face " << face.name << " has non-orthogonal rotation";
    EXPECT_NEAR(face.r_sphere_from_face.determinant(), 1.0, 1e-12)
        << "face " << face.name << " is a reflection, not a rotation";
  }
}

TEST(SphereCubeFaceSpecs, FaceCenterMapsToCanonicalDirection) {
  // For each face, the center ray in face-cam-frame is (0, 0, 1). Rotated
  // through r_sphere_from_face, it should equal the canonical sphere-frame
  // direction for that face (forward for F, backward for B, etc.).
  const Eigen::Vector3d face_center_ray(0, 0, 1);

  struct Expected {
    const char* name;
    Eigen::Vector3d dir;
  };
  const Expected expected[] = {
      {"F", Eigen::Vector3d(0, 0, 1)},   // +Z forward
      {"B", Eigen::Vector3d(0, 0, -1)},  // -Z back
      {"L", Eigen::Vector3d(-1, 0, 0)},  // -X left
      {"R", Eigen::Vector3d(1, 0, 0)},   // +X right
      {"U", Eigen::Vector3d(0, -1, 0)},  // -Y up
      {"D", Eigen::Vector3d(0, 1, 0)},   // +Y down
  };

  const auto faces = SphereCubeFaceSpecs();
  ASSERT_EQ(faces.size(), 6u);
  for (size_t i = 0; i < faces.size(); ++i) {
    const Eigen::Vector3d sphere_dir =
        faces[i].r_sphere_from_face * face_center_ray;
    EXPECT_STREQ(faces[i].name, expected[i].name);
    EXPECT_THAT(sphere_dir, EigenMatrixNear(expected[i].dir, 1e-12))
        << "face " << faces[i].name;
  }
}

TEST(SphereCubeFaceSpecs, CoverFullSphereWithoutOverlapAtFaceCenters) {
  // Every point of the unit sphere projects into exactly one cube face
  // when the face FOV is 90° (the canonical cube map). We sanity-check a
  // grid of sphere directions: each must land in the interior of exactly
  // one face (i.e., pass the "in front of face and within |x/z|, |y/z| < 1"
  // test), except on the face boundaries where two are equidistant.
  const auto faces = SphereCubeFaceSpecs();

  // Sample the unit sphere with a fairly dense grid of azimuth/elevation.
  int covered = 0;
  int boundary = 0;
  int uncovered = 0;
  for (int i = 0; i < 37; ++i) {
    const double theta = -M_PI + (2.0 * M_PI) * i / 36.0;  // azimuth
    for (int j = 0; j < 19; ++j) {
      const double phi = -M_PI / 2.0 + M_PI * j / 18.0;  // elevation
      const Eigen::Vector3d ray_sphere(
          std::cos(phi) * std::sin(theta),
          -std::sin(phi),
          std::cos(phi) * std::cos(theta));

      int num_interior_matches = 0;
      int num_boundary_matches = 0;
      for (const auto& face : faces) {
        const Eigen::Vector3d ray_face =
            face.r_sphere_from_face.transpose() * ray_sphere;
        if (ray_face.z() <= 1e-9) continue;
        const double u = ray_face.x() / ray_face.z();
        const double v = ray_face.y() / ray_face.z();
        const double u_abs = std::abs(u);
        const double v_abs = std::abs(v);
        // 90° FOV cube face: projection stays within |u| < 1 and |v| < 1.
        if (u_abs < 1.0 - 1e-9 && v_abs < 1.0 - 1e-9) {
          ++num_interior_matches;
        } else if (u_abs < 1.0 + 1e-9 && v_abs < 1.0 + 1e-9) {
          ++num_boundary_matches;
        }
      }
      if (num_interior_matches == 1) {
        ++covered;
      } else if (num_interior_matches == 0 && num_boundary_matches >= 1) {
        ++boundary;
      } else {
        ++uncovered;
      }
    }
  }
  // Most sampled directions must land in the interior of exactly one face.
  // Boundary samples (on cube edges/corners) are acceptable; we just need
  // no direction to land outside every face.
  EXPECT_GT(covered, 500) << "too few interior coverage samples";
  EXPECT_EQ(uncovered, 0) << "some directions didn't land in any cube face";
}

TEST(SphereCubeFaceSpecs, ReprojectionIntoFaceImageBounds) {
  // A 3D point at (0, 0, 1) in the sphere frame (directly forward) must
  // project into the F face at the image center, and not be visible in any
  // other face with Z > 0.
  const auto faces = SphereCubeFaceSpecs();
  const Eigen::Vector3d point_sphere(0, 0, 1);
  const int face_size = 1024;
  const double fov_deg = 90.0;
  const double focal_px =
      face_size / (2.0 * std::tan(fov_deg * M_PI / 180.0 / 2.0));
  const double cx = face_size / 2.0;
  const double cy = face_size / 2.0;

  std::vector<std::string> hits;
  for (const auto& face : faces) {
    const Eigen::Vector3d point_face =
        face.r_sphere_from_face.transpose() * point_sphere;
    if (point_face.z() <= 1e-9) continue;
    const double u = focal_px * point_face.x() / point_face.z() + cx;
    const double v = focal_px * point_face.y() / point_face.z() + cy;
    if (u >= 0 && u < face_size && v >= 0 && v < face_size) {
      hits.emplace_back(face.name);
      if (std::string(face.name) == "F") {
        EXPECT_NEAR(u, cx, 1e-9);
        EXPECT_NEAR(v, cy, 1e-9);
      }
    }
  }
  // The forward ray should hit exactly face F.
  ASSERT_EQ(hits.size(), 1u) << "forward ray hit " << hits.size() << " faces";
  EXPECT_EQ(hits[0], "F");
}

}  // namespace
}  // namespace colmap
