// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/poisson_meshing.h"

#include "colmap/math/random_eigen.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

void WriteRandomPlyPoints(const std::filesystem::path& path,
                          int num_points = 100) {
  std::vector<PlyPoint> ply_points;
  ply_points.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    const Eigen::Vector3d point3D = RandomEigenVectord<3>();
    PlyPoint ply_point;
    ply_point.x = point3D.x();
    ply_point.y = point3D.y();
    ply_point.z = point3D.z();
    ply_point.nx = 0.0f;
    ply_point.ny = 0.0f;
    ply_point.nz = 1.0f;
    ply_point.r = 0;
    ply_point.g = 64;
    ply_point.b = 128;
    ply_points.push_back(ply_point);
  }
  WriteBinaryPlyPoints(
      path, ply_points, /*write_normal=*/true, /*write_rgb=*/true);
}

TEST(PoissonMeshing, Integration) {
  const auto test_dir = CreateTestDir();
  const auto input_path = test_dir / "points.ply";
  const auto output_path = test_dir / "mesh.ply";
  WriteRandomPlyPoints(input_path);

  PoissonMeshingOptions options;
  options.point_weight = 1.0;
  options.depth = 3;   // Use smaller depth for faster test
  options.trim = 0.0;  // Disable trimming
  options.num_threads = 1;

  EXPECT_TRUE(PoissonMeshing(options, input_path, output_path));

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

TEST(PoissonMeshing, WithTrimming) {
  const auto test_dir = CreateTestDir();
  const auto input_path = test_dir / "points.ply";
  const auto output_path = test_dir / "mesh.ply";
  WriteRandomPlyPoints(input_path);

  PoissonMeshingOptions options;
  options.point_weight = 1.0;
  options.depth = 3;
  options.trim = 5.0;
  options.num_threads = 1;

  EXPECT_TRUE(PoissonMeshing(options, input_path, output_path));
  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  // With random data and trimming, we can't make strong assumptions about
  // the number of vertices, but reading the file ensures valid PLY format.
  EXPECT_GE(mesh_vertices.size(), 0);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
