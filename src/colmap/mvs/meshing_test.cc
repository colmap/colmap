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

#include "colmap/mvs/meshing.h"

#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(PoissonMeshing, Integration) {
  const std::string test_dir = CreateTestDir();
  const std::string input_path = JoinPaths(test_dir, "points.ply");
  const std::string output_path = JoinPaths(test_dir, "mesh.ply");

  constexpr int kNumPoints = 100;
  std::vector<PlyPoint> ply_points;
  ply_points.reserve(kNumPoints);
  for (int i = 0; i < kNumPoints; ++i) {
    const Eigen::Vector3d point3D = Eigen::Vector3d::Random();
    PlyPoint ply_point;
    ply_point.x = point3D.x();
    ply_point.y = point3D.y();
    ply_point.z = point3D.z();
    // Set simple normals pointing upward
    ply_point.nx = 0.0f;
    ply_point.ny = 0.0f;
    ply_point.nz = 1.0f;
    ply_point.r = 0;
    ply_point.g = 64;
    ply_point.b = 128;
    ply_points.push_back(ply_point);
  }
  WriteBinaryPlyPoints(
      input_path, ply_points, /*write_normal=*/true, /*write_rgb=*/true);

  // Run Poisson meshing
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

#if defined(COLMAP_CGAL_ENABLED)

TEST(SparseDelaunayMeshing, Integration) {
  const std::string test_dir = CreateTestDir();
  const std::string sparse_path = JoinPaths(test_dir, "sparse");
  const std::string output_path = JoinPaths(test_dir, "mesh.ply");
  CreateDirIfNotExists(sparse_path);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  reconstruction.Write(sparse_path);

  DelaunayMeshingOptions options;
  options.num_threads = 1;
  SparseDelaunayMeshing(options, sparse_path, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

#endif  // COLMAP_CGAL_ENABLED

}  // namespace
}  // namespace mvs
}  // namespace colmap
