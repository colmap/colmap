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

#include "colmap/mvs/advancing_front_meshing.h"

#include "colmap/scene/synthetic.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

void WriteRandomPlyPoints(const std::filesystem::path& path,
                          int num_points = 100) {
  std::vector<PlyPoint> ply_points;
  ply_points.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    const Eigen::Vector3d point3D = Eigen::Vector3d::Random();
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

Reconstruction CreateAndWriteSyntheticReconstruction(
    const std::filesystem::path& sparse_path,
    int num_frames = 5,
    int num_points3D = 100) {
  CreateDirIfNotExists(sparse_path);
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = num_frames;
  synthetic_dataset_options.num_points3D = num_points3D;
  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  reconstruction.Write(sparse_path);
  return reconstruction;
}

#if defined(COLMAP_CGAL_ENABLED)

TEST(AdvancingFrontMeshing, NoVisibility) {
  const auto test_dir = CreateTestDir();
  const auto fused_path = test_dir / "fused.ply";
  const auto output_path = test_dir / "mesh.ply";
  WriteRandomPlyPoints(fused_path, 200);

  AdvancingFrontMeshingOptions options;
  options.visibility_filtering = false;
  options.num_threads = 1;

  AdvancingFrontMeshing(options, test_dir, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const auto mesh = ReadPlyMesh(output_path);
  EXPECT_GE(mesh.mesh.vertices.size(), 3);
  EXPECT_GE(mesh.mesh.faces.size(), 1);
}

TEST(AdvancingFrontMeshing, WithMaxEdgeLength) {
  const auto test_dir = CreateTestDir();
  const auto fused_path = test_dir / "fused.ply";
  const auto output_path = test_dir / "mesh.ply";
  WriteRandomPlyPoints(fused_path, 200);

  AdvancingFrontMeshingOptions options;
  options.visibility_filtering = false;
  options.max_edge_length = 0.5;
  options.num_threads = 1;

  AdvancingFrontMeshing(options, test_dir, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const auto mesh = ReadPlyMesh(output_path);
  EXPECT_GE(mesh.mesh.vertices.size(), 3);
}

TEST(AdvancingFrontMeshing, WithVisibility) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  const auto reconstruction =
      CreateAndWriteSyntheticReconstruction(sparse_path, 3, 50);

  // Create fused.ply from reconstruction points.
  std::vector<PlyPoint> ply_points;
  ply_points.reserve(reconstruction.NumPoints3D());
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    PlyPoint ply_point;
    ply_point.x = static_cast<float>(point3D.xyz(0));
    ply_point.y = static_cast<float>(point3D.xyz(1));
    ply_point.z = static_cast<float>(point3D.xyz(2));
    ply_point.nx = 0.0f;
    ply_point.ny = 0.0f;
    ply_point.nz = 1.0f;
    ply_point.r = 128;
    ply_point.g = 128;
    ply_point.b = 128;
    ply_points.push_back(ply_point);
  }
  WriteBinaryPlyPoints(test_dir / "fused.ply",
                       ply_points,
                       /*write_normal=*/true,
                       /*write_rgb=*/true);

  // Create fused.ply.vis.
  const auto vis_path = test_dir / "fused.ply.vis";
  std::fstream vis_file(vis_path, std::ios::out | std::ios::binary);
  THROW_CHECK_FILE_OPEN(vis_file, vis_path);
  const uint64_t num_points = ply_points.size();
  const uint32_t num_visible =
      static_cast<uint32_t>(reconstruction.NumRegImages());
  WriteBinaryLittleEndian<uint64_t>(&vis_file, num_points);
  for (size_t i = 0; i < num_points; ++i) {
    WriteBinaryLittleEndian<uint32_t>(&vis_file, num_visible);
    for (uint32_t j = 0; j < num_visible; ++j) {
      WriteBinaryLittleEndian<uint32_t>(&vis_file, j);
    }
  }
  vis_file.close();

  // Test with post-filtering.
  {
    AdvancingFrontMeshingOptions options;
    options.visibility_filtering = true;
    options.visibility_post_filtering = true;
    options.num_threads = 1;

    AdvancingFrontMeshing(options, test_dir, output_path);

    EXPECT_TRUE(ExistsFile(output_path));
    const auto mesh = ReadPlyMesh(output_path);
    EXPECT_GE(mesh.mesh.vertices.size(), 3);
  }

  // Test with pre-filtering.
  {
    const auto output_path2 = test_dir / "mesh2.ply";
    AdvancingFrontMeshingOptions options;
    options.visibility_filtering = true;
    options.visibility_post_filtering = false;
    options.num_threads = 1;

    AdvancingFrontMeshing(options, test_dir, output_path2);

    EXPECT_TRUE(ExistsFile(output_path2));
    const auto mesh = ReadPlyMesh(output_path2);
    EXPECT_GE(mesh.mesh.vertices.size(), 3);
  }
}

TEST(AdvancingFrontMeshing, BlockWise) {
  const auto test_dir = CreateTestDir();
  const auto fused_path = test_dir / "fused.ply";
  const auto output_path = test_dir / "mesh.ply";
  WriteRandomPlyPoints(fused_path, 500);

  AdvancingFrontMeshingOptions options;
  options.visibility_filtering = false;
  options.block_size = 1.0;
  options.block_overlap = 0.2;
  options.num_threads = 2;

  AdvancingFrontMeshing(options, test_dir, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const auto mesh = ReadPlyMesh(output_path);
  EXPECT_GE(mesh.mesh.vertices.size(), 3);
  EXPECT_GE(mesh.mesh.faces.size(), 1);
}

TEST(AdvancingFrontMeshing, BlockWiseWithVisibility) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  const auto reconstruction =
      CreateAndWriteSyntheticReconstruction(sparse_path, 3, 50);

  std::vector<PlyPoint> ply_points;
  ply_points.reserve(reconstruction.NumPoints3D());
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    PlyPoint ply_point;
    ply_point.x = static_cast<float>(point3D.xyz(0));
    ply_point.y = static_cast<float>(point3D.xyz(1));
    ply_point.z = static_cast<float>(point3D.xyz(2));
    ply_point.nx = 0.0f;
    ply_point.ny = 0.0f;
    ply_point.nz = 1.0f;
    ply_point.r = 128;
    ply_point.g = 128;
    ply_point.b = 128;
    ply_points.push_back(ply_point);
  }
  WriteBinaryPlyPoints(test_dir / "fused.ply",
                       ply_points,
                       /*write_normal=*/true,
                       /*write_rgb=*/true);

  const auto vis_path = test_dir / "fused.ply.vis";
  std::fstream vis_file(vis_path, std::ios::out | std::ios::binary);
  THROW_CHECK_FILE_OPEN(vis_file, vis_path);
  const uint64_t num_points = ply_points.size();
  const uint32_t num_visible =
      static_cast<uint32_t>(reconstruction.NumRegImages());
  WriteBinaryLittleEndian<uint64_t>(&vis_file, num_points);
  for (size_t i = 0; i < num_points; ++i) {
    WriteBinaryLittleEndian<uint32_t>(&vis_file, num_visible);
    for (uint32_t j = 0; j < num_visible; ++j) {
      WriteBinaryLittleEndian<uint32_t>(&vis_file, j);
    }
  }
  vis_file.close();

  AdvancingFrontMeshingOptions options;
  options.visibility_filtering = true;
  options.visibility_post_filtering = true;
  options.block_size = 5.0;
  options.block_overlap = 0.2;
  options.num_threads = 2;

  AdvancingFrontMeshing(options, test_dir, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const auto mesh = ReadPlyMesh(output_path);
  EXPECT_GE(mesh.mesh.vertices.size(), 3);
  EXPECT_GE(mesh.mesh.faces.size(), 1);
}

#endif  // COLMAP_CGAL_ENABLED

}  // namespace
}  // namespace mvs
}  // namespace colmap
