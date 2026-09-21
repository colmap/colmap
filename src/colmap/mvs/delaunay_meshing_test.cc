// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/delaunay_meshing.h"

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

TEST(SparseDelaunayMeshing, Integration) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  CreateAndWriteSyntheticReconstruction(sparse_path);

  DelaunayMeshingOptions options;
  options.num_threads = 1;
  SparseDelaunayMeshing(options, sparse_path, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

TEST(SparseDelaunayMeshing, NonSubsampled) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  CreateAndWriteSyntheticReconstruction(sparse_path);

  // Setting max_proj_dist=0 exercises the non-subsampled
  // CreateDelaunayTriangulation() path instead of
  // CreateSubSampledDelaunayTriangulation().
  DelaunayMeshingOptions options;
  options.max_proj_dist = 0;
  options.num_threads = 1;
  SparseDelaunayMeshing(options, sparse_path, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

TEST(DenseDelaunayMeshing, Integration) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  const auto reconstruction =
      CreateAndWriteSyntheticReconstruction(sparse_path, 3, 50);

  // Create fused.ply from reconstruction points
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

  // Create fused.ply.vis: for each point, list visible image indices.
  // Each point is visible in all images to give sufficient multi-view
  // information for the graph-cut optimization.
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

  DelaunayMeshingOptions options;
  options.num_threads = 1;
  DenseDelaunayMeshing(options, test_dir, output_path);

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

#endif  // COLMAP_CGAL_ENABLED

}  // namespace
}  // namespace mvs
}  // namespace colmap
