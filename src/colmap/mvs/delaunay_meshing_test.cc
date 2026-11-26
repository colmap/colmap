// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/delaunay_meshing.h"

#include "colmap/math/random.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <tuple>
#include <vector>

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

void ExpectValidMesh(const std::filesystem::path& path) {
  const auto mesh = ReadPlyMesh(path).mesh;
  ASSERT_GE(mesh.vertices.size(), 3);
  ASSERT_FALSE(mesh.faces.empty());
  for (const auto& vertex : mesh.vertices) {
    EXPECT_TRUE(std::isfinite(vertex.x));
    EXPECT_TRUE(std::isfinite(vertex.y));
    EXPECT_TRUE(std::isfinite(vertex.z));
  }
  for (const auto& face : mesh.faces) {
    ASSERT_LT(face.vertex_idx1, mesh.vertices.size());
    ASSERT_LT(face.vertex_idx2, mesh.vertices.size());
    ASSERT_LT(face.vertex_idx3, mesh.vertices.size());
    const auto& a = mesh.vertices[face.vertex_idx1];
    const auto& b = mesh.vertices[face.vertex_idx2];
    const auto& c = mesh.vertices[face.vertex_idx3];
    const Eigen::Vector3d ab(b.x - a.x, b.y - a.y, b.z - a.z);
    const Eigen::Vector3d ac(c.x - a.x, c.y - a.y, c.z - a.z);
    EXPECT_GT(ab.cross(ac).squaredNorm(), 0);
  }
}

TEST(SparseDelaunayMeshing, Integration) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  CreateAndWriteSyntheticReconstruction(sparse_path);

  DelaunayMeshingOptions options;
  options.num_threads = 1;
  SparseDelaunayMeshing(options, sparse_path, output_path);

  ExpectValidMesh(output_path);
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

  ExpectValidMesh(output_path);
}

TEST(SparseDelaunayMeshing, CoincidentPoints) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  Reconstruction reconstruction =
      CreateAndWriteSyntheticReconstruction(sparse_path);
  const auto point3D_ids = reconstruction.Point3DIds();
  const Eigen::Vector3d duplicate_position =
      reconstruction.Point3D(*point3D_ids.begin()).xyz;
  size_t num_duplicates = 0;
  for (const auto point3D_id : point3D_ids) {
    if (++num_duplicates > point3D_ids.size() / 4) {
      break;
    }
    reconstruction.Point3D(point3D_id).xyz = duplicate_position;
  }
  reconstruction.Write(sparse_path);

  DelaunayMeshingOptions options;
  options.num_threads = 1;
  const auto output_path = test_dir / "mesh.ply";
  SparseDelaunayMeshing(options, sparse_path, output_path);

  ExpectValidMesh(output_path);
}

TEST(SparseDelaunayMeshing, PointAtCameraCenter) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  Reconstruction reconstruction =
      CreateAndWriteSyntheticReconstruction(sparse_path);

  // Move one point onto the projection center of an image that observes it.
  // Its viewing ray from that image then has zero length, which used to make
  // the ray caster walk the triangulation forever.
  const point3D_t point3D_id = *reconstruction.Point3DIds().begin();
  Point3D& point3D = reconstruction.Point3D(point3D_id);
  const image_t image_id = point3D.track.Element(0).image_id;
  point3D.xyz = reconstruction.Image(image_id).ProjectionCenter();
  reconstruction.Write(sparse_path);

  DelaunayMeshingOptions options;
  options.num_threads = 1;
  SparseDelaunayMeshing(options, sparse_path, output_path);

  ExpectValidMesh(output_path);
}

TEST(DenseDelaunayMeshing, Integration) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  SetPRNGSeed(17);
  const auto reconstruction =
      CreateAndWriteSyntheticReconstruction(sparse_path, 5, 500);

  std::vector<point3D_t> point3D_ids;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    point3D_ids.push_back(point3D_id);
  }
  std::sort(point3D_ids.begin(), point3D_ids.end());

  // Create fused.ply from reconstruction points
  std::vector<PlyPoint> ply_points;
  ply_points.reserve(reconstruction.NumPoints3D());
  for (const auto point3D_id : point3D_ids) {
    // Include interior points, rather than keeping the synthetic points on a
    // sphere. The dense cloud may differ from the sparse reconstruction.
    const Eigen::Vector3d position = reconstruction.Point3D(point3D_id).xyz *
                                     RandomUniformReal<double>(0.2, 1.0);
    PlyPoint ply_point;
    ply_point.x = static_cast<float>(position(0));
    ply_point.y = static_cast<float>(position(1));
    ply_point.z = static_cast<float>(position(2));
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
  SetPRNGSeed(0);
  DenseDelaunayMeshing(options, test_dir, output_path);

  ExpectValidMesh(output_path);

  // Four unique vertices need not form a 3D cell. Start with eight coplanar or
  // collinear points before later insertions lift the triangulation into 3D.
  std::vector<size_t> point_order(ply_points.size());
  std::iota(point_order.begin(), point_order.end(), 0);
  SetPRNGSeed(0);
  Shuffle(point_order.size(), &point_order);
  for (const bool collinear : {false, true}) {
    SCOPED_TRACE(collinear);
    auto initial_points = ply_points;
    for (size_t i = 0; i < 8; ++i) {
      auto& point = initial_points[point_order[i]];
      point.z = 0;
      if (collinear) {
        point.y = 0;
      }
    }
    WriteBinaryPlyPoints(test_dir / "fused.ply", initial_points, true, true);
    SetPRNGSeed(0);
    DenseDelaunayMeshing(options, test_dir, output_path);
    ExpectValidMesh(output_path);
  }
}

// Final lower-dimensional input is distinct from the temporary prefixes above.
class DelaunayDegenerateInputTest
    : public ::testing::TestWithParam<std::tuple<int, double>> {};

TEST_P(DelaunayDegenerateInputTest, EmptyMesh) {
  const auto [dimension, max_proj_dist] = GetParam();
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  SetPRNGSeed(17);
  auto reconstruction = CreateAndWriteSyntheticReconstruction(
      sparse_path, 5, dimension < 0 ? 0 : 20);
  std::vector<point3D_t> point_ids;
  for (const auto point_id : reconstruction.Point3DIds()) {
    point_ids.push_back(point_id);
  }
  std::sort(point_ids.begin(), point_ids.end());
  for (size_t i = 0; i < point_ids.size(); ++i) {
    auto& position = reconstruction.Point3D(point_ids[i]).xyz;
    if (dimension == 0) {
      position = Eigen::Vector3d(1.25, 0.5, 2.0);
    } else if (dimension == 1) {
      position = Eigen::Vector3d(0.25 * i, 0, 0);
    } else {
      const size_t row = i / 5;
      position = Eigen::Vector3d(0.25 * (i % 5), 0.25 * row, 0);
    }
  }
  reconstruction.Write(sparse_path);

  DelaunayMeshingOptions options;
  options.max_proj_dist = max_proj_dist;
  options.num_threads = 1;
  SetPRNGSeed(0);
  SparseDelaunayMeshing(options, sparse_path, output_path);
  const auto mesh = ReadPlyMesh(output_path).mesh;
  EXPECT_TRUE(mesh.vertices.empty());
  EXPECT_TRUE(mesh.faces.empty());
}

INSTANTIATE_TEST_SUITE_P(FiniteDegeneracy,
                         DelaunayDegenerateInputTest,
                         ::testing::Combine(::testing::Values(-1, 0, 1, 2),
                                            ::testing::Values(0.0, 20.0)));

#endif  // COLMAP_CGAL_ENABLED

}  // namespace
}  // namespace mvs
}  // namespace colmap
