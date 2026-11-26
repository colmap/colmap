// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/delaunay_meshing.h"

#include "colmap/math/random.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <fstream>

#include <gtest/gtest.h>

#if defined(COLMAP_CGAL_ENABLED)
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#endif

namespace colmap {
namespace mvs {

#if defined(COLMAP_CGAL_ENABLED)
using FacetAngleKernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using FacetAngleDelaunay =
    CGAL::Delaunay_triangulation_3<FacetAngleKernel, CGAL::Fast_location>;

double ComputeCosFacetCellAngle(const FacetAngleDelaunay& triangulation,
                                const FacetAngleDelaunay::Facet& facet);
#endif

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

TEST(DelaunayMeshingFacetAngle, LargeFiniteSquaredNorms) {
  std::fenv_t environment;
  ASSERT_EQ(std::fegetenv(&environment), 0);
  struct RestoreEnvironment {
    const std::fenv_t& environment;
    ~RestoreEnvironment() { std::fesetenv(&environment); }
  } restore{environment};
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  for (const int exponent : {0, 104}) {
    SCOPED_TRACE(exponent);
    const double length = std::ldexp(1.0, exponent);
    const double height = std::ldexp(1.0, -exponent);
    FacetAngleDelaunay triangulation;
    triangulation.insert(FacetAngleKernel::Point_3(0, 0, 0));
    triangulation.insert(FacetAngleKernel::Point_3(length, 0, 0));
    triangulation.insert(FacetAngleKernel::Point_3(0, length, 0));
    const FacetAngleKernel::Point_3 top(2 * length, 2 * length, height);
    const auto top_vertex = triangulation.insert(top);
    ASSERT_EQ(triangulation.dimension(), 3);
    const FacetAngleDelaunay::Cell_handle cell =
        triangulation.finite_cells_begin();
    const FacetAngleDelaunay::Facet facet(cell, cell->index(top_vertex));
    const auto triangle = triangulation.triangle(facet);
    const auto orientation =
        CGAL::orientation(triangle[0], triangle[1], triangle[2], top);
    ASSERT_NE(orientation, CGAL::COPLANAR);
    // The analytic center is (L/2,L/2,2L^2/H+H/2). Its cosine rounds to
    // one in the large case; the ordinary L=H=1 cosine is 5/sqrt(27).
    const double magnitude = exponent == 0 ? 5 / std::sqrt(27.0) : 1.0;
    const double expected =
        orientation == CGAL::POSITIVE ? magnitude : -magnitude;
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    const double cosine = ComputeCosFacetCellAngle(triangulation, facet);
    EXPECT_EQ(std::fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW), 0);
    ASSERT_TRUE(std::isfinite(cosine));
    EXPECT_NEAR(cosine, expected, 1e-15);
  }
}

TEST(DelaunayMeshingFacetAngle, SmallFiniteSquaredNorms) {
  constexpr double length = 0x1p-200;
  FacetAngleDelaunay triangulation;
  triangulation.insert(FacetAngleKernel::Point_3(0, 0, 0));
  triangulation.insert(FacetAngleKernel::Point_3(length, 0, 0));
  triangulation.insert(FacetAngleKernel::Point_3(0, length, 0));
  const auto top_vertex =
      triangulation.insert(FacetAngleKernel::Point_3(0, 0, length));
  ASSERT_EQ(triangulation.dimension(), 3);
  const FacetAngleDelaunay::Cell_handle cell =
      triangulation.finite_cells_begin();
  const FacetAngleDelaunay::Facet facet(cell, cell->index(top_vertex));

  // The two squared norms are nonzero, but their product underflows to zero.
  const double cosine = ComputeCosFacetCellAngle(triangulation, facet);
  ASSERT_TRUE(std::isfinite(cosine));
  EXPECT_NEAR(std::abs(cosine), 1.0 / std::sqrt(3.0), 1e-15);
}

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

  EXPECT_TRUE(ExistsFile(output_path));
  const std::vector<PlyPoint> mesh_vertices = ReadPly(output_path);
  EXPECT_GE(mesh_vertices.size(), 3);
}

TEST(SparseDelaunayMeshing, NearlyPlanarPoints) {
  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  const auto output_path = test_dir / "mesh.ply";
  SetPRNGSeed(17);
  Reconstruction reconstruction =
      CreateAndWriteSyntheticReconstruction(sparse_path, 5, 500);
  std::vector<point3D_t> point3D_ids;
  point3D_ids.reserve(reconstruction.NumPoints3D());
  for (const auto point3D_id : reconstruction.Point3DIds()) {
    point3D_ids.push_back(point3D_id);
  }
  std::sort(point3D_ids.begin(), point3D_ids.end());
  for (const auto point3D_id : point3D_ids) {
    auto& position = reconstruction.Point3D(point3D_id).xyz;
    position *= RandomUniformReal<double>(0.2, 1.0);
    position.z() *= 1e-5;
  }
  reconstruction.Write(sparse_path);

  // The thin but still volumetric cloud produces nearly parallel facet
  // normals and cell co-tangents. Rounding just one squared norm to float can
  // make their computed cosine exceed one and create a negative graph weight.
  DelaunayMeshingOptions options;
  options.num_threads = 1;
  SetPRNGSeed(0);
  SparseDelaunayMeshing(options, sparse_path, output_path);

  const auto mesh = ReadPlyMesh(output_path).mesh;
  EXPECT_GE(mesh.vertices.size(), 3);
  EXPECT_FALSE(mesh.faces.empty());
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
