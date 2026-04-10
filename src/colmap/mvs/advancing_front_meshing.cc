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

#include "colmap/util/logging.h"

#if defined(COLMAP_CGAL_ENABLED)

#include "colmap/mvs/fusion.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <vector>

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/Cartesian_converter.h>
#include <CGAL/Filtered_kernel.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/Euler_operations.h>
#include <boost/functional/hash.hpp>
#include <omp.h>

namespace {

// Float32 kernel with exact predicates for memory efficiency and speed.
// Uses filtered predicates wrapping Simple_cartesian<float> to maintain
// robustness while halving memory compared to the default double kernel.
class K : public CGAL::Filtered_kernel_adaptor<CGAL::Type_equality_wrapper<
              CGAL::Simple_cartesian<float>::Base<K>::Type,
              K>> {};

// AFSR-specific triangulation types.
using AFSRVb = CGAL::Advancing_front_surface_reconstruction_vertex_base_3<K>;
using AFSRCb = CGAL::Advancing_front_surface_reconstruction_cell_base_3<K>;
using AFSRTds = CGAL::Triangulation_data_structure_3<AFSRVb, AFSRCb>;
using AFSRTriangulation =
    CGAL::Delaunay_triangulation_3<K, AFSRTds, CGAL::Fast_location>;

using SurfaceMesh = CGAL::Surface_mesh<K::Point_3>;

// AABB tree types for post-filtering.
using AABBPrimitive = CGAL::AABB_face_graph_triangle_primitive<SurfaceMesh>;
using AABBTraits = CGAL::AABB_traits_3<K, AABBPrimitive>;
using AABBTree = CGAL::AABB_tree<AABBTraits>;

// Visibility counter: maps triangulation facets to intersection counts.
using VisibilityCounter =
    std::unordered_map<AFSRTriangulation::Facet,
                       int,
                       boost::hash<AFSRTriangulation::Facet>>;

// Priority functor for the advancing front surface reconstruction.
// Controls which facets are accepted based on edge length and visibility.
struct AFSRPriority {
  AFSRPriority(const double max_edge_length_sq,
               const VisibilityCounter* vis_counter,
               const int max_vis)
      : max_edge_length_sq_(max_edge_length_sq),
        vis_counter_(vis_counter),
        max_vis_(max_vis) {}

  template <typename AdvancingFront, typename Cell_handle>
  double operator()(const AdvancingFront& adv,
                    Cell_handle& c,
                    const int& index) const {
    if (vis_counter_ != nullptr) {
      const auto vis = vis_counter_->find(AFSRTriangulation::Facet(c, index));
      if (vis != vis_counter_->end() && vis->second > max_vis_) {
        return adv.infinity();
      }
    }

    if (max_edge_length_sq_ > 0.0) {
      if (CGAL::squared_distance(c->vertex((index + 1) % 4)->point(),
                                 c->vertex((index + 2) % 4)->point()) >
          max_edge_length_sq_) {
        return adv.infinity();
      }
      if (CGAL::squared_distance(c->vertex((index + 2) % 4)->point(),
                                 c->vertex((index + 3) % 4)->point()) >
          max_edge_length_sq_) {
        return adv.infinity();
      }
      if (CGAL::squared_distance(c->vertex((index + 1) % 4)->point(),
                                 c->vertex((index + 3) % 4)->point()) >
          max_edge_length_sq_) {
        return adv.infinity();
      }
    }

    return adv.smallest_radius_delaunay_sphere(c, index);
  }

 private:
  const double max_edge_length_sq_ = 0.0;
  const VisibilityCounter* const vis_counter_ = nullptr;
  const int max_vis_ = 0;
};

using AFSRReconstruction =
    CGAL::Advancing_front_surface_reconstruction<AFSRTriangulation,
                                                 AFSRPriority>;

// Ray caster through the cells of an AFSR Delaunay triangulation.
// Traces a ray segment and collects all intersected facets.
struct AFSRRayCaster {
  explicit AFSRRayCaster(const AFSRTriangulation& triangulation)
      : triangulation_(triangulation) {
    FindHullFacets();
  }

  void CastRaySegment(
      const K::Segment_3& ray_segment,
      std::vector<AFSRTriangulation::Facet>* intersections) const {
    intersections->clear();

    AFSRTriangulation::Cell_handle next_cell =
        triangulation_.locate(ray_segment.start());

    bool next_cell_found = true;
    while (next_cell_found) {
      next_cell_found = false;

      if (triangulation_.is_infinite(next_cell)) {
        for (const auto& hull_facet : hull_facets_) {
          const K::Triangle_3 triangle = triangulation_.triangle(hull_facet);
          if (CGAL::orientation(
                  triangle[0], triangle[1], triangle[2], ray_segment.start()) ==
              K::Orientation::NEGATIVE) {
            continue;
          }

          if (!CGAL::do_intersect(ray_segment, triangle)) {
            continue;
          }

          intersections->push_back(
              AFSRTriangulation::Facet(hull_facet.first, hull_facet.second));
          next_cell = hull_facet.first->neighbor(hull_facet.second);
          next_cell_found = true;
          break;
        }
      } else {
        for (int i = 0; i < 4; ++i) {
          const K::Triangle_3 triangle = triangulation_.triangle(next_cell, i);
          if (CGAL::orientation(
                  triangle[0], triangle[1], triangle[2], ray_segment.start()) ==
              K::Orientation::NEGATIVE) {
            continue;
          }

          if (!CGAL::do_intersect(ray_segment, triangle)) {
            continue;
          }

          intersections->push_back(AFSRTriangulation::Facet(next_cell, i));
          next_cell = next_cell->neighbor(i);
          next_cell_found = true;
          break;
        }
      }
    }
  }

 private:
  void FindHullFacets() {
    for (auto it = triangulation_.all_cells_begin();
         it != triangulation_.all_cells_end();
         ++it) {
      if (triangulation_.is_infinite(it)) {
        for (int i = 0; i < 4; ++i) {
          if (!triangulation_.is_infinite(it, i)) {
            hull_facets_.emplace_back(it, i);
          }
        }
      }
    }
  }

  const AFSRTriangulation& triangulation_;
  std::vector<AFSRTriangulation::Facet> hull_facets_;
};

// Helper to construct a Surface_mesh with colored vertices from PLY points.
struct MeshConstructor {
  SurfaceMesh& mesh;
  SurfaceMesh::Property_map<SurfaceMesh::Vertex_index, CGAL::Color> vcolors;

  MeshConstructor(SurfaceMesh& mesh,
                  const std::vector<colmap::PlyPoint>& points)
      : mesh(mesh) {
    vcolors =
        mesh.add_property_map<SurfaceMesh::Vertex_index, CGAL::Color>("v:color")
            .first;
    for (const auto& p : points) {
      const auto v = mesh.add_vertex(K::Point_3(p.x, p.y, p.z));
      vcolors[v] = CGAL::Color(p.r, p.g, p.b);
    }
  }

  void AddFacets(const AFSRReconstruction& reconstruction) {
    using FaceIterator = typename AFSRReconstruction::TDS_2::Face_iterator;
    const auto& tds = reconstruction.triangulation_data_structure_2();
    for (FaceIterator fit = tds.faces_begin(); fit != tds.faces_end(); ++fit) {
      if (fit->is_on_surface()) {
        using VD = boost::graph_traits<SurfaceMesh>::vertex_descriptor;
        using ST = boost::graph_traits<SurfaceMesh>::vertices_size_type;
        mesh.add_face(VD(static_cast<ST>(fit->vertex(0)->vertex_3()->id())),
                      VD(static_cast<ST>(fit->vertex(1)->vertex_3()->id())),
                      VD(static_cast<ST>(fit->vertex(2)->vertex_3()->id())));
      }
    }
  }
};

// Convert a CGAL Surface_mesh to COLMAP PlyMesh.
colmap::PlyMesh SurfaceMeshToPlyMesh(const SurfaceMesh& mesh) {
  colmap::PlyMesh ply_mesh;

  const auto vcolors =
      mesh.property_map<SurfaceMesh::Vertex_index, CGAL::Color>("v:color");

  ply_mesh.vertices.reserve(mesh.number_of_vertices());
  if (vcolors.has_value()) {
    const auto& colors = vcolors.value();
    for (const auto v : mesh.vertices()) {
      const auto& p = mesh.point(v);
      const auto& c = colors[v];
      ply_mesh.vertices.emplace_back(
          p.x(), p.y(), p.z(), c.red(), c.green(), c.blue());
    }
  } else {
    for (const auto v : mesh.vertices()) {
      const auto& p = mesh.point(v);
      ply_mesh.vertices.emplace_back(p.x(), p.y(), p.z());
    }
  }

  ply_mesh.faces.reserve(mesh.number_of_faces());
  for (const auto f : mesh.faces()) {
    auto h = mesh.halfedge(f);
    const auto v0 = mesh.target(h);
    h = mesh.next(h);
    const auto v1 = mesh.target(h);
    h = mesh.next(h);
    const auto v2 = mesh.target(h);
    ply_mesh.faces.emplace_back(static_cast<size_t>(v0),
                                static_cast<size_t>(v1),
                                static_cast<size_t>(v2));
  }

  return ply_mesh;
}

// Read camera positions indexed by MVS image index (not image_id).
// The .vis file uses MVS image indices which correspond to the sequential
// position of each image in the reconstruction's RegImageIds() order.
// This matches the indexing used by mvs::Model::ReadFromCOLMAP and
// mvs::StereoFusion when writing the .vis file.
std::vector<Eigen::Vector3f> ReadCameraPositions(
    const std::filesystem::path& sparse_path) {
  colmap::Reconstruction reconstruction;
  reconstruction.Read(sparse_path);

  std::vector<Eigen::Vector3f> positions;
  positions.reserve(reconstruction.NumRegImages());
  for (const auto image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    positions.push_back(image.ProjectionCenter().cast<float>());
  }
  return positions;
}

// Build visibility rays for a set of points from their visibility data.
// Each ray goes from the camera center to the observed point, trimmed by
// ray_trim_offset at the target end to avoid self-intersection.
// global_point_indices maps local point index -> global index in visibility.
// If empty, the identity mapping is assumed (local index == global index).
std::vector<K::Segment_3> BuildVisibilityRays(
    const std::vector<colmap::PlyPoint>& points,
    const std::vector<Eigen::Vector3f>& cam_positions,
    const std::vector<std::vector<int>>& visibility,
    const std::vector<size_t>& global_point_indices,
    double ray_trim_offset) {
  std::vector<K::Segment_3> rays;

  for (size_t local_idx = 0; local_idx < points.size(); ++local_idx) {
    const size_t global_idx = global_point_indices.empty()
                                  ? local_idx
                                  : global_point_indices[local_idx];
    const auto& ply_point = points[local_idx];
    const K::Point_3 target(ply_point.x, ply_point.y, ply_point.z);

    for (const int image_idx : visibility[global_idx]) {
      if (static_cast<size_t>(image_idx) >= cam_positions.size()) {
        continue;
      }

      const auto& cam_pos = cam_positions[image_idx];
      const K::Point_3 source(cam_pos.x(), cam_pos.y(), cam_pos.z());

      const K::Vector_3 direction = target - source;
      const double length = std::sqrt(direction.squared_length());
      if (length <= ray_trim_offset) {
        continue;
      }

      const K::Point_3 trimmed_target =
          target - direction * (ray_trim_offset / length);
      rays.emplace_back(source, trimmed_target);
    }
  }

  return rays;
}

struct VisibilityData {
  std::vector<Eigen::Vector3f> cam_positions;
  std::vector<std::vector<int>> point_visibility;
};

// Load visibility data from a dense COLMAP workspace.
// Returns empty if the visibility file does not exist.
VisibilityData LoadVisibilityData(const std::filesystem::path& workspace_path,
                                  const size_t num_points) {
  const auto vis_path = workspace_path / "fused.ply.vis";
  if (!colmap::ExistsFile(vis_path)) {
    LOG(WARNING) << "Visibility file not found: " << vis_path
                 << ". Proceeding without visibility filtering.";
    return {};
  }

  VisibilityData data;
  data.cam_positions = ReadCameraPositions(workspace_path / "sparse");
  data.point_visibility =
      colmap::mvs::ReadPointsVisibility(vis_path, num_points);
  return data;
}

// Reconstruct a single block of points.
colmap::PlyMesh ReconstructBlock(
    const std::vector<colmap::PlyPoint>& points,
    const std::vector<K::Segment_3>& rays,
    const colmap::mvs::AdvancingFrontMeshingOptions& options) {
  // Build the Surface_mesh with colored vertices.
  SurfaceMesh mesh;
  MeshConstructor constructor(mesh, points);

  // Build 3D Delaunay triangulation from the mesh vertices.
  using CC = CGAL::Cartesian_converter<K, K>;
  CC cc;
  AFSRTriangulation triangulation(
      boost::make_transform_iterator(
          mesh.points().begin(), CGAL::AFSR::Auto_count_cc<K::Point_3, CC>(cc)),
      boost::make_transform_iterator(
          mesh.points().end(), CGAL::AFSR::Auto_count_cc<K::Point_3, CC>(cc)));

  LOG(INFO) << "Built triangulation with " << triangulation.number_of_vertices()
            << " vertices.";

  // Pre-filtering: cast visibility rays through triangulation.
  VisibilityCounter visibility_counter;
  if (!rays.empty() && !options.visibility_post_filtering) {
    LOG(INFO) << "Pre-filtering: casting " << rays.size()
              << " visibility rays...";
    const AFSRRayCaster ray_caster(triangulation);
    const int num_omp_threads = omp_get_max_threads();
    std::vector<VisibilityCounter> thread_counters(num_omp_threads);
#pragma omp parallel
    {
      std::vector<AFSRTriangulation::Facet> intersections;
      auto& local_counter = thread_counters[omp_get_thread_num()];
#pragma omp for schedule(dynamic)
      for (size_t i = 0; i < rays.size(); ++i) {
        ray_caster.CastRaySegment(rays[i], &intersections);
        for (const auto& intersection : intersections) {
          local_counter[intersection]++;
        }
      }
    }
    // Start from the largest thread-local map to minimize rehashing.
    const auto max_it = std::max_element(
        thread_counters.begin(),
        thread_counters.end(),
        [](const auto& a, const auto& b) { return a.size() < b.size(); });
    visibility_counter = std::move(*max_it);
    for (auto& tc : thread_counters) {
      if (&tc == &*max_it) continue;
      for (auto& [facet, count] : tc) {
        visibility_counter[facet] += count;
      }
    }
    LOG(INFO) << "Visibility counter has " << visibility_counter.size()
              << " entries.";
  }

  // Run advancing front surface reconstruction.
  LOG(INFO) << "Running advancing front surface reconstruction...";
  const double max_edge_length_sq =
      options.max_edge_length * options.max_edge_length;
  const VisibilityCounter* vis_ptr =
      visibility_counter.empty() ? nullptr : &visibility_counter;
  AFSRPriority priority(max_edge_length_sq,
                        vis_ptr,
                        options.visibility_filtering_max_intersections);
  AFSRReconstruction reconstruction(triangulation, priority);
  reconstruction.run();

  constructor.AddFacets(reconstruction);

  // Post-filtering: remove faces intersected by visibility rays.
  if (!rays.empty() && options.visibility_post_filtering) {
    LOG(INFO) << "Post-filtering: casting " << rays.size()
              << " visibility rays through mesh...";
    AABBTree tree(faces(mesh).first, faces(mesh).second, mesh);
    const int num_omp_threads = omp_get_max_threads();
    using FaceIndex = SurfaceMesh::Face_index;
    std::vector<std::unordered_map<FaceIndex, int>> thread_counters(
        num_omp_threads);
#pragma omp parallel
    {
      std::vector<AABBTree::Primitive_id> primitives;
      auto& local_counter = thread_counters[omp_get_thread_num()];
#pragma omp for schedule(dynamic)
      for (size_t i = 0; i < rays.size(); ++i) {
        primitives.clear();
        tree.all_intersected_primitives(rays[i],
                                        std::back_inserter(primitives));
        for (const auto& f : primitives) {
          local_counter[f]++;
        }
      }
    }
    // Merge per-thread counters.
    const auto max_it = std::max_element(
        thread_counters.begin(),
        thread_counters.end(),
        [](const auto& a, const auto& b) { return a.size() < b.size(); });
    auto face_vis_counts = std::move(*max_it);
    for (auto& tc : thread_counters) {
      if (&tc == &*max_it) continue;
      for (const auto& [f, count] : tc) {
        face_vis_counts[f] += count;
      }
    }
    // Remove faces exceeding the visibility threshold.
    size_t num_removed = 0;
    for (const auto& [f, count] : face_vis_counts) {
      if (count > options.visibility_filtering_max_intersections) {
        CGAL::Euler::remove_face(mesh.halfedge(f), mesh);
        ++num_removed;
      }
    }
    LOG(INFO) << "Removed " << num_removed << " of " << face_vis_counts.size()
              << " intersected faces.";
  }

  CGAL::Polygon_mesh_processing::remove_isolated_vertices(mesh);
  mesh.collect_garbage();

  LOG(INFO) << "Mesh has " << mesh.number_of_vertices() << " vertices and "
            << mesh.number_of_faces() << " faces.";

  return SurfaceMeshToPlyMesh(mesh);
}

struct BlockGrid {
  // Minimum corner of the point cloud bounding box.
  Eigen::Vector3d min;
  // Side length of each block (in world units).
  double block_size;
  // Overlap margin around each block (in world units, derived from
  // block_overlap).
  double overlap;
  // Number of blocks along each axis.
  int nx, ny, nz;

  // Total number of blocks in the grid.
  int NumBlocks() const { return nx * ny * nz; }

  // Convert 3D block coordinates to a flat index.
  int BlockIndex(int bx, int by, int bz) const {
    return bx * ny * nz + by * nz + bz;
  }

  // Convert a flat block index back to 3D block coordinates.
  void BlockCoords(int block_idx, int& bx, int& by, int& bz) const {
    bx = block_idx / (ny * nz);
    by = (block_idx % (ny * nz)) / nz;
    bz = block_idx % nz;
  }

  // Compute the axis-aligned bounding box for a block, including overlap.
  CGAL::Bbox_3 BlockBbox(int bx, int by, int bz) const {
    return CGAL::Bbox_3(min.x() + bx * block_size - overlap,
                        min.y() + by * block_size - overlap,
                        min.z() + bz * block_size - overlap,
                        min.x() + (bx + 1) * block_size + overlap,
                        min.y() + (by + 1) * block_size + overlap,
                        min.z() + (bz + 1) * block_size + overlap);
  }
};

BlockGrid ComputeBlockGrid(const std::vector<colmap::PlyPoint>& points,
                           const double block_size,
                           const double block_overlap) {
  Eigen::AlignedBox3d bbox;
  for (const auto& p : points) {
    bbox.extend(Eigen::Vector3d(p.x, p.y, p.z));
  }
  const Eigen::Vector3d range = bbox.max() - bbox.min();
  BlockGrid grid;
  grid.min = bbox.min();
  grid.block_size = block_size;
  grid.overlap = block_size * block_overlap;
  grid.nx = std::max(1, static_cast<int>(std::ceil(range.x() / block_size)));
  grid.ny = std::max(1, static_cast<int>(std::ceil(range.y() / block_size)));
  grid.nz = std::max(1, static_cast<int>(std::ceil(range.z() / block_size)));
  return grid;
}

std::vector<std::vector<size_t>> AssignPointsToBlocks(
    const std::vector<colmap::PlyPoint>& points, const BlockGrid& grid) {
  const Eigen::Vector3i grid_dims(grid.nx, grid.ny, grid.nz);
  const double inv_block_size = 1.0 / grid.block_size;

  std::vector<std::vector<size_t>> block_point_indices(grid.NumBlocks());
  for (size_t i = 0; i < points.size(); ++i) {
    const Eigen::Vector3d rel(points[i].x - grid.min.x(),
                              points[i].y - grid.min.y(),
                              points[i].z - grid.min.z());
    const Eigen::Vector3i b_min =
        ((rel.array() - grid.overlap) * inv_block_size)
            .floor()
            .cast<int>()
            .max(0);
    const Eigen::Vector3i b_max =
        ((rel.array() + grid.overlap) * inv_block_size)
            .floor()
            .cast<int>()
            .min(grid_dims.array() - 1);

    for (int bx = b_min.x(); bx <= b_max.x(); ++bx) {
      for (int by = b_min.y(); by <= b_max.y(); ++by) {
        for (int bz = b_min.z(); bz <= b_max.z(); ++bz) {
          block_point_indices[grid.BlockIndex(bx, by, bz)].push_back(i);
        }
      }
    }
  }
  return block_point_indices;
}

std::vector<std::vector<size_t>> AssignRaysToBlocks(
    const std::vector<K::Segment_3>& rays,
    const std::vector<std::vector<size_t>>& block_point_indices,
    const BlockGrid& grid) {
  // For each ray, compute the range of blocks its AABB overlaps with,
  // then do an exact segment-box intersection test only on those blocks.
  // This is O(rays * local_blocks) instead of O(rays * total_blocks).
  const Eigen::Vector3i grid_dims(grid.nx, grid.ny, grid.nz);
  const double inv_block_size = 1.0 / grid.block_size;

  std::vector<std::vector<size_t>> block_ray_indices(grid.NumBlocks());
  const size_t log_interval = std::max(rays.size() / 10, size_t{1});
  for (size_t ray_idx = 0; ray_idx < rays.size(); ++ray_idx) {
    if (ray_idx % log_interval == 0) {
      LOG(INFO) << colmap::StringPrintf(
          "Assigning rays to blocks [%d/%d]", ray_idx, rays.size());
    }

    // Compute the ray's AABB in grid coordinates.
    const auto& seg = rays[ray_idx];
    const Eigen::Vector3d p0(seg.source().x() - grid.min.x(),
                             seg.source().y() - grid.min.y(),
                             seg.source().z() - grid.min.z());
    const Eigen::Vector3d p1(seg.target().x() - grid.min.x(),
                             seg.target().y() - grid.min.y(),
                             seg.target().z() - grid.min.z());
    const Eigen::Vector3d ray_min = p0.cwiseMin(p1);
    const Eigen::Vector3d ray_max = p0.cwiseMax(p1);

    // Determine the block range that the ray's AABB overlaps (with overlap).
    const Eigen::Vector3i b_min =
        ((ray_min.array() - grid.overlap) * inv_block_size)
            .floor()
            .cast<int>()
            .max(0);
    const Eigen::Vector3i b_max =
        ((ray_max.array() + grid.overlap) * inv_block_size)
            .floor()
            .cast<int>()
            .min(grid_dims.array() - 1);

    for (int bx = b_min.x(); bx <= b_max.x(); ++bx) {
      for (int by = b_min.y(); by <= b_max.y(); ++by) {
        for (int bz = b_min.z(); bz <= b_max.z(); ++bz) {
          const int idx = grid.BlockIndex(bx, by, bz);
          if (block_point_indices[idx].empty()) {
            continue;
          }
          if (CGAL::do_intersect(seg, grid.BlockBbox(bx, by, bz))) {
            block_ray_indices[idx].push_back(ray_idx);
          }
        }
      }
    }
  }
  return block_ray_indices;
}

std::vector<K::Segment_3> GatherBlockRays(
    const std::vector<K::Segment_3>& all_rays,
    const std::vector<size_t>& ray_indices) {
  std::vector<K::Segment_3> block_rays;
  block_rays.reserve(ray_indices.size());
  for (const size_t ray_idx : ray_indices) {
    block_rays.push_back(all_rays[ray_idx]);
  }
  return block_rays;
}

colmap::PlyMesh CropMeshToRegion(const colmap::PlyMesh& mesh,
                                 const Eigen::AlignedBox3d& crop_box) {
  colmap::PlyMesh cropped;
  cropped.vertices = mesh.vertices;
  for (const auto& face : mesh.faces) {
    const auto& v0 = mesh.vertices[face.vertex_idx1];
    const auto& v1 = mesh.vertices[face.vertex_idx2];
    const auto& v2 = mesh.vertices[face.vertex_idx3];
    const Eigen::Vector3d centroid((v0.x + v1.x + v2.x) / 3.0,
                                   (v0.y + v1.y + v2.y) / 3.0,
                                   (v0.z + v1.z + v2.z) / 3.0);
    if (crop_box.contains(centroid)) {
      cropped.faces.push_back(face);
    }
  }
  return cropped;
}

// Assign points to spatial blocks and reconstruct each block independently.
colmap::PlyMesh ReconstructBlocks(
    const std::vector<colmap::PlyPoint>& points,
    const VisibilityData& vis_data,
    const colmap::mvs::AdvancingFrontMeshingOptions& options) {
  const auto grid =
      ComputeBlockGrid(points, options.block_size, options.block_overlap);

  LOG(INFO) << "Block-wise processing: " << grid.nx << "x" << grid.ny << "x"
            << grid.nz << " = " << grid.NumBlocks()
            << " blocks (size=" << grid.block_size
            << ", overlap=" << grid.overlap << ").";

  const auto block_point_indices = AssignPointsToBlocks(points, grid);

  // Build all visibility rays and pre-assign to blocks by AABB intersection.
  const bool use_vis =
      options.visibility_filtering && !vis_data.cam_positions.empty();
  std::vector<K::Segment_3> all_rays;
  std::vector<std::vector<size_t>> block_ray_indices(grid.NumBlocks());

  if (use_vis) {
    all_rays = BuildVisibilityRays(points,
                                   vis_data.cam_positions,
                                   vis_data.point_visibility,
                                   {},
                                   options.visibility_ray_trim_offset);
    LOG(INFO) << "Built " << all_rays.size()
              << " visibility rays. Assigning to blocks...";
    block_ray_indices = AssignRaysToBlocks(all_rays, block_point_indices, grid);
  }

  colmap::PlyMesh merged_mesh;
  std::mutex merge_mutex;
  std::atomic<int> blocks_completed{0};
  int num_active_blocks = 0;
  for (int i = 0; i < grid.NumBlocks(); ++i) {
    if (!block_point_indices[i].empty()) {
      ++num_active_blocks;
    }
  }

  const int num_threads = colmap::GetEffectiveNumThreads(options.num_threads);
  colmap::ThreadPool thread_pool(num_threads);

  for (int block_idx = 0; block_idx < grid.NumBlocks(); ++block_idx) {
    if (block_point_indices[block_idx].empty()) {
      continue;
    }

    thread_pool.AddTask([&, block_idx]() {
      // Disable OMP parallelism within each block task to avoid
      // oversubscription since ThreadPool handles inter-block parallelism.
      omp_set_num_threads(1);
#ifdef _MSC_VER
      omp_set_nested(0);
#else
      omp_set_max_active_levels(1);
#endif

      const auto& indices = block_point_indices[block_idx];
      std::vector<colmap::PlyPoint> block_points;
      block_points.reserve(indices.size());
      for (const size_t idx : indices) {
        block_points.push_back(points[idx]);
      }

      std::vector<K::Segment_3> block_rays;
      if (use_vis) {
        block_rays = GatherBlockRays(all_rays, block_ray_indices[block_idx]);
      }

      const auto block_mesh =
          ReconstructBlock(block_points, block_rays, options);

      // Crop to the core region plus half the overlap as seam margin.
      int bx, by, bz;
      grid.BlockCoords(block_idx, bx, by, bz);
      const double margin = grid.overlap * 0.5;
      const Eigen::AlignedBox3d crop_box(
          Eigen::Vector3d(grid.min.x() + bx * grid.block_size - margin,
                          grid.min.y() + by * grid.block_size - margin,
                          grid.min.z() + bz * grid.block_size - margin),
          Eigen::Vector3d(grid.min.x() + (bx + 1) * grid.block_size + margin,
                          grid.min.y() + (by + 1) * grid.block_size + margin,
                          grid.min.z() + (bz + 1) * grid.block_size + margin));
      const auto cropped_mesh = CropMeshToRegion(block_mesh, crop_box);

      const int completed = blocks_completed.fetch_add(1) + 1;
      LOG(INFO) << colmap::StringPrintf(
          "Block [%d/%d]: %d points, %d rays, %d faces",
          completed,
          num_active_blocks,
          block_points.size(),
          block_rays.size(),
          cropped_mesh.faces.size());

      std::lock_guard<std::mutex> lock(merge_mutex);
      const size_t vertex_offset = merged_mesh.vertices.size();
      merged_mesh.vertices.insert(merged_mesh.vertices.end(),
                                  cropped_mesh.vertices.begin(),
                                  cropped_mesh.vertices.end());
      for (const auto& face : cropped_mesh.faces) {
        merged_mesh.faces.emplace_back(face.vertex_idx1 + vertex_offset,
                                       face.vertex_idx2 + vertex_offset,
                                       face.vertex_idx3 + vertex_offset);
      }
    });
  }

  thread_pool.Wait();

  LOG(INFO) << "Merged mesh: " << merged_mesh.vertices.size() << " vertices, "
            << merged_mesh.faces.size() << " faces.";

  return merged_mesh;
}

}  // namespace

#endif  // COLMAP_CGAL_ENABLED

namespace colmap {
namespace mvs {

bool AdvancingFrontMeshingOptions::Check() const {
  CHECK_OPTION_GE(max_edge_length, 0);
  CHECK_OPTION_GT(visibility_filtering_max_intersections, 0);
  CHECK_OPTION_GE(visibility_ray_trim_offset, 0);
  CHECK_OPTION_GE(block_size, 0);
  CHECK_OPTION_GT(block_overlap, 0);
  CHECK_OPTION_LE(block_overlap, 1);
  CHECK_OPTION_GE(num_threads, -1);
  CHECK_OPTION_NE(num_threads, 0);
  return true;
}

#if defined(COLMAP_CGAL_ENABLED)

void AdvancingFrontMeshing(const AdvancingFrontMeshingOptions& options,
                           const std::filesystem::path& input_path,
                           const std::filesystem::path& output_path) {
  THROW_CHECK(options.Check());
  THROW_CHECK_HAS_FILE_EXTENSION(output_path, ".ply");
  THROW_CHECK_PATH_OPEN(output_path);

  Timer timer;
  timer.Start();

  // Read input point cloud.
  const auto fused_path = input_path / "fused.ply";
  THROW_CHECK_FILE_EXISTS(fused_path);
  LOG(INFO) << "Reading point cloud from " << fused_path;
  const auto ply_points = ReadPly(fused_path);
  LOG(INFO) << "Read " << ply_points.size() << " points.";

  // Load visibility data if requested.
  VisibilityData vis_data;
  if (options.visibility_filtering) {
    vis_data = LoadVisibilityData(input_path, ply_points.size());
  }

  PlyMesh mesh;
  if (options.block_size > 0) {
    mesh = ReconstructBlocks(ply_points, vis_data, options);
  } else {
    std::vector<K::Segment_3> rays;
    if (!vis_data.cam_positions.empty()) {
      rays = BuildVisibilityRays(ply_points,
                                 vis_data.cam_positions,
                                 vis_data.point_visibility,
                                 {},
                                 options.visibility_ray_trim_offset);
      LOG(INFO) << "Built " << rays.size() << " visibility rays from "
                << ply_points.size() << " points.";
    }
#pragma omp parallel num_threads(1)
    {
      omp_set_num_threads(GetEffectiveNumThreads(options.num_threads));
#ifdef _MSC_VER
      omp_set_nested(1);
#else
      omp_set_max_active_levels(1);
#endif
      mesh = ReconstructBlock(ply_points, rays, options);
    }
  }

  LOG(INFO) << "Writing mesh to " << output_path;
  WriteBinaryPlyMesh(output_path, PlyTexturedMesh{std::move(mesh)});

  timer.PrintSeconds();
}

#endif  // COLMAP_CGAL_ENABLED

}  // namespace mvs
}  // namespace colmap
