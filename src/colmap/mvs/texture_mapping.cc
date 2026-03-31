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

#include "colmap/mvs/texture_mapping.h"

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <variant>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#if defined(COLMAP_CGAL_ENABLED)
#include <CGAL/version.h>
#if CGAL_VERSION_MAJOR >= 6
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#else
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#endif
#include <CGAL/AABB_tree.h>
#include <CGAL/Simple_cartesian.h>
#endif  // COLMAP_CGAL_ENABLED

#ifdef _OPENMP
#include <omp.h>
#endif

namespace colmap {
namespace mvs {
namespace {

// Map from face index to its adjacent face indices.
using FaceAdjacencyMap = std::vector<std::vector<size_t>>;

// A connected component of faces sharing the same view.
struct FaceRegion {
  int view_id = -1;
  std::vector<size_t> face_ids;
};

// Axis-aligned bounding box in 2D (integer coordinates).
struct PackRect {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
  size_t region_idx = 0;
};

// Per-region projected vertex positions and bounding box.
struct RegionProjection {
  std::vector<std::array<Eigen::Vector2f, 3>> face_projections;
  int bbox_x = 0;
  int bbox_y = 0;
  int bbox_width = 0;
  int bbox_height = 0;
};

// Atlas layout after packing.
struct AtlasLayout {
  int atlas_width = 0;
  int atlas_height = 0;
  std::vector<PackRect> placements;
};

inline uint64_t EdgeKey(size_t a, size_t b) {
  if (a > b) std::swap(a, b);
  return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

Eigen::Vector3f GetVertex(const PlyMesh& mesh, const size_t idx) {
  return Eigen::Vector3f(
      mesh.vertices[idx].x, mesh.vertices[idx].y, mesh.vertices[idx].z);
}

std::array<size_t, 3> GetFaceIndices(const PlyMeshFace& face) {
  return {face.vertex_idx1, face.vertex_idx2, face.vertex_idx3};
}

Eigen::Vector2f ProjectPoint(const float* P, const Eigen::Vector3f& point) {
  const Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> P_m(P);
  const Eigen::Vector4f ph(point.x(), point.y(), point.z(), 1.0f);
  const Eigen::Vector3f proj = P_m * ph;
  return Eigen::Vector2f(proj(0) / proj(2), proj(1) / proj(2));
}

float ProjectPointDepth(const float* P, const Eigen::Vector3f& point) {
  const Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> P_m(P);
  const Eigen::Vector4f ph(point.x(), point.y(), point.z(), 1.0f);
  return (P_m.row(2) * ph)(0, 0);
}

Eigen::Vector3f ComputeCameraCenter(const float* R, const float* T) {
  const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R_m(R);
  const Eigen::Map<const Eigen::Vector3f> T_m(T);
  return -R_m.transpose() * T_m;
}

std::vector<Eigen::Vector3f> ComputeFaceNormals(const PlyMesh& mesh) {
  const size_t num_faces = mesh.faces.size();
  std::vector<Eigen::Vector3f> normals(num_faces);
  for (size_t i = 0; i < num_faces; ++i) {
    const std::array<size_t, 3> idx = GetFaceIndices(mesh.faces[i]);
    const Eigen::Vector3f v0 = GetVertex(mesh, idx[0]);
    const Eigen::Vector3f v1 = GetVertex(mesh, idx[1]);
    const Eigen::Vector3f v2 = GetVertex(mesh, idx[2]);
    const Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
    const float len = n.norm();
    if (len > 1e-10f) {
      normals[i] = n / len;
    } else {
      normals[i] = Eigen::Vector3f::Zero();
    }
  }
  return normals;
}

FaceAdjacencyMap BuildFaceAdjacency(const PlyMesh& mesh) {
  const size_t num_faces = mesh.faces.size();
  std::unordered_map<uint64_t, std::vector<size_t>> edge_to_faces;
  edge_to_faces.reserve(num_faces * 3);

  for (size_t fi = 0; fi < num_faces; ++fi) {
    const std::array<size_t, 3> idx = GetFaceIndices(mesh.faces[fi]);
    for (int e = 0; e < 3; ++e) {
      const uint64_t key = EdgeKey(idx[e], idx[(e + 1) % 3]);
      edge_to_faces[key].push_back(fi);
    }
  }

  FaceAdjacencyMap adjacency(num_faces);
  for (const auto& [edge_key, face_list] : edge_to_faces) {
    if (face_list.size() == 2) {
      adjacency[face_list[0]].push_back(face_list[1]);
      adjacency[face_list[1]].push_back(face_list[0]);
    }
  }

  for (auto& neighbors : adjacency) {
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());
  }

  return adjacency;
}

#if defined(COLMAP_CGAL_ENABLED)

using CGALKernel = CGAL::Simple_cartesian<float>;
using CGALPoint = CGALKernel::Point_3;
using CGALTriangle = CGALKernel::Triangle_3;
using CGALSegment = CGALKernel::Segment_3;
#if CGAL_VERSION_MAJOR >= 6
using CGALPrimitiveId =
    CGAL::AABB_triangle_primitive_3<CGALKernel,
                                    std::vector<CGALTriangle>::const_iterator>;
using CGALTraits = CGAL::AABB_traits_3<CGALKernel, CGALPrimitiveId>;
#else
using CGALPrimitiveId =
    CGAL::AABB_triangle_primitive<CGALKernel,
                                  std::vector<CGALTriangle>::const_iterator>;
using CGALTraits = CGAL::AABB_traits<CGALKernel, CGALPrimitiveId>;
#endif
using CGALAABBTree = CGAL::AABB_tree<CGALTraits>;

struct OcclusionTester {
  std::vector<CGALTriangle> triangles;
  // Maps from index in `triangles` back to the original face index.
  std::vector<size_t> triangle_to_face;
  CGALAABBTree tree;

  void Build(const PlyMesh& mesh) {
    const size_t num_faces = mesh.faces.size();
    triangles.reserve(num_faces);
    triangle_to_face.reserve(num_faces);
    for (size_t i = 0; i < num_faces; ++i) {
      const std::array<size_t, 3> idx = GetFaceIndices(mesh.faces[i]);
      const Eigen::Vector3f v0 = GetVertex(mesh, idx[0]);
      const Eigen::Vector3f v1 = GetVertex(mesh, idx[1]);
      const Eigen::Vector3f v2 = GetVertex(mesh, idx[2]);
      // Skip degenerate (zero-area) triangles that cause CGAL assertion
      // failures in intersection tests.
      const Eigen::Vector3f cross = (v1 - v0).cross(v2 - v0);
      if (cross.squaredNorm() < 1e-20f) continue;
      triangles.emplace_back(CGALPoint(v0.x(), v0.y(), v0.z()),
                             CGALPoint(v1.x(), v1.y(), v1.z()),
                             CGALPoint(v2.x(), v2.y(), v2.z()));
      triangle_to_face.push_back(i);
    }
    tree.rebuild(triangles.begin(), triangles.end());
    tree.accelerate_distance_queries();
  }

  bool IsOccluded(const Eigen::Vector3f& camera_center,
                  const Eigen::Vector3f& vertex,
                  const size_t face_idx) const {
    constexpr float kEps = 1e-4f;
    const Eigen::Vector3f dir = vertex - camera_center;
    const float dist = dir.norm();
    if (dist < kEps) return false;

    const CGALPoint origin(
        camera_center.x(), camera_center.y(), camera_center.z());
    const CGALPoint target(vertex.x(), vertex.y(), vertex.z());
    const CGALSegment segment(origin, target);

    const auto intersection = tree.any_intersection(segment);
    if (!intersection) return false;

    const auto& primitive_id = intersection->second;
    const size_t tri_idx =
        static_cast<size_t>(primitive_id - triangles.begin());
    const size_t hit_face = triangle_to_face[tri_idx];
    if (hit_face == face_idx) return false;

#if CGAL_VERSION_MAJOR >= 6
    const auto* hit_point = std::get_if<CGALPoint>(&(intersection->first));
#else
    const auto* hit_point = boost::get<CGALPoint>(&(intersection->first));
#endif
    if (hit_point) {
      const Eigen::Vector3f hp(hit_point->x(), hit_point->y(), hit_point->z());
      const float hit_dist = (hp - camera_center).norm();
      if (hit_dist < dist - kEps) return true;
    }
    return false;
  }
};

#endif  // COLMAP_CGAL_ENABLED

std::vector<int> SelectViews(const PlyMesh& mesh,
                             const std::vector<Eigen::Vector3f>& face_normals,
                             const std::vector<Image>& images,
                             const FaceAdjacencyMap& adjacency,
                             const MeshTextureMappingOptions& options) {
  const size_t num_faces = mesh.faces.size();
  const size_t num_images = images.size();

  if (num_faces == 0 || num_images == 0) {
    return std::vector<int>(num_faces, -1);
  }

#if defined(COLMAP_CGAL_ENABLED)
  OcclusionTester occlusion_tester;
  occlusion_tester.Build(mesh);
#endif

  // Flat score buffer: scores[fi * num_images + ii].
  std::vector<double> scores(num_faces * num_images, -1.0);

#ifdef _OPENMP
  [[maybe_unused]] const int num_threads =
      options.num_threads > 0
          ? options.num_threads
          : std::max(1, static_cast<int>(omp_get_max_threads()));
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
#endif
  for (int64_t fi = 0; fi < static_cast<int64_t>(num_faces); ++fi) {
    const Eigen::Vector3f& normal = face_normals[fi];
    if (normal.squaredNorm() < 1e-10f) continue;

    const std::array<size_t, 3> idx = GetFaceIndices(mesh.faces[fi]);
    const Eigen::Vector3f v0 = GetVertex(mesh, idx[0]);
    const Eigen::Vector3f v1 = GetVertex(mesh, idx[1]);
    const Eigen::Vector3f v2 = GetVertex(mesh, idx[2]);
    const Eigen::Vector3f centroid = (v0 + v1 + v2) / 3.0f;
    const std::array<Eigen::Vector3f, 3> verts = {v0, v1, v2};

    for (size_t ii = 0; ii < num_images; ++ii) {
      const Image& img = images[ii];
      const Eigen::Vector3f cam_center =
          ComputeCameraCenter(img.GetR(), img.GetT());

      const Eigen::Vector3f view_dir = (cam_center - centroid).normalized();
      const float cos_angle = normal.dot(view_dir);
      if (cos_angle < static_cast<float>(options.min_cos_normal_angle)) {
        continue;
      }

      int visible_count = 0;
      std::array<Eigen::Vector2f, 3> proj;
      bool behind_camera = false;
      for (int vi = 0; vi < 3; ++vi) {
        const float depth = ProjectPointDepth(img.GetP(), verts[vi]);
        if (depth <= 0) {
          behind_camera = true;
          break;
        }
        proj[vi] = ProjectPoint(img.GetP(), verts[vi]);
        if (proj[vi].x() >= 0 &&
            proj[vi].x() < static_cast<float>(img.GetWidth()) &&
            proj[vi].y() >= 0 &&
            proj[vi].y() < static_cast<float>(img.GetHeight())) {
          ++visible_count;
        }
      }
      if (behind_camera) continue;
      if (visible_count < options.min_visible_vertices) continue;

#if defined(COLMAP_CGAL_ENABLED)
      bool occluded = false;
      for (int vi = 0; vi < 3; ++vi) {
        if (occlusion_tester.IsOccluded(cam_center, verts[vi], fi)) {
          occluded = true;
          break;
        }
      }
      if (occluded) continue;
#endif

      const Eigen::Vector2f e1 = proj[1] - proj[0];
      const Eigen::Vector2f e2 = proj[2] - proj[0];
      const double area =
          std::abs(static_cast<double>(e1.x()) * static_cast<double>(e2.y()) -
                   static_cast<double>(e1.y()) * static_cast<double>(e2.x()));
      scores[fi * num_images + ii] = area;
    }
  }

  std::vector<int> view_per_face(num_faces, -1);
  for (size_t fi = 0; fi < num_faces; ++fi) {
    double best_score = -1.0;
    for (size_t ii = 0; ii < num_images; ++ii) {
      const double s = scores[fi * num_images + ii];
      if (s > best_score) {
        best_score = s;
        view_per_face[fi] = static_cast<int>(ii);
      }
    }
  }

  for (int iter = 0; iter < options.view_selection_smoothing_iterations;
       ++iter) {
    std::vector<int> new_views = view_per_face;
    for (size_t fi = 0; fi < num_faces; ++fi) {
      if (view_per_face[fi] < 0) continue;

      std::unordered_map<int, int> label_counts;
      for (const size_t ni : adjacency[fi]) {
        if (view_per_face[ni] >= 0) {
          ++label_counts[view_per_face[ni]];
        }
      }

      int best_label = view_per_face[fi];
      int best_count =
          label_counts.count(best_label) ? label_counts[best_label] : 0;
      for (const auto& [label, count] : label_counts) {
        if (count > best_count && scores[fi * num_images + label] > 0) {
          best_count = count;
          best_label = label;
        }
      }
      new_views[fi] = best_label;
    }
    view_per_face = new_views;
  }

  return view_per_face;
}

std::vector<FaceRegion> ExtractFaceRegions(
    const std::vector<int>& view_per_face,
    const FaceAdjacencyMap& adjacency,
    const size_t num_faces) {
  std::vector<bool> visited(num_faces, false);
  std::vector<FaceRegion> regions;

  for (size_t fi = 0; fi < num_faces; ++fi) {
    if (visited[fi] || view_per_face[fi] < 0) continue;

    FaceRegion region;
    region.view_id = view_per_face[fi];

    std::queue<size_t> queue;
    queue.push(fi);
    visited[fi] = true;

    while (!queue.empty()) {
      const size_t current = queue.front();
      queue.pop();
      region.face_ids.push_back(current);

      for (const size_t ni : adjacency[current]) {
        if (!visited[ni] && view_per_face[ni] == region.view_id) {
          visited[ni] = true;
          queue.push(ni);
        }
      }
    }

    regions.push_back(std::move(region));
  }

  return regions;
}

std::vector<RegionProjection> ComputeRegionProjections(
    const PlyMesh& mesh,
    const std::vector<FaceRegion>& regions,
    const std::vector<Image>& images) {
  std::vector<RegionProjection> projections(regions.size());

  for (size_t ri = 0; ri < regions.size(); ++ri) {
    const FaceRegion& region = regions[ri];
    const Image& img = images[region.view_id];
    RegionProjection& rp = projections[ri];
    rp.face_projections.resize(region.face_ids.size());

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < region.face_ids.size(); ++i) {
      const size_t fi = region.face_ids[i];
      const std::array<size_t, 3> idx = GetFaceIndices(mesh.faces[fi]);
      for (int vi = 0; vi < 3; ++vi) {
        const Eigen::Vector3f v = GetVertex(mesh, idx[vi]);
        const Eigen::Vector2f p = ProjectPoint(img.GetP(), v);
        rp.face_projections[i][vi] = p;
        min_x = std::min(min_x, p.x());
        min_y = std::min(min_y, p.y());
        max_x = std::max(max_x, p.x());
        max_y = std::max(max_y, p.y());
      }
    }

    rp.bbox_x = static_cast<int>(std::floor(min_x));
    rp.bbox_y = static_cast<int>(std::floor(min_y));
    rp.bbox_width = static_cast<int>(std::ceil(max_x)) - rp.bbox_x + 1;
    rp.bbox_height = static_cast<int>(std::ceil(max_y)) - rp.bbox_y + 1;
  }

  return projections;
}

void ScaleRegionProjections(std::vector<RegionProjection>& projections,
                            const double scale) {
  const float sf = static_cast<float>(scale);
  for (auto& rp : projections) {
    for (auto& fp : rp.face_projections) {
      for (auto& p : fp) {
        p *= sf;
      }
    }
    // Recompute bounding box from scaled projections.
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    for (const auto& fp : rp.face_projections) {
      for (const auto& p : fp) {
        min_x = std::min(min_x, p.x());
        min_y = std::min(min_y, p.y());
        max_x = std::max(max_x, p.x());
        max_y = std::max(max_y, p.y());
      }
    }
    rp.bbox_x = static_cast<int>(std::floor(min_x));
    rp.bbox_y = static_cast<int>(std::floor(min_y));
    rp.bbox_width = static_cast<int>(std::ceil(max_x)) - rp.bbox_x + 1;
    rp.bbox_height = static_cast<int>(std::ceil(max_y)) - rp.bbox_y + 1;
  }
}

AtlasLayout PackAtlas(const std::vector<RegionProjection>& projections,
                      const std::vector<FaceRegion>& regions,
                      const int padding) {
  AtlasLayout layout;

  if (projections.empty()) {
    return layout;
  }

  struct RectEntry {
    int width;
    int height;
    size_t region_idx;
  };
  std::vector<RectEntry> rects(projections.size());
  int64_t total_area = 0;
  int max_rect_width = 0;

  for (size_t i = 0; i < projections.size(); ++i) {
    rects[i].width = projections[i].bbox_width + 2 * padding;
    rects[i].height = projections[i].bbox_height + 2 * padding;
    rects[i].region_idx = i;
    total_area += static_cast<int64_t>(rects[i].width) * rects[i].height;
    max_rect_width = std::max(max_rect_width, rects[i].width);
  }

  std::sort(
      rects.begin(), rects.end(), [](const RectEntry& a, const RectEntry& b) {
        return a.height > b.height;
      });

  const int atlas_side = std::max(
      static_cast<int>(std::ceil(std::sqrt(total_area * 1.3))), max_rect_width);
  int atlas_width = 1;
  while (atlas_width < atlas_side) atlas_width *= 2;
  int atlas_height = atlas_width;

  // Shelf packing.
  const auto TryPack = [&](const int aw,
                           const int ah,
                           std::vector<PackRect>& placements_out) -> bool {
    placements_out.resize(rects.size());
    int shelf_x = 0;
    int shelf_y = 0;
    int shelf_height = 0;

    for (size_t i = 0; i < rects.size(); ++i) {
      if (shelf_x + rects[i].width > aw) {
        shelf_y += shelf_height;
        shelf_x = 0;
        shelf_height = 0;
      }
      if (shelf_y + rects[i].height > ah) {
        return false;
      }
      placements_out[i].x = shelf_x + padding;
      placements_out[i].y = shelf_y + padding;
      placements_out[i].width = projections[rects[i].region_idx].bbox_width;
      placements_out[i].height = projections[rects[i].region_idx].bbox_height;
      placements_out[i].region_idx = rects[i].region_idx;
      shelf_x += rects[i].width;
      shelf_height = std::max(shelf_height, rects[i].height);
    }
    return true;
  };

  std::vector<PackRect> placements;
  constexpr int kMaxAtlasDim = 1 << 16;  // 65536
  while (!TryPack(atlas_width, atlas_height, placements)) {
    THROW_CHECK_LE(atlas_width, kMaxAtlasDim)
        << "Atlas dimensions exceeded maximum (" << kMaxAtlasDim << ")";
    atlas_width *= 2;
    atlas_height *= 2;
  }

  // Shrink height to actual used extent.
  int max_used_y = 0;
  for (const PackRect& p : placements) {
    max_used_y = std::max(max_used_y, p.y + p.height + padding);
  }
  atlas_height = std::max(max_used_y, 1);

  // Reorder placements by region index.
  layout.placements.resize(projections.size());
  for (const PackRect& p : placements) {
    layout.placements[p.region_idx] = p;
  }

  layout.atlas_width = atlas_width;
  layout.atlas_height = atlas_height;
  return layout;
}

std::vector<float> ComputeFaceUVs(
    const std::vector<FaceRegion>& regions,
    const std::vector<RegionProjection>& projections,
    const AtlasLayout& layout,
    const size_t num_faces) {
  std::vector<float> uvs(num_faces * 6, 0.0f);
  const float inv_atlas_width = 1.0f / static_cast<float>(layout.atlas_width);
  const float inv_atlas_height = 1.0f / static_cast<float>(layout.atlas_height);

  for (size_t ri = 0; ri < regions.size(); ++ri) {
    const FaceRegion& region = regions[ri];
    const RegionProjection& rp = projections[ri];
    const PackRect& placement = layout.placements[ri];

    for (size_t i = 0; i < region.face_ids.size(); ++i) {
      const size_t fi = region.face_ids[i];
      for (int vi = 0; vi < 3; ++vi) {
        const Eigen::Vector2f& proj = rp.face_projections[i][vi];
        const float atlas_x = proj.x() - rp.bbox_x + placement.x;
        const float atlas_y = proj.y() - rp.bbox_y + placement.y;
        uvs[fi * 6 + vi * 2 + 0] = atlas_x * inv_atlas_width;
        uvs[fi * 6 + vi * 2 + 1] = 1.0f - atlas_y * inv_atlas_height;
      }
    }
  }

  return uvs;
}

// Returns (u, v, w) where P = u*A + v*B + w*C.
Eigen::Vector3f Barycentric(const Eigen::Vector2f& P,
                            const Eigen::Vector2f& A,
                            const Eigen::Vector2f& B,
                            const Eigen::Vector2f& C) {
  const Eigen::Vector2f v0 = B - A;
  const Eigen::Vector2f v1 = C - A;
  const Eigen::Vector2f v2 = P - A;
  const float d00 = v0.dot(v0);
  const float d01 = v0.dot(v1);
  const float d11 = v1.dot(v1);
  const float d20 = v2.dot(v0);
  const float d21 = v2.dot(v1);
  const float denom = d00 * d11 - d01 * d01;
  if (std::abs(denom) < 1e-10f) {
    return Eigen::Vector3f(-1, -1, -1);
  }
  const float v = (d11 * d20 - d01 * d21) / denom;
  const float w = (d00 * d21 - d01 * d20) / denom;
  const float u = 1.0f - v - w;
  return Eigen::Vector3f(u, v, w);
}

// Compute atlas-space vertex positions for a face within a region.
std::array<Eigen::Vector2f, 3> ComputeAtlasVerts(const RegionProjection& rp,
                                                 const PackRect& placement,
                                                 const size_t face_in_region) {
  std::array<Eigen::Vector2f, 3> atlas_verts;
  for (int vi = 0; vi < 3; ++vi) {
    const Eigen::Vector2f& proj = rp.face_projections[face_in_region][vi];
    atlas_verts[vi].x() = proj.x() - rp.bbox_x + placement.x;
    atlas_verts[vi].y() = proj.y() - rp.bbox_y + placement.y;
  }
  return atlas_verts;
}

void BakeTexture(Bitmap* atlas,
                 std::vector<bool>* baked_mask,
                 const PlyMesh& mesh,
                 const std::vector<FaceRegion>& regions,
                 const std::vector<RegionProjection>& projections,
                 const AtlasLayout& layout,
                 const std::vector<Image>& images,
                 const MeshTextureMappingOptions& options) {
  const int aw = layout.atlas_width;
  const int ah = layout.atlas_height;

  baked_mask->assign(static_cast<size_t>(aw) * ah, false);

  for (size_t ri = 0; ri < regions.size(); ++ri) {
    const FaceRegion& region = regions[ri];
    const RegionProjection& rp = projections[ri];
    const PackRect& placement = layout.placements[ri];
    const Image& img = images[region.view_id];
    const Bitmap& src_bmp = img.GetBitmap();

    for (size_t i = 0; i < region.face_ids.size(); ++i) {
      const std::array<Eigen::Vector2f, 3> atlas_verts =
          ComputeAtlasVerts(rp, placement, i);

      // Bounding box with 1-pixel border for seam coverage.
      const int min_px = std::max(
          0,
          static_cast<int>(std::floor(std::min(
              {atlas_verts[0].x(), atlas_verts[1].x(), atlas_verts[2].x()}))) -
              1);
      const int min_py = std::max(
          0,
          static_cast<int>(std::floor(std::min(
              {atlas_verts[0].y(), atlas_verts[1].y(), atlas_verts[2].y()}))) -
              1);
      const int max_px = std::min(
          aw - 1,
          static_cast<int>(std::ceil(std::max(
              {atlas_verts[0].x(), atlas_verts[1].x(), atlas_verts[2].x()}))) +
              1);
      const int max_py = std::min(
          ah - 1,
          static_cast<int>(std::ceil(std::max(
              {atlas_verts[0].y(), atlas_verts[1].y(), atlas_verts[2].y()}))) +
              1);

      const float texture_inv_scale_factor =
          static_cast<float>(1.0 / options.texture_scale_factor);

      for (int py = min_py; py <= max_py; ++py) {
        for (int px = min_px; px <= max_px; ++px) {
          const Eigen::Vector2f pixel_center(px + 0.5f, py + 0.5f);
          const Eigen::Vector3f bary = Barycentric(
              pixel_center, atlas_verts[0], atlas_verts[1], atlas_verts[2]);

          const float min_bary = std::min({bary.x(), bary.y(), bary.z()});
          if (min_bary < -1e-4f) continue;

          const Eigen::Vector2f img_pos =
              (bary.x() * rp.face_projections[i][0] +
               bary.y() * rp.face_projections[i][1] +
               bary.z() * rp.face_projections[i][2]) *
              texture_inv_scale_factor;

          const auto color =
              src_bmp.InterpolateBilinear(static_cast<double>(img_pos.x()),
                                          static_cast<double>(img_pos.y()));
          if (!color) {
            continue;
          }

          atlas->SetPixel(px, py, color->Cast<uint8_t>());
          (*baked_mask)[static_cast<size_t>(py) * aw + px] = true;
        }
      }
    }
  }
}

void ApplyGlobalColorCorrection(
    Bitmap* atlas,
    const PlyMesh& mesh,
    const std::vector<FaceRegion>& regions,
    const std::vector<RegionProjection>& projections,
    const AtlasLayout& layout,
    const std::vector<Image>& images,
    const FaceAdjacencyMap& adjacency,
    const std::vector<int>& view_per_face,
    const std::vector<bool>& baked_mask,
    const MeshTextureMappingOptions& options) {
  struct SeamEdge {
    size_t face_l;
    size_t face_r;
    size_t vert_a;
    size_t vert_b;
  };

  const size_t num_faces = mesh.faces.size();
  std::vector<SeamEdge> seam_edges;
  seam_edges.reserve(num_faces);

  for (size_t fi = 0; fi < num_faces; ++fi) {
    if (view_per_face[fi] < 0) continue;
    const std::array<size_t, 3> idx = GetFaceIndices(mesh.faces[fi]);
    for (int e = 0; e < 3; ++e) {
      const size_t va = idx[e];
      const size_t vb = idx[(e + 1) % 3];
      const uint64_t ekey = EdgeKey(va, vb);

      for (const size_t ni : adjacency[fi]) {
        if (ni <= fi) continue;
        if (view_per_face[ni] < 0) continue;
        if (view_per_face[ni] == view_per_face[fi]) continue;

        const std::array<size_t, 3> nidx = GetFaceIndices(mesh.faces[ni]);
        bool shares_edge = false;
        for (int ne = 0; ne < 3; ++ne) {
          if (EdgeKey(nidx[ne], nidx[(ne + 1) % 3]) == ekey) {
            shares_edge = true;
            break;
          }
        }
        if (shares_edge) {
          seam_edges.push_back({fi, ni, va, vb});
        }
      }
    }
  }

  if (seam_edges.empty()) return;

  // Build per-region vertex-to-variable mapping.
  struct RegionVertexMap {
    std::unordered_map<size_t, size_t> vert_to_var;
  };
  std::vector<RegionVertexMap> region_vert_maps(regions.size());

  std::vector<int> face_to_region(num_faces, -1);
  size_t total_vars = 0;
  for (size_t ri = 0; ri < regions.size(); ++ri) {
    for (const size_t fi : regions[ri].face_ids) {
      face_to_region[fi] = static_cast<int>(ri);
      const std::array<size_t, 3> idx = GetFaceIndices(mesh.faces[fi]);
      for (int vi = 0; vi < 3; ++vi) {
        if (region_vert_maps[ri].vert_to_var.count(idx[vi]) == 0) {
          region_vert_maps[ri].vert_to_var[idx[vi]] = total_vars++;
        }
      }
    }
  }

  if (total_vars == 0) return;

  const int aw = layout.atlas_width;
  const int ah = layout.atlas_height;
  const double beta = options.color_correction_regularization;

  std::vector<std::array<double, 3>> offsets(total_vars, {0.0, 0.0, 0.0});

  // Estimate triplet count: 4 per seam vertex pair + 1 per variable.
  const size_t estimated_triplets = seam_edges.size() * 8 + total_vars;

  for (int ch = 0; ch < 3; ++ch) {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(estimated_triplets);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(total_vars);

    for (const SeamEdge& se : seam_edges) {
      const int ri_l = face_to_region[se.face_l];
      const int ri_r = face_to_region[se.face_r];
      if (ri_l < 0 || ri_r < 0) continue;

      for (const size_t sv : {se.vert_a, se.vert_b}) {
        const auto it_l = region_vert_maps[ri_l].vert_to_var.find(sv);
        const auto it_r = region_vert_maps[ri_r].vert_to_var.find(sv);
        if (it_l == region_vert_maps[ri_l].vert_to_var.end() ||
            it_r == region_vert_maps[ri_r].vert_to_var.end()) {
          continue;
        }

        const size_t var_l = it_l->second;
        const size_t var_r = it_r->second;

        const Eigen::Vector3f vert = GetVertex(mesh, sv);
        const Image& img_l = images[regions[ri_l].view_id];
        const Image& img_r = images[regions[ri_r].view_id];

        const Eigen::Vector2f proj_l = ProjectPoint(img_l.GetP(), vert);
        const Eigen::Vector2f proj_r = ProjectPoint(img_r.GetP(), vert);

        const auto color_l =
            img_l.GetBitmap().InterpolateBilinear(proj_l.x(), proj_l.y());
        const auto color_r =
            img_r.GetBitmap().InterpolateBilinear(proj_r.x(), proj_r.y());
        if (!color_l || !color_r) {
          continue;
        }

        const double f_l = (ch == 0)   ? color_l->r
                           : (ch == 1) ? color_l->g
                                       : color_l->b;
        const double f_r = (ch == 0)   ? color_r->r
                           : (ch == 1) ? color_r->g
                                       : color_r->b;

        triplets.emplace_back(var_l, var_l, 1.0);
        triplets.emplace_back(var_r, var_r, 1.0);
        triplets.emplace_back(var_l, var_r, -1.0);
        triplets.emplace_back(var_r, var_l, -1.0);
        rhs(var_l) += (f_r - f_l);
        rhs(var_r) += (f_l - f_r);
      }
    }

    for (size_t i = 0; i < total_vars; ++i) {
      triplets.emplace_back(i, i, beta);
    }

    Eigen::SparseMatrix<double> A(total_vars, total_vars);
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
      LOG(WARNING) << "Color correction: failed to factorize system";
      return;
    }

    const Eigen::VectorXd x = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
      LOG(WARNING) << "Color correction: failed to solve system";
      return;
    }

    for (size_t i = 0; i < total_vars; ++i) {
      offsets[i][ch] = x(i);
    }
  }

  // Apply offsets to the atlas.
  for (size_t ri = 0; ri < regions.size(); ++ri) {
    const FaceRegion& region = regions[ri];
    const RegionProjection& rp = projections[ri];
    const PackRect& placement = layout.placements[ri];

    for (size_t i = 0; i < region.face_ids.size(); ++i) {
      const size_t fi = region.face_ids[i];
      const std::array<size_t, 3> idx = GetFaceIndices(mesh.faces[fi]);

      std::array<std::array<double, 3>, 3> vert_offsets;
      for (int vi = 0; vi < 3; ++vi) {
        const size_t var_id = region_vert_maps[ri].vert_to_var[idx[vi]];
        vert_offsets[vi] = offsets[var_id];
      }

      const std::array<Eigen::Vector2f, 3> atlas_verts =
          ComputeAtlasVerts(rp, placement, i);

      const int min_px = std::max(
          0,
          static_cast<int>(std::floor(std::min(
              {atlas_verts[0].x(), atlas_verts[1].x(), atlas_verts[2].x()}))));
      const int min_py = std::max(
          0,
          static_cast<int>(std::floor(std::min(
              {atlas_verts[0].y(), atlas_verts[1].y(), atlas_verts[2].y()}))));
      const int max_px = std::min(
          aw - 1,
          static_cast<int>(std::ceil(std::max(
              {atlas_verts[0].x(), atlas_verts[1].x(), atlas_verts[2].x()}))));
      const int max_py = std::min(
          ah - 1,
          static_cast<int>(std::ceil(std::max(
              {atlas_verts[0].y(), atlas_verts[1].y(), atlas_verts[2].y()}))));

      for (int py = min_py; py <= max_py; ++py) {
        for (int px = min_px; px <= max_px; ++px) {
          if (!baked_mask[static_cast<size_t>(py) * aw + px]) continue;

          const Eigen::Vector2f pixel_center(px + 0.5f, py + 0.5f);
          const Eigen::Vector3f bary = Barycentric(
              pixel_center, atlas_verts[0], atlas_verts[1], atlas_verts[2]);
          if (bary.x() < -0.01f || bary.y() < -0.01f || bary.z() < -0.01f) {
            continue;
          }

          std::array<double, 3> offset_interp = {0, 0, 0};
          for (int c = 0; c < 3; ++c) {
            offset_interp[c] = bary.x() * vert_offsets[0][c] +
                               bary.y() * vert_offsets[1][c] +
                               bary.z() * vert_offsets[2][c];
          }

          auto color =
              atlas->GetPixel(px, py).value_or(BitmapColor<uint8_t>(0));
          color.r = static_cast<uint8_t>(
              std::max(0.0, std::min(255.0, color.r + offset_interp[0])));
          color.g = static_cast<uint8_t>(
              std::max(0.0, std::min(255.0, color.g + offset_interp[1])));
          color.b = static_cast<uint8_t>(
              std::max(0.0, std::min(255.0, color.b + offset_interp[2])));
          atlas->SetPixel(px, py, color);
        }
      }
    }
  }
}

void InpaintAtlas(Bitmap* atlas,
                  const std::vector<bool>& baked_mask,
                  const int inpaint_radius) {
  const int aw = atlas->Width();
  const int ah = atlas->Height();

  if (inpaint_radius <= 0 || aw == 0 || ah == 0) return;

  const size_t num_pixels = static_cast<size_t>(aw) * ah;

  std::vector<int> dist(num_pixels, std::numeric_limits<int>::max());
  std::vector<BitmapColor<uint8_t>> fill_colors(num_pixels);
  std::queue<std::pair<int, int>> queue;

  for (int y = 0; y < ah; ++y) {
    for (int x = 0; x < aw; ++x) {
      const size_t idx = static_cast<size_t>(y) * aw + x;
      if (baked_mask[idx]) {
        dist[idx] = 0;
        fill_colors[idx] =
            atlas->GetPixel(x, y).value_or(BitmapColor<uint8_t>(0));
        queue.push({x, y});
      }
    }
  }

  constexpr std::array<int, 4> kDx = {-1, 1, 0, 0};
  constexpr std::array<int, 4> kDy = {0, 0, -1, 1};

  while (!queue.empty()) {
    const auto [cx, cy] = queue.front();
    queue.pop();
    const size_t cidx = static_cast<size_t>(cy) * aw + cx;
    const int cdist = dist[cidx];
    if (cdist >= inpaint_radius) continue;

    for (int d = 0; d < 4; ++d) {
      const int nx = cx + kDx[d];
      const int ny = cy + kDy[d];
      if (nx < 0 || nx >= aw || ny < 0 || ny >= ah) continue;
      const size_t nidx = static_cast<size_t>(ny) * aw + nx;
      if (dist[nidx] <= cdist + 1) continue;
      dist[nidx] = cdist + 1;
      fill_colors[nidx] = fill_colors[cidx];
      queue.push({nx, ny});
    }
  }

  for (int y = 0; y < ah; ++y) {
    for (int x = 0; x < aw; ++x) {
      const size_t idx = static_cast<size_t>(y) * aw + x;
      if (!baked_mask[idx] && dist[idx] <= inpaint_radius) {
        atlas->SetPixel(x, y, fill_colors[idx]);
      }
    }
  }
}

}  // namespace

#define PrintOption(option) LOG(INFO) << #option ": " << option

bool MeshTextureMappingOptions::Check() const {
  CHECK_OPTION_GT(min_cos_normal_angle, 0.0);
  CHECK_OPTION_LE(min_cos_normal_angle, 1.0);
  CHECK_OPTION_GE(min_visible_vertices, 1);
  CHECK_OPTION_LE(min_visible_vertices, 3);
  CHECK_OPTION_GE(view_selection_smoothing_iterations, 0);
  CHECK_OPTION_GE(atlas_patch_padding, 0);
  CHECK_OPTION_GE(inpaint_radius, 0);
  CHECK_OPTION_GT(color_correction_regularization, 0.0);
  CHECK_OPTION_GT(texture_scale_factor, 0.0);
  return true;
}

void MeshTextureMappingOptions::Print() const {
  LOG_HEADING2("MeshTextureMappingOptions");
  PrintOption(min_cos_normal_angle);
  PrintOption(min_visible_vertices);
  PrintOption(view_selection_smoothing_iterations);
  PrintOption(atlas_patch_padding);
  PrintOption(inpaint_radius);
  PrintOption(apply_color_correction);
  PrintOption(color_correction_regularization);
  PrintOption(num_threads);
  PrintOption(texture_scale_factor);
}

#undef PrintOption

MeshTextureMappingResult MeshTextureMapping(
    const PlyMesh& mesh,
    const std::vector<Image>& images,
    const MeshTextureMappingOptions& options) {
  THROW_CHECK(options.Check());

#if !defined(COLMAP_CGAL_ENABLED)
  LOG(WARNING) << "CGAL is disabled; occlusion testing will be skipped. "
                  "Some faces may be textured from views where they are "
                  "occluded by other geometry.";
#endif

  MeshTextureMappingResult result;

  if (mesh.faces.empty() || mesh.vertices.empty()) {
    result.face_view_ids.assign(mesh.faces.size(), -1);
    result.face_uvs.assign(mesh.faces.size() * 6, 0.0f);
    return result;
  }

  LOG(INFO) << "Computing face normals...";
  const std::vector<Eigen::Vector3f> face_normals = ComputeFaceNormals(mesh);

  LOG(INFO) << "Building face adjacency...";
  const FaceAdjacencyMap adjacency = BuildFaceAdjacency(mesh);

  LOG(INFO) << "Selecting views for " << mesh.faces.size() << " faces from "
            << images.size() << " images...";
  const std::vector<int> view_per_face =
      SelectViews(mesh, face_normals, images, adjacency, options);
  result.face_view_ids = view_per_face;

  const size_t num_assigned = std::count_if(view_per_face.begin(),
                                            view_per_face.end(),
                                            [](const int v) { return v >= 0; });
  if (num_assigned == 0) {
    LOG(WARNING) << "No faces were assigned to any view";
    result.face_uvs.assign(mesh.faces.size() * 6, 0.0f);
    return result;
  }
  LOG(INFO) << num_assigned << " / " << mesh.faces.size()
            << " faces assigned to views";

  LOG(INFO) << "Extracting face regions...";
  const std::vector<FaceRegion> regions =
      ExtractFaceRegions(view_per_face, adjacency, mesh.faces.size());
  LOG(INFO) << "Found " << regions.size() << " regions";

  LOG(INFO) << "Computing region projections...";
  std::vector<RegionProjection> projections =
      ComputeRegionProjections(mesh, regions, images);

  if (options.texture_scale_factor != 1.0) {
    LOG(INFO) << "Scaling region projections by factor "
              << options.texture_scale_factor << "...";
    ScaleRegionProjections(projections, options.texture_scale_factor);
  }

  LOG(INFO) << "Packing texture atlas...";
  const AtlasLayout layout =
      PackAtlas(projections, regions, options.atlas_patch_padding);
  result.atlas_width = layout.atlas_width;
  result.atlas_height = layout.atlas_height;
  LOG(INFO) << "Atlas size: " << layout.atlas_width << " x "
            << layout.atlas_height;

  LOG(INFO) << "Computing face UVs...";
  result.face_uvs =
      ComputeFaceUVs(regions, projections, layout, mesh.faces.size());

  LOG(INFO) << "Baking texture...";
  result.texture_atlas =
      Bitmap(layout.atlas_width, layout.atlas_height, /*as_rgb=*/true);
  result.texture_atlas.Fill(BitmapColor<uint8_t>(0));
  std::vector<bool> baked_mask;
  BakeTexture(&result.texture_atlas,
              &baked_mask,
              mesh,
              regions,
              projections,
              layout,
              images,
              options);

  if (options.apply_color_correction) {
    LOG(INFO) << "Applying global color correction...";
    ApplyGlobalColorCorrection(&result.texture_atlas,
                               mesh,
                               regions,
                               projections,
                               layout,
                               images,
                               adjacency,
                               view_per_face,
                               baked_mask,
                               options);
  }

  if (options.inpaint_radius > 0) {
    LOG(INFO) << "Inpainting atlas...";
    InpaintAtlas(&result.texture_atlas, baked_mask, options.inpaint_radius);
  }

  LOG(INFO) << "Surface texture mapping complete";
  return result;
}

}  // namespace mvs
}  // namespace colmap
