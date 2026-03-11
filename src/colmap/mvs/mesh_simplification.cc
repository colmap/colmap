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

#include "colmap/mvs/mesh_simplification.h"

#include "colmap/math/math.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <omp.h>

namespace colmap {
namespace mvs {

bool MeshSimplificationOptions::Check() const {
  CHECK_OPTION_GT(target_face_ratio, 0.0);
  CHECK_OPTION_LE(target_face_ratio, 1.0);
  CHECK_OPTION_GE(max_error, 0.0);
  CHECK_OPTION_GE(boundary_weight, 0.0);
  CHECK_OPTION_GE(num_threads, -1);
  CHECK_OPTION_NE(num_threads, 0);
  return true;
}

namespace {

constexpr double kEpsilon = 1e-12;

struct VertexData {
  Eigen::Matrix4d quadric = Eigen::Matrix4d::Zero();
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Vector3f color = Eigen::Vector3f::Constant(200.0f);
  uint32_t timestamp = 0;
  std::vector<size_t> adjacent_faces;
  std::vector<size_t> adjacent_vertices;
  bool removed = false;
};

struct CollapseCandidate {
  double cost = 0.0;
  size_t v1 = 0;
  size_t v2 = 0;
  Eigen::Vector3d optimal_position = Eigen::Vector3d::Zero();
  Eigen::Vector3f optimal_color = Eigen::Vector3f::Zero();
  uint32_t timestamp_v1 = 0;
  uint32_t timestamp_v2 = 0;
};

struct CompareCandidateCost {
  bool operator()(const CollapseCandidate& a,
                  const CollapseCandidate& b) const {
    return a.cost > b.cost;
  }
};

using Edge = std::pair<size_t, size_t>;

Edge MakeEdge(const size_t a, const size_t b) {
  return a < b ? Edge(a, b) : Edge(b, a);
}

struct PairHash {
  size_t operator()(const Edge& e) const {
    const size_t h1 = std::hash<size_t>{}(e.first);
    const size_t h2 = std::hash<size_t>{}(e.second);
    return h1 ^ (h2 * 2654435761u);
  }
};

// Sorted vector helpers for small adjacency lists (~6 elements typical).
// Linear scan beats hash table for these sizes.
void SortedInsert(std::vector<size_t>& vec, const size_t val) {
  const auto it = std::lower_bound(vec.begin(), vec.end(), val);
  if (it == vec.end() || *it != val) {
    vec.insert(it, val);
  }
}

void SortedErase(std::vector<size_t>& vec, const size_t val) {
  const auto it = std::lower_bound(vec.begin(), vec.end(), val);
  if (it != vec.end() && *it == val) {
    vec.erase(it);
  }
}

double ComputeQuadricError(const Eigen::Matrix4d& Q,
                           const Eigen::Vector3d& pos) {
  const Eigen::Vector4d v(pos.x(), pos.y(), pos.z(), 1.0);
  return v.transpose() * Q * v;
}

CollapseCandidate ComputeEdgeCollapse(const std::vector<VertexData>& vertices,
                                      const size_t v1,
                                      const size_t v2,
                                      const bool interpolate_colors) {
  CollapseCandidate candidate;
  candidate.v1 = v1;
  candidate.v2 = v2;
  candidate.timestamp_v1 = vertices[v1].timestamp;
  candidate.timestamp_v2 = vertices[v2].timestamp;

  const Eigen::Matrix4d Q_bar = vertices[v1].quadric + vertices[v2].quadric;

  // Precompute per-vertex errors (used by fallback position solve and
  // non-interpolated color selection).
  const double err_v1 = ComputeQuadricError(Q_bar, vertices[v1].position);
  const double err_v2 = ComputeQuadricError(Q_bar, vertices[v2].position);

  // Try to solve the 4x4 system for optimal position (Garland & Heckbert).
  Eigen::Matrix4d A = Q_bar;
  A.row(3) = Eigen::Vector4d(0, 0, 0, 1);
  const Eigen::Vector4d rhs(0, 0, 0, 1);

  // Fast path: direct inverse for non-singular matrices.
  // For 4x4 matrices, Eigen computes the closed-form inverse which is
  // much cheaper than a full-pivot LU decomposition.
  const double det = A.determinant();
  if (std::abs(det) > kEpsilon) {
    const Eigen::Vector4d v = A.inverse() * rhs;
    candidate.optimal_position = v.head<3>();
    candidate.cost =
        std::max(0.0, ComputeQuadricError(Q_bar, candidate.optimal_position));
  } else {
    // Fallback: evaluate at v1, v2, and midpoint; pick minimum.
    const Eigen::Vector3d mid =
        0.5 * (vertices[v1].position + vertices[v2].position);
    const double err_mid = ComputeQuadricError(Q_bar, mid);

    if (err_v1 <= err_v2 && err_v1 <= err_mid) {
      candidate.optimal_position = vertices[v1].position;
      candidate.cost = std::max(0.0, err_v1);
    } else if (err_v2 <= err_mid) {
      candidate.optimal_position = vertices[v2].position;
      candidate.cost = std::max(0.0, err_v2);
    } else {
      candidate.optimal_position = mid;
      candidate.cost = std::max(0.0, err_mid);
    }
  }

  // Compute optimal color.
  if (interpolate_colors) {
    const Eigen::Vector3d edge_dir =
        vertices[v2].position - vertices[v1].position;
    const double edge_len_sq = edge_dir.squaredNorm();
    float t = 0.5f;
    if (edge_len_sq > 0) {
      t = static_cast<float>(
          (candidate.optimal_position - vertices[v1].position).dot(edge_dir) /
          edge_len_sq);
      t = Clamp(t, 0.0f, 1.0f);
    }
    candidate.optimal_color =
        (1.0f - t) * vertices[v1].color + t * vertices[v2].color;
  } else {
    candidate.optimal_color =
        (err_v1 <= err_v2) ? vertices[v1].color : vertices[v2].color;
  }

  return candidate;
}

bool WouldCauseFlip(const std::vector<VertexData>& vertices,
                    const std::vector<std::array<size_t, 3>>& face_indices,
                    const std::vector<bool>& face_removed,
                    const size_t v1,
                    const size_t v2,
                    const Eigen::Vector3d& new_pos) {
  // Check whether moving v_check to new_pos would flip any of its
  // adjacent faces (excluding faces shared with v_other).
  const auto would_flip_vertex = [&](const size_t v_check,
                                     const size_t v_other) -> bool {
    for (const size_t fi : vertices[v_check].adjacent_faces) {
      if (face_removed[fi]) continue;
      const auto& f = face_indices[fi];
      if (f[0] == v_other || f[1] == v_other || f[2] == v_other) continue;

      const Eigen::Vector3d& p0 = vertices[f[0]].position;
      const Eigen::Vector3d& p1 = vertices[f[1]].position;
      const Eigen::Vector3d& p2 = vertices[f[2]].position;
      const Eigen::Vector3d old_normal = (p1 - p0).cross(p2 - p0);

      Eigen::Vector3d np0 = p0, np1 = p1, np2 = p2;
      if (f[0] == v_check) {
        np0 = new_pos;
      } else if (f[1] == v_check) {
        np1 = new_pos;
      } else if (f[2] == v_check) {
        np2 = new_pos;
      }
      const Eigen::Vector3d new_normal = (np1 - np0).cross(np2 - np0);

      if (old_normal.dot(new_normal) < 0) return true;
    }
    return false;
  };

  return would_flip_vertex(v1, v2) || would_flip_vertex(v2, v1);
}

}  // namespace

PlyMesh SimplifyMesh(const PlyMesh& mesh,
                     const MeshSimplificationOptions& options) {
  THROW_CHECK(options.Check());

  if (mesh.faces.empty() || mesh.vertices.empty()) {
    return mesh;
  }

  // Validate that all face vertex indices are within bounds.
  const size_t num_verts = mesh.vertices.size();
  for (size_t fi = 0; fi < mesh.faces.size(); ++fi) {
    const auto& face = mesh.faces[fi];
    THROW_CHECK_LT(face.vertex_idx1, num_verts)
        << "Face " << fi << " has out-of-bounds vertex index";
    THROW_CHECK_LT(face.vertex_idx2, num_verts)
        << "Face " << fi << " has out-of-bounds vertex index";
    THROW_CHECK_LT(face.vertex_idx3, num_verts)
        << "Face " << fi << " has out-of-bounds vertex index";
  }

  const size_t num_faces = mesh.faces.size();
  const size_t target_faces = std::max(
      static_cast<size_t>(1),
      static_cast<size_t>(std::floor(num_faces * options.target_face_ratio)));

  if (target_faces >= num_faces) {
    return mesh;
  }

  const size_t num_vertices = mesh.vertices.size();
  std::vector<VertexData> vertex_data(num_vertices);
  for (size_t i = 0; i < num_vertices; ++i) {
    vertex_data[i].position = Eigen::Vector3d(
        mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z);
    vertex_data[i].color = Eigen::Vector3f(
        mesh.vertices[i].r, mesh.vertices[i].g, mesh.vertices[i].b);
  }

  // Build mutable face index array.
  std::vector<std::array<size_t, 3>> face_indices(num_faces);
  std::vector<bool> face_removed(num_faces, false);

  // Build topology.
  LOG(INFO) << "Building mesh topology...";
  for (size_t fi = 0; fi < num_faces; ++fi) {
    face_indices[fi] = {mesh.faces[fi].vertex_idx1,
                        mesh.faces[fi].vertex_idx2,
                        mesh.faces[fi].vertex_idx3};
    const auto& f = face_indices[fi];

    // Skip degenerate faces.
    if (f[0] == f[1] || f[0] == f[2] || f[1] == f[2]) {
      face_removed[fi] = true;
      continue;
    }

    for (int j = 0; j < 3; ++j) {
      const size_t va = f[j];
      const size_t vb = f[(j + 1) % 3];
      SortedInsert(vertex_data[va].adjacent_faces, fi);
      SortedInsert(vertex_data[va].adjacent_vertices, vb);
      SortedInsert(vertex_data[vb].adjacent_vertices, va);
    }
  }

  size_t current_faces = 0;
  for (size_t fi = 0; fi < num_faces; ++fi) {
    if (!face_removed[fi]) ++current_faces;
  }

  // Step 1: Compute initial quadrics.
  const int eff_num_threads = GetEffectiveNumThreads(options.num_threads);
  LOG(INFO) << "Computing vertex quadrics using " << eff_num_threads
            << " threads...";

  // Compute per-face quadric matrices in parallel (Section 5).
  std::vector<Eigen::Matrix4d> face_quadrics(num_faces,
                                             Eigen::Matrix4d::Zero());
  {
    const int64_t num_faces_signed = static_cast<int64_t>(num_faces);
#pragma omp parallel for schedule(static) num_threads(eff_num_threads)
    for (int64_t fi = 0; fi < num_faces_signed; ++fi) {
      if (face_removed[fi]) continue;
      const auto& f = face_indices[fi];
      const Eigen::Vector3d& p0 = vertex_data[f[0]].position;
      const Eigen::Vector3d& p1 = vertex_data[f[1]].position;
      const Eigen::Vector3d& p2 = vertex_data[f[2]].position;

      Eigen::Vector3d normal = (p1 - p0).cross(p2 - p0);
      const double len = normal.norm();
      if (len < kEpsilon) continue;
      normal /= len;

      const double d = -normal.dot(p0);
      const Eigen::Vector4d plane(normal.x(), normal.y(), normal.z(), d);
      face_quadrics[fi] = plane * plane.transpose();
    }
  }

  // Accumulate per-face quadrics into vertices (sequential due to shared
  // vertex writes).
  for (size_t fi = 0; fi < num_faces; ++fi) {
    if (face_removed[fi]) continue;
    const auto& f = face_indices[fi];
    vertex_data[f[0]].quadric += face_quadrics[fi];
    vertex_data[f[1]].quadric += face_quadrics[fi];
    vertex_data[f[2]].quadric += face_quadrics[fi];
  }
  face_quadrics.clear();
  face_quadrics.shrink_to_fit();

  // Boundary preservation quadrics (Section 6, geometric discontinuity only).
  if (options.boundary_weight > 0) {
    LOG(INFO) << "Computing boundary preservation quadrics...";
    // Build edge-to-face mapping.
    std::unordered_map<Edge, std::vector<size_t>, PairHash> edge_faces;
    edge_faces.reserve(num_faces * 3 / 2);
    for (size_t fi = 0; fi < num_faces; ++fi) {
      if (face_removed[fi]) continue;
      const auto& f = face_indices[fi];
      for (int j = 0; j < 3; ++j) {
        edge_faces[MakeEdge(f[j], f[(j + 1) % 3])].push_back(fi);
      }
    }

    for (const auto& [edge, faces_list] : edge_faces) {
      if (faces_list.size() != 1) continue;

      const size_t fi = faces_list[0];
      const auto& f = face_indices[fi];
      const Eigen::Vector3d& p0 = vertex_data[f[0]].position;
      const Eigen::Vector3d& p1 = vertex_data[f[1]].position;
      const Eigen::Vector3d& p2 = vertex_data[f[2]].position;
      const Eigen::Vector3d face_normal = (p1 - p0).cross(p2 - p0);

      const Eigen::Vector3d edge_dir =
          vertex_data[edge.second].position - vertex_data[edge.first].position;
      Eigen::Vector3d constraint_normal = edge_dir.cross(face_normal);
      const double clen = constraint_normal.norm();
      if (clen < kEpsilon) continue;
      constraint_normal /= clen;

      const double cd =
          -constraint_normal.dot(vertex_data[edge.first].position);
      const Eigen::Vector4d cplane(constraint_normal.x(),
                                   constraint_normal.y(),
                                   constraint_normal.z(),
                                   cd);
      const Eigen::Matrix4d Q_boundary =
          options.boundary_weight * cplane * cplane.transpose();

      vertex_data[edge.first].quadric += Q_boundary;
      vertex_data[edge.second].quadric += Q_boundary;
    }
  }

  LOG(INFO) << "Computing initial edge collapse candidates...";

  // Step 2: Collect unique edges — only real mesh edges (vi, vj) with vi < vj
  // are considered (not pairs of close-enough vertices as in Section 8).
  // Each undirected edge is emitted exactly once since adjacency is symmetric.
  std::vector<Edge> unique_edges;
  for (size_t vi = 0; vi < num_vertices; ++vi) {
    if (vertex_data[vi].removed) continue;
    for (const size_t vj : vertex_data[vi].adjacent_vertices) {
      if (vi < vj) {
        unique_edges.emplace_back(vi, vj);
      }
    }
  }

  // Step 3: Compute collapse candidates in parallel.
  std::vector<CollapseCandidate> initial_candidates(unique_edges.size());
  {
    const int64_t num_edges_signed = static_cast<int64_t>(unique_edges.size());
#pragma omp parallel for schedule(static) num_threads(eff_num_threads)
    for (int64_t i = 0; i < num_edges_signed; ++i) {
      initial_candidates[i] = ComputeEdgeCollapse(vertex_data,
                                                  unique_edges[i].first,
                                                  unique_edges[i].second,
                                                  options.interpolate_colors);
    }
  }

  // Step 4: Initialize priority queue from the computed candidates in O(n).
  using MinHeap = std::priority_queue<CollapseCandidate,
                                      std::vector<CollapseCandidate>,
                                      CompareCandidateCost>;
  MinHeap pq(CompareCandidateCost{}, std::move(initial_candidates));

  LOG(INFO) << "Collapsing edges: " << current_faces << " -> " << target_faces
            << " target faces, " << pq.size() << " initial edge candidates";

  // Step 5: Iterative collapse loop.
  const size_t faces_to_remove = current_faces - target_faces;
  size_t faces_removed = 0;
  int last_progress_percent = -1;

  while (current_faces > target_faces && !pq.empty()) {
    const CollapseCandidate candidate = pq.top();
    pq.pop();

    // Skip stale candidates (lazy deletion).
    if (vertex_data[candidate.v1].removed ||
        vertex_data[candidate.v2].removed) {
      continue;
    }
    if (vertex_data[candidate.v1].timestamp != candidate.timestamp_v1 ||
        vertex_data[candidate.v2].timestamp != candidate.timestamp_v2) {
      continue;
    }

    // Check max error threshold.
    if (options.max_error > 0 && candidate.cost > options.max_error) {
      LOG(INFO) << "Stopping early: quadric error " << candidate.cost
                << " exceeds max_error " << options.max_error
                << " after removing " << faces_removed << " faces";
      break;
    }

    // Check for face flips.
    if (WouldCauseFlip(vertex_data,
                       face_indices,
                       face_removed,
                       candidate.v1,
                       candidate.v2,
                       candidate.optimal_position)) {
      continue;
    }

    // Execute collapse: merge v2 into v1.
    const size_t v1 = candidate.v1;
    const size_t v2 = candidate.v2;

    vertex_data[v1].position = candidate.optimal_position;
    vertex_data[v1].color = candidate.optimal_color;
    vertex_data[v1].quadric += vertex_data[v2].quadric;

    // Process faces adjacent to v2.
    for (const size_t fi : vertex_data[v2].adjacent_faces) {
      if (face_removed[fi]) continue;
      auto& f = face_indices[fi];

      // Check if this is a shared face (contains both v1 and v2).
      const bool has_v1 = (f[0] == v1 || f[1] == v1 || f[2] == v1);
      if (has_v1) {
        // Shared face: remove it.
        face_removed[fi] = true;
        --current_faces;
        for (int j = 0; j < 3; ++j) {
          if (f[j] != v1 && f[j] != v2) {
            SortedErase(vertex_data[f[j]].adjacent_faces, fi);
          }
        }
        SortedErase(vertex_data[v1].adjacent_faces, fi);
      } else {
        // Remap v2 -> v1 in this face.
        for (int j = 0; j < 3; ++j) {
          if (f[j] == v2) {
            f[j] = v1;
            break;
          }
        }
        // Check for degenerate face after remapping.
        if (f[0] == f[1] || f[0] == f[2] || f[1] == f[2]) {
          face_removed[fi] = true;
          --current_faces;
          for (int j = 0; j < 3; ++j) {
            SortedErase(vertex_data[f[j]].adjacent_faces, fi);
          }
        } else {
          SortedInsert(vertex_data[v1].adjacent_faces, fi);
        }
      }
    }

    // Update vertex adjacency.
    for (const size_t u : vertex_data[v2].adjacent_vertices) {
      if (u == v1) continue;
      SortedErase(vertex_data[u].adjacent_vertices, v2);
      SortedInsert(vertex_data[u].adjacent_vertices, v1);
      SortedInsert(vertex_data[v1].adjacent_vertices, u);
    }
    SortedErase(vertex_data[v1].adjacent_vertices, v2);

    // Mark v2 as removed.
    vertex_data[v2].removed = true;
    vertex_data[v2].adjacent_faces.clear();
    vertex_data[v2].adjacent_vertices.clear();

    // Increment v1's timestamp for lazy PQ deletion.
    ++vertex_data[v1].timestamp;

    // Recompute candidates for all edges incident to v1.
    for (const size_t u : vertex_data[v1].adjacent_vertices) {
      if (vertex_data[u].removed) continue;
      pq.push(
          ComputeEdgeCollapse(vertex_data, v1, u, options.interpolate_colors));
    }

    // Log progress at 10% intervals.
    faces_removed = num_faces - current_faces;
    const int progress_percent =
        static_cast<int>(100.0 * faces_removed / faces_to_remove);
    if (progress_percent / 10 > last_progress_percent / 10) {
      last_progress_percent = progress_percent;
      LOG(INFO) << "  " << progress_percent << "% complete (" << current_faces
                << " faces remaining, error=" << candidate.cost << ")";
    }
  }

  // Compaction — build output mesh with only surviving geometry.
  LOG(INFO) << "Compacting mesh...";
  PlyMesh result;
  constexpr size_t kUnmapped = std::numeric_limits<size_t>::max();
  std::vector<size_t> old_to_new(num_vertices, kUnmapped);

  for (size_t fi = 0; fi < num_faces; ++fi) {
    if (face_removed[fi]) continue;
    const auto& f = face_indices[fi];
    for (int j = 0; j < 3; ++j) {
      if (old_to_new[f[j]] == kUnmapped) {
        old_to_new[f[j]] = result.vertices.size();
        const auto& vd = vertex_data[f[j]];
        const Eigen::Vector3f clamped_color =
            vd.color.array().round().max(0.0f).min(255.0f);
        result.vertices.emplace_back(static_cast<float>(vd.position.x()),
                                     static_cast<float>(vd.position.y()),
                                     static_cast<float>(vd.position.z()),
                                     static_cast<uint8_t>(clamped_color.x()),
                                     static_cast<uint8_t>(clamped_color.y()),
                                     static_cast<uint8_t>(clamped_color.z()));
      }
    }
    result.faces.emplace_back(
        old_to_new[f[0]], old_to_new[f[1]], old_to_new[f[2]]);
  }

  LOG(INFO) << "Mesh simplification complete: " << num_faces << " -> "
            << result.faces.size() << " faces, " << num_vertices << " -> "
            << result.vertices.size() << " vertices";

  return result;
}

}  // namespace mvs
}  // namespace colmap
