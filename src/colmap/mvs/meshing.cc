// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/mvs/meshing.h"

#include <fstream>
#include <unordered_map>
#include <vector>

#if defined(COLMAP_CGAL_ENABLED)
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#endif  // CGAL_ENABLED

#include "colmap/math/graph_cut.h"
#include "colmap/math/random.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/endian.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/ply.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"

#include "lib/PoissonRecon/PoissonRecon.h"
#include "lib/PoissonRecon/SurfaceTrimmer.h"

#if defined(COLMAP_CGAL_ENABLED)

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K, CGAL::Fast_location> Delaunay;

namespace std {

template <>
struct hash<Delaunay::Vertex_handle> {
  std::size_t operator()(const Delaunay::Vertex_handle& handle) const {
    return reinterpret_cast<std::size_t>(&*handle);
  }
};

template <>
struct hash<const Delaunay::Vertex_handle> {
  std::size_t operator()(const Delaunay::Vertex_handle& handle) const {
    return reinterpret_cast<std::size_t>(&*handle);
  }
};

template <>
struct hash<Delaunay::Cell_handle> {
  std::size_t operator()(const Delaunay::Cell_handle& handle) const {
    return reinterpret_cast<std::size_t>(&*handle);
  }
};

template <>
struct hash<const Delaunay::Cell_handle> {
  std::size_t operator()(const Delaunay::Cell_handle& handle) const {
    return reinterpret_cast<std::size_t>(&*handle);
  }
};

}  // namespace std

#endif  // CGAL_ENABLED

namespace colmap {
namespace mvs {

bool PoissonMeshingOptions::Check() const {
  CHECK_OPTION_GE(point_weight, 0);
  CHECK_OPTION_GT(depth, 0);
  CHECK_OPTION_GE(color, 0);
  CHECK_OPTION_GE(trim, 0);
  CHECK_OPTION_GE(num_threads, -1);
  CHECK_OPTION_NE(num_threads, 0);
  return true;
}

bool DelaunayMeshingOptions::Check() const {
  CHECK_OPTION_GE(max_proj_dist, 0);
  CHECK_OPTION_GE(max_depth_dist, 0);
  CHECK_OPTION_LE(max_depth_dist, 1);
  CHECK_OPTION_GT(visibility_sigma, 0);
  CHECK_OPTION_GT(distance_sigma_factor, 0);
  CHECK_OPTION_GE(quality_regularization, 0);
  CHECK_OPTION_GE(max_side_length_factor, 0);
  CHECK_OPTION_GE(max_side_length_percentile, 0);
  CHECK_OPTION_LE(max_side_length_percentile, 100);
  CHECK_OPTION_GE(num_threads, -1);
  CHECK_OPTION_NE(num_threads, 0);
  return true;
}

bool PoissonMeshing(const PoissonMeshingOptions& options,
                    const std::string& input_path,
                    const std::string& output_path) {
  CHECK(options.Check());

  std::vector<std::string> args;

  args.push_back("./binary");

  args.push_back("--in");
  args.push_back(input_path);

  args.push_back("--out");
  args.push_back(output_path);

  args.push_back("--pointWeight");
  args.push_back(std::to_string(options.point_weight));

  args.push_back("--depth");
  args.push_back(std::to_string(options.depth));

  if (options.color > 0) {
    args.push_back("--color");
    args.push_back(std::to_string(options.color));
  }

#if defined(COLMAP_OPENMP_ENABLED)
  if (options.num_threads > 0) {
    args.push_back("--threads");
    args.push_back(std::to_string(options.num_threads));
  }
#endif  // OPENMP_ENABLED

  if (options.trim > 0) {
    args.push_back("--density");
  }

  std::vector<const char*> args_cstr;
  args_cstr.reserve(args.size());
  for (const auto& arg : args) {
    args_cstr.push_back(arg.c_str());
  }

  if (PoissonRecon(args_cstr.size(), const_cast<char**>(args_cstr.data())) !=
      EXIT_SUCCESS) {
    return false;
  }

  if (options.trim == 0) {
    return true;
  }

  args.clear();
  args_cstr.clear();

  args.push_back("./binary");

  args.push_back("--in");
  args.push_back(output_path);

  args.push_back("--out");
  args.push_back(output_path);

  args.push_back("--trim");
  args.push_back(std::to_string(options.trim));

  args_cstr.reserve(args.size());
  for (const auto& arg : args) {
    args_cstr.push_back(arg.c_str());
  }

  return SurfaceTrimmer(args_cstr.size(),
                        const_cast<char**>(args_cstr.data())) == EXIT_SUCCESS;
}

#if defined(COLMAP_CGAL_ENABLED)

K::Point_3 EigenToCGAL(const Eigen::Vector3f& point) {
  return K::Point_3(point.x(), point.y(), point.z());
}

Eigen::Vector3f CGALToEigen(const K::Point_3& point) {
  return Eigen::Vector3f(point.x(), point.y(), point.z());
}

class DelaunayMeshingInput {
 public:
  struct Image {
    camera_t camera_id = kInvalidCameraId;
    Eigen::Matrix3x4f proj_matrix = Eigen::Matrix3x4f::Identity();
    Eigen::Vector3f proj_center = Eigen::Vector3f::Zero();
    std::vector<size_t> point_idxs;
  };

  struct Point {
    Eigen::Vector3f position = Eigen::Vector3f::Zero();
    uint32_t num_visible_images = 0;
  };

  std::unordered_map<camera_t, Camera> cameras;
  std::vector<Image> images;
  std::vector<Point> points;

  void ReadSparseReconstruction(const std::string& path) {
    Reconstruction reconstruction;
    reconstruction.Read(path);
    CopyFromSparseReconstruction(reconstruction);
  }

  void CopyFromSparseReconstruction(const Reconstruction& reconstruction) {
    images.reserve(reconstruction.NumRegImages());
    points.reserve(reconstruction.NumPoints3D());

    cameras = reconstruction.Cameras();

    std::unordered_map<point3D_t, size_t> point_id_to_idx;
    point_id_to_idx.reserve(reconstruction.NumPoints3D());
    for (const auto& point3D : reconstruction.Points3D()) {
      point_id_to_idx.emplace(point3D.first, points.size());
      DelaunayMeshingInput::Point input_point;
      input_point.position = point3D.second.XYZ().cast<float>();
      input_point.num_visible_images = point3D.second.Track().Length();
      points.push_back(input_point);
    }

    for (const auto image_id : reconstruction.RegImageIds()) {
      const auto& image = reconstruction.Image(image_id);
      DelaunayMeshingInput::Image input_image;
      input_image.camera_id = image.CameraId();
      input_image.proj_matrix = image.CamFromWorld().ToMatrix().cast<float>();
      input_image.proj_center = image.ProjectionCenter().cast<float>();
      input_image.point_idxs.reserve(image.NumPoints3D());
      for (const auto& point2D : image.Points2D()) {
        if (point2D.HasPoint3D()) {
          input_image.point_idxs.push_back(
              point_id_to_idx.at(point2D.point3D_id));
        }
      }
      images.push_back(input_image);
    }
  }

  void ReadDenseReconstruction(const std::string& path) {
    {
      Reconstruction reconstruction;
      reconstruction.Read(JoinPaths(path, "sparse"));

      cameras = reconstruction.Cameras();

      images.reserve(reconstruction.NumRegImages());
      for (const auto& image_id : reconstruction.RegImageIds()) {
        const auto& image = reconstruction.Image(image_id);
        DelaunayMeshingInput::Image input_image;
        input_image.camera_id = image.CameraId();
        input_image.proj_matrix = image.CamFromWorld().ToMatrix().cast<float>();
        input_image.proj_center = image.ProjectionCenter().cast<float>();
        images.push_back(input_image);
      }
    }

    const auto& ply_points = ReadPly(JoinPaths(path, "fused.ply"));

    const std::string vis_path = JoinPaths(path, "fused.ply.vis");
    std::fstream vis_file(vis_path, std::ios::in | std::ios::binary);
    CHECK(vis_file.is_open()) << vis_path;

    const size_t vis_num_points = ReadBinaryLittleEndian<uint64_t>(&vis_file);
    CHECK_EQ(vis_num_points, ply_points.size());

    points.reserve(ply_points.size());
    for (const auto& ply_point : ply_points) {
      const int point_idx = points.size();
      DelaunayMeshingInput::Point input_point;
      input_point.position =
          Eigen::Vector3f(ply_point.x, ply_point.y, ply_point.z);
      input_point.num_visible_images =
          ReadBinaryLittleEndian<uint32_t>(&vis_file);
      for (uint32_t i = 0; i < input_point.num_visible_images; ++i) {
        const int image_idx = ReadBinaryLittleEndian<uint32_t>(&vis_file);
        images.at(image_idx).point_idxs.push_back(point_idx);
      }
      points.push_back(input_point);
    }
  }

  Delaunay CreateDelaunayTriangulation() const {
    std::vector<Delaunay::Point> delaunay_points(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
      delaunay_points[i] = Delaunay::Point(points[i].position.x(),
                                           points[i].position.y(),
                                           points[i].position.z());
    }
    return Delaunay(delaunay_points.begin(), delaunay_points.end());
  }

  Delaunay CreateSubSampledDelaunayTriangulation(
      const float max_proj_dist, const float max_depth_dist) const {
    CHECK_GE(max_proj_dist, 0);

    if (max_proj_dist == 0) {
      return CreateDelaunayTriangulation();
    }

    std::vector<std::vector<uint32_t>> points_visible_image_idxs(points.size());
    for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
      for (const auto& point_idx : images[image_idx].point_idxs) {
        points_visible_image_idxs[point_idx].push_back(image_idx);
      }
    }

    std::vector<size_t> point_idxs(points.size());
    std::iota(point_idxs.begin(), point_idxs.end(), 0);
    Shuffle(point_idxs.size(), &point_idxs);

    Delaunay triangulation;

    const float max_squared_proj_dist = max_proj_dist * max_proj_dist;
    const float min_depth_ratio = 1.0f - max_depth_dist;
    const float max_depth_ratio = 1.0f + max_depth_dist;

    for (const auto point_idx : point_idxs) {
      const auto& point = points[point_idx];
      const auto& visible_image_idxs = points_visible_image_idxs[point_idx];

      const K::Point_3 point_position = EigenToCGAL(point.position);

      // Insert point into triangulation until there is one cell.
      if (triangulation.number_of_vertices() < 4) {
        triangulation.insert(point_position);
        continue;
      }

      const Delaunay::Cell_handle cell = triangulation.locate(point_position);

      // If the point is outside the current hull, then extend the hull.
      if (triangulation.is_infinite(cell)) {
        triangulation.insert(point_position);
        continue;
      }

      // Project point and located cell vertices to all visible images and
      // determine reprojection error.

      bool insert_point = false;

      for (const auto& image_idx : visible_image_idxs) {
        const auto& image = images[image_idx];
        const auto& camera = cameras.at(image.camera_id);

        for (int i = 0; i < 4; ++i) {
          const Eigen::Vector3f cell_point =
              CGALToEigen(cell->vertex(i)->point());

          const Eigen::Vector3f point_local =
              image.proj_matrix * point.position.homogeneous();
          const Eigen::Vector3f cell_point_local =
              image.proj_matrix * cell_point.homogeneous();

          // Ensure that both points are infront of camera.
          if (point_local.z() <= 0 || cell_point_local.z() <= 0) {
            insert_point = true;
            break;
          }

          // Check depth ratio between the two points.
          const float depth_ratio = point_local.z() / cell_point_local.z();
          if (depth_ratio < min_depth_ratio || depth_ratio > max_depth_ratio) {
            insert_point = true;
            break;
          }

          // Check reprojection error between the two points.
          const Eigen::Vector2f point_proj =
              camera.ImgFromCam(point_local.hnormalized().cast<double>())
                  .cast<float>();
          const Eigen::Vector2f cell_point_proj =
              camera.ImgFromCam(cell_point_local.hnormalized().cast<double>())
                  .cast<float>();
          const float squared_proj_dist =
              (point_proj - cell_point_proj).squaredNorm();
          if (squared_proj_dist > max_squared_proj_dist) {
            insert_point = true;
            break;
          }
        }

        if (insert_point) {
          break;
        }
      }

      if (insert_point) {
        triangulation.insert(point_position);
      }
    }

    std::cout << StringPrintf("Triangulation has %d using %d points.",
                              triangulation.number_of_vertices(),
                              points.size())
              << std::endl;

    return triangulation;
  }
};

struct DelaunayMeshingEdgeWeightComputer {
  DelaunayMeshingEdgeWeightComputer(const Delaunay& triangulation,
                                    const double visibility_sigma,
                                    const double distance_sigma_factor)
      : visibility_threshold_(5 * visibility_sigma),
        visibility_normalization_(-0.5 /
                                  (visibility_sigma * visibility_sigma)) {
    std::vector<float> edge_lengths;
    edge_lengths.reserve(triangulation.number_of_finite_edges());

    for (auto it = triangulation.finite_edges_begin();
         it != triangulation.finite_edges_end();
         ++it) {
      edge_lengths.push_back((it->first->vertex(it->second)->point() -
                              it->first->vertex(it->third)->point())
                                 .squared_length());
    }

    distance_sigma_ = distance_sigma_factor *
                      std::max(std::sqrt(Percentile(edge_lengths, 25)), 1e-7f);
    distance_threshold_ = 5 * distance_sigma_;
    distance_normalization_ = -0.5 / (distance_sigma_ * distance_sigma_);
  }

  double DistanceSigma() const { return distance_sigma_; }

  double ComputeVisibilityProb(const double visibility_squared) const {
    if (visibility_squared < visibility_threshold_) {
      return std::max(
          0.0, 1.0 - std::exp(visibility_squared * visibility_normalization_));
    } else {
      return 1.0;
    }
  }

  double ComputeDistanceProb(const double distance_squared) const {
    if (distance_squared < distance_threshold_) {
      return std::max(
          0.0, 1.0 - std::exp(distance_squared * distance_normalization_));
    } else {
      return 1.0;
    }
  }

 private:
  double visibility_threshold_;
  double visibility_normalization_;
  double distance_sigma_;
  double distance_threshold_;
  double distance_normalization_;
};

// Ray caster through the cells of a Delaunay triangulation. The tracing locates
// the cell of the ray origin and then iteratively intersects the ray with all
// facets of the current cell and advances to the neighboring cell of the
// intersected facet. Note that the ray can also pass through outside of the
// hull of the triangulation, i.e. lie within the infinite cells/facets.
// The ray caster collects the intersected facets along the ray.
struct DelaunayTriangulationRayCaster {
  struct Intersection {
    Delaunay::Facet facet;
    double target_distance_squared = 0.0;
  };

  explicit DelaunayTriangulationRayCaster(const Delaunay& triangulation)
      : triangulation_(triangulation) {
    FindHullFacets();
  }

  void CastRaySegment(const K::Segment_3& ray_segment,
                      std::vector<Intersection>* intersections) const {
    intersections->clear();

    Delaunay::Cell_handle next_cell =
        triangulation_.locate(ray_segment.start());

    bool next_cell_found = true;
    while (next_cell_found) {
      next_cell_found = false;

      if (triangulation_.is_infinite(next_cell)) {
        // Linearly check all hull facets for intersection.

        for (const auto& hull_facet : hull_facets_) {
          // Check if the ray origin is infront of the facet.
          const K::Triangle_3 triangle = triangulation_.triangle(hull_facet);
          if (CGAL::orientation(
                  triangle[0], triangle[1], triangle[2], ray_segment.start()) ==
              K::Orientation::NEGATIVE) {
            continue;
          }

          // Check if the segment intersects the facet.
          K::Point_3 intersection_point;
          if (!CGAL::assign(intersection_point,
                            CGAL::intersection(ray_segment, triangle))) {
            continue;
          }

          // Make sure the next intersection is closer to target than previous.
          const double target_distance_squared =
              (intersection_point - ray_segment.end()).squared_length();
          if (!intersections->empty() &&
              intersections->back().target_distance_squared <
                  target_distance_squared) {
            continue;
          }

          Intersection intersection;
          intersection.facet =
              Delaunay::Facet(hull_facet.first, hull_facet.second);
          intersection.target_distance_squared = target_distance_squared;
          intersections->push_back(intersection);

          next_cell = hull_facet.first->neighbor(hull_facet.second);
          next_cell_found = true;

          break;
        }
      } else {
        // Check all neighboring finite facets for intersection.

        for (int i = 0; i < 4; ++i) {
          // Check if the ray origin is infront of the facet.
          const K::Triangle_3 triangle = triangulation_.triangle(next_cell, i);
          if (CGAL::orientation(
                  triangle[0], triangle[1], triangle[2], ray_segment.start()) ==
              K::Orientation::NEGATIVE) {
            continue;
          }

          // Check if the segment intersects the facet.
          K::Point_3 intersection_point;
          if (!CGAL::assign(intersection_point,
                            CGAL::intersection(ray_segment, triangle))) {
            continue;
          }

          // Make sure the next intersection is closer to target than previous.
          const double target_distance_squared =
              (intersection_point - ray_segment.end()).squared_length();
          if (!intersections->empty() &&
              intersections->back().target_distance_squared <
                  target_distance_squared) {
            continue;
          }

          Intersection intersection;
          intersection.facet = Delaunay::Facet(next_cell, i);
          intersection.target_distance_squared = target_distance_squared;
          intersections->push_back(intersection);

          next_cell = next_cell->neighbor(i);
          next_cell_found = true;

          break;
        }
      }
    }
  }

 private:
  // Find all finite facets of infinite cells.
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

  const Delaunay& triangulation_;
  std::vector<Delaunay::Facet> hull_facets_;
};

// Implementation of geometry visualized in Figure 9 in P. Labatut, J‐P. Pons,
// and R. Keriven. "Robust and efficient surface reconstruction from range
// data." Computer graphics forum, 2009.
double ComputeCosFacetCellAngle(const Delaunay& triangulation,
                                const Delaunay::Facet& facet) {
  if (triangulation.is_infinite(facet.first)) {
    return 1.0;
  }

  const K::Triangle_3 triangle = triangulation.triangle(facet);

  const K::Vector_3 facet_normal =
      CGAL::cross_product(triangle[1] - triangle[0], triangle[2] - triangle[0]);
  const double facet_normal_length_squared = facet_normal.squared_length();
  if (facet_normal_length_squared == 0.0) {
    return 0.5;
  }

  const K::Vector_3 co_tangent = facet.first->circumcenter() - triangle[0];
  const float co_tangent_length_squared = co_tangent.squared_length();
  if (co_tangent_length_squared == 0.0) {
    return 0.5;
  }

  return (facet_normal * co_tangent) /
         std::sqrt(facet_normal_length_squared * co_tangent_length_squared);
}

void WriteDelaunayTriangulationPly(const std::string& path,
                                   const Delaunay& triangulation) {
  std::fstream file(path, std::ios::out);
  CHECK(file.is_open());

  file << "ply" << std::endl;
  file << "format ascii 1.0" << std::endl;
  file << "element vertex " << triangulation.number_of_vertices() << std::endl;
  file << "property float x" << std::endl;
  file << "property float y" << std::endl;
  file << "property float z" << std::endl;
  file << "element edge " << triangulation.number_of_finite_edges()
       << std::endl;
  file << "property int vertex1" << std::endl;
  file << "property int vertex2" << std::endl;
  file << "element face " << triangulation.number_of_finite_facets()
       << std::endl;
  file << "property list uchar int vertex_index" << std::endl;
  file << "end_header" << std::endl;

  std::unordered_map<const Delaunay::Vertex_handle, size_t> vertex_indices;
  vertex_indices.reserve(triangulation.number_of_vertices());
  for (auto it = triangulation.finite_vertices_begin();
       it != triangulation.finite_vertices_end();
       ++it) {
    vertex_indices.emplace(it, vertex_indices.size());
    file << it->point().x() << " " << it->point().y() << " " << it->point().z()
         << std::endl;
  }

  for (auto it = triangulation.finite_edges_begin();
       it != triangulation.finite_edges_end();
       ++it) {
    file << vertex_indices.at(it->first->vertex(it->second)) << " "
         << vertex_indices.at(it->first->vertex(it->third)) << std::endl;
  }

  for (auto it = triangulation.finite_facets_begin();
       it != triangulation.finite_facets_end();
       ++it) {
    file << "3 "
         << vertex_indices.at(it->first->vertex(
                triangulation.vertex_triple_index(it->second, 0)))
         << " "
         << vertex_indices.at(it->first->vertex(
                triangulation.vertex_triple_index(it->second, 1)))
         << " "
         << vertex_indices.at(it->first->vertex(
                triangulation.vertex_triple_index(it->second, 2)))
         << std::endl;
  }
}

struct DelaunayCellData {
  DelaunayCellData() : DelaunayCellData(-1) {}
  explicit DelaunayCellData(const int index)
      : index(index),
        source_weight(0),
        sink_weight(0),
        edge_weights({{0, 0, 0, 0}}) {}
  int index;
  float source_weight;
  float sink_weight;
  std::array<float, 4> edge_weights;
};

PlyMesh DelaunayMeshing(const DelaunayMeshingOptions& options,
                        const DelaunayMeshingInput& input_data) {
  CHECK(options.Check());

  // Create a delaunay triangulation of all input points.
  std::cout << "Triangulating points..." << std::endl;
  const auto triangulation = input_data.CreateSubSampledDelaunayTriangulation(
      options.max_proj_dist, options.max_depth_dist);

  // Helper class to efficiently trace rays through the triangulation.
  std::cout << "Initializing ray tracer..." << std::endl;
  const DelaunayTriangulationRayCaster ray_caster(triangulation);

  // Helper class to efficiently compute edge weights in the s-t graph.
  const DelaunayMeshingEdgeWeightComputer edge_weight_computer(
      triangulation, options.visibility_sigma, options.distance_sigma_factor);

  // Initialize the s-t graph with cells as nodes and oriented facets as edges.

  std::cout << "Initializing graph optimization..." << std::endl;

  typedef std::unordered_map<const Delaunay::Cell_handle, DelaunayCellData>
      CellGraphData;

  CellGraphData cell_graph_data;
  cell_graph_data.reserve(triangulation.number_of_cells());
  for (auto it = triangulation.all_cells_begin();
       it != triangulation.all_cells_end();
       ++it) {
    cell_graph_data.emplace(it, DelaunayCellData(cell_graph_data.size()));
  }

  // Spawn threads for parallelized integration of images.
  const int num_threads = GetEffectiveNumThreads(options.num_threads);
  ThreadPool thread_pool(num_threads);
  JobQueue<CellGraphData> result_queue(num_threads);

  // Function that accumulates edge weights in the s-t graph for a single image.
  auto IntegreateImage = [&](const size_t image_idx) {
    // Accumulated weights for the current image only.
    CellGraphData image_cell_graph_data;

    // Image that is integrated into s-t graph.
    const auto& image = input_data.images[image_idx];
    const K::Point_3 image_position = EigenToCGAL(image.proj_center);

    // Intersections between viewing rays and Delaunay triangulation.
    std::vector<DelaunayTriangulationRayCaster::Intersection> intersections;

    // Iterate through all image observations and integrate them into the graph.
    for (const auto& point_idx : image.point_idxs) {
      const auto& point = input_data.points[point_idx];

      // Likelihood of the point observation.
      const double alpha = edge_weight_computer.ComputeVisibilityProb(
          point.num_visible_images * point.num_visible_images);

      const K::Point_3 point_position = EigenToCGAL(point.position);
      const K::Ray_3 viewing_ray = K::Ray_3(image_position, point_position);
      const K::Vector_3 viewing_direction = point_position - image_position;
      const K::Vector_3 viewing_direction_normalized =
          viewing_direction / std::sqrt(viewing_direction.squared_length());
      const K::Vector_3 viewing_direction_epsilon =
          0.001 * edge_weight_computer.DistanceSigma() *
          viewing_direction_normalized;

      // Find intersected facets between image and point.
      ray_caster.CastRaySegment(
          K::Segment_3(image_position,
                       point_position - viewing_direction_epsilon),
          &intersections);

      // Accumulate source weights for cell containing image.
      if (!intersections.empty()) {
        image_cell_graph_data[intersections.front().facet.first]
            .source_weight += alpha;
      }

      // Accumulate edge weights from image to point.
      for (const auto& intersection : intersections) {
        image_cell_graph_data[intersection.facet.first]
            .edge_weights[intersection.facet.second] +=
            alpha * edge_weight_computer.ComputeDistanceProb(
                        intersection.target_distance_squared);
      }

      // Accumulate edge weights from point to extended point
      // and accumulate sink weight of the cell inside the surface.

      {
        // Find the first facet that is intersected by the extended ray behind
        // the observed point. Then accumulate the edge weight of that facet
        // and accumulate the sink weight of the cell behind that facet.

        const Delaunay::Cell_handle behind_point_cell =
            triangulation.locate(point_position + viewing_direction_epsilon);

        int behind_neighbor_idx = -1;
        double behind_distance_squared = 0.0;
        for (int neighbor_idx = 0; neighbor_idx < 4; ++neighbor_idx) {
          const K::Triangle_3 triangle =
              triangulation.triangle(behind_point_cell, neighbor_idx);

          K::Point_3 inter_point;
          if (CGAL::assign(inter_point,
                           CGAL::intersection(viewing_ray, triangle))) {
            const double distance_squared =
                (inter_point - point_position).squared_length();
            if (distance_squared > behind_distance_squared) {
              behind_distance_squared = distance_squared;
              behind_neighbor_idx = neighbor_idx;
            }
          }
        }

        if (behind_neighbor_idx >= 0) {
          image_cell_graph_data[behind_point_cell]
              .edge_weights[behind_neighbor_idx] +=
              alpha *
              edge_weight_computer.ComputeDistanceProb(behind_distance_squared);

          const auto& inside_cell =
              behind_point_cell->neighbor(behind_neighbor_idx);
          image_cell_graph_data[inside_cell].sink_weight += alpha;
        }
      }
    }

    CHECK(result_queue.Push(std::move(image_cell_graph_data)));
  };

  // Add first batch of images to the thread job queue.
  size_t image_idx = 0;
  const size_t init_num_tasks =
      std::min(input_data.images.size(), 2 * thread_pool.NumThreads());
  for (; image_idx < init_num_tasks; ++image_idx) {
    thread_pool.AddTask(IntegreateImage, image_idx);
  }

  // Pop the integrated images from the thread job queue and integrate their
  // accumulated weights into the global graph.
  for (size_t i = 0; i < input_data.images.size(); ++i) {
    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Integrating image [%d/%d]",
                              i + 1,
                              input_data.images.size())
              << std::flush;

    // Push the next image to the queue.
    if (image_idx < input_data.images.size()) {
      thread_pool.AddTask(IntegreateImage, image_idx);
      image_idx += 1;
    }

    // Pop the next results from the queue.
    auto result = result_queue.Pop();
    CHECK(result.IsValid());

    // Accumulate the weights of the image into the global graph.
    const auto& image_cell_graph_data = result.Data();
    for (const auto& image_cell_data : image_cell_graph_data) {
      auto& cell_data = cell_graph_data.at(image_cell_data.first);
      cell_data.sink_weight += image_cell_data.second.sink_weight;
      cell_data.source_weight += image_cell_data.second.source_weight;
      for (size_t j = 0; j < cell_data.edge_weights.size(); ++j) {
        cell_data.edge_weights[j] += image_cell_data.second.edge_weights[j];
      }
    }

    std::cout << StringPrintf(" in %.3fs", timer.ElapsedSeconds()) << std::endl;
  }

  // Setup the min-cut (max-flow) graph optimization.

  std::cout << "Setting up optimization..." << std::endl;

  // Each oriented facet in the Delaunay triangulation corresponds to a directed
  // edge and each cell corresponds to a node in the graph.
  MinSTGraphCut<size_t, float> graph_cut(cell_graph_data.size());

  // Iterate all cells in the triangulation.
  for (auto& cell_data : cell_graph_data) {
    graph_cut.AddNode(cell_data.second.index,
                      cell_data.second.source_weight,
                      cell_data.second.sink_weight);

    // Iterate all facets of the current cell to accumulate edge weight.
    for (int i = 0; i < 4; ++i) {
      // Compose the current facet.
      const Delaunay::Facet facet = std::make_pair(cell_data.first, i);

      // Extract the mirrored facet of the current cell (opposite orientation).
      const Delaunay::Facet mirror_facet = triangulation.mirror_facet(facet);
      const auto& mirror_cell_data = cell_graph_data.at(mirror_facet.first);

      // Avoid duplicate edges in graph.
      if (cell_data.second.index < mirror_cell_data.index) {
        continue;
      }

      // Implementation of geometry visualized in Figure 9 in P. Labatut, J‐P.
      // Pons, and R. Keriven. "Robust and efficient surface reconstruction from
      // range data." Computer graphics forum, 2009.
      const double edge_shape_weight =
          options.quality_regularization *
          (1.0 -
           std::min(ComputeCosFacetCellAngle(triangulation, facet),
                    ComputeCosFacetCellAngle(triangulation, mirror_facet)));

      const float forward_edge_weight =
          cell_data.second.edge_weights[facet.second] + edge_shape_weight;
      const float backward_edge_weight =
          mirror_cell_data.edge_weights[mirror_facet.second] +
          edge_shape_weight;

      graph_cut.AddEdge(cell_data.second.index,
                        mirror_cell_data.index,
                        forward_edge_weight,
                        backward_edge_weight);
    }
  }

  // Extract the surface facets as the oriented min-cut of the graph.

  std::cout << "Running graph-cut optimization..." << std::endl;
  graph_cut.Compute();

  std::cout << "Extracting surface as min-cut..." << std::endl;

  std::unordered_set<Delaunay::Vertex_handle> surface_vertices;
  std::vector<Delaunay::Facet> surface_facets;
  std::vector<float> surface_facet_side_lengths;

  for (auto it = triangulation.finite_facets_begin();
       it != triangulation.finite_facets_end();
       ++it) {
    const auto& cell_data = cell_graph_data.at(it->first);
    const auto& mirror_cell_data =
        cell_graph_data.at(it->first->neighbor(it->second));

    // Obtain labeling after the graph-cut.
    const bool cell_is_source = graph_cut.IsConnectedToSource(cell_data.index);
    const bool mirror_cell_is_source =
        graph_cut.IsConnectedToSource(mirror_cell_data.index);

    // The surface is equal to the location of the cut, which is at the
    // transition between source and sink nodes.
    if (cell_is_source == mirror_cell_is_source) {
      continue;
    }

    // Remember all unique vertices of the surface mesh.
    for (int i = 0; i < 3; ++i) {
      const auto& vertex =
          it->first->vertex(triangulation.vertex_triple_index(it->second, i));
      surface_vertices.insert(vertex);
    }

    // Determine maximum side length of facet.
    const K::Triangle_3 triangle = triangulation.triangle(*it);
    const float max_squared_side_length =
        std::max({(triangle[0] - triangle[1]).squared_length(),
                  (triangle[0] - triangle[2]).squared_length(),
                  (triangle[1] - triangle[2]).squared_length()});
    surface_facet_side_lengths.push_back(std::sqrt(max_squared_side_length));

    // Remember surface mesh facet and make sure it is oriented correctly.
    if (cell_is_source) {
      surface_facets.push_back(*it);
    } else {
      surface_facets.push_back(triangulation.mirror_facet(*it));
    }
  }

  std::cout << "Creating surface mesh model..." << std::endl;

  PlyMesh mesh;

  std::unordered_map<const Delaunay::Vertex_handle, size_t>
      surface_vertex_indices;
  surface_vertex_indices.reserve(surface_vertices.size());
  mesh.vertices.reserve(surface_vertices.size());
  for (const auto& vertex : surface_vertices) {
    mesh.vertices.emplace_back(
        vertex->point().x(), vertex->point().y(), vertex->point().z());
    surface_vertex_indices.emplace(vertex, surface_vertex_indices.size());
  }

  const float max_facet_side_length =
      options.max_side_length_factor *
      Percentile(surface_facet_side_lengths,
                 options.max_side_length_percentile);

  mesh.faces.reserve(surface_facets.size());

  for (size_t i = 0; i < surface_facets.size(); ++i) {
    // Note that skipping some of the facets here means that there will be
    // some unused vertices in the final mesh.
    if (surface_facet_side_lengths[i] > max_facet_side_length) {
      continue;
    }

    const auto& facet = surface_facets[i];
    mesh.faces.emplace_back(
        surface_vertex_indices.at(facet.first->vertex(
            triangulation.vertex_triple_index(facet.second, 0))),
        surface_vertex_indices.at(facet.first->vertex(
            triangulation.vertex_triple_index(facet.second, 1))),
        surface_vertex_indices.at(facet.first->vertex(
            triangulation.vertex_triple_index(facet.second, 2))));
  }

  return mesh;
}

void SparseDelaunayMeshing(const DelaunayMeshingOptions& options,
                           const std::string& input_path,
                           const std::string& output_path) {
  Timer timer;
  timer.Start();

  DelaunayMeshingInput input_data;
  input_data.ReadSparseReconstruction(input_path);

  const auto mesh = DelaunayMeshing(options, input_data);

  std::cout << "Writing surface mesh..." << std::endl;
  WriteBinaryPlyMesh(output_path, mesh);

  timer.PrintSeconds();
}

void DenseDelaunayMeshing(const DelaunayMeshingOptions& options,
                          const std::string& input_path,
                          const std::string& output_path) {
  Timer timer;
  timer.Start();

  DelaunayMeshingInput input_data;
  input_data.ReadDenseReconstruction(input_path);

  const auto mesh = DelaunayMeshing(options, input_data);

  std::cout << "Writing surface mesh..." << std::endl;
  WriteBinaryPlyMesh(output_path, mesh);

  timer.PrintSeconds();
}

#endif  // CGAL_ENABLED

}  // namespace mvs
}  // namespace colmap
