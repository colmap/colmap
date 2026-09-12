// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <filesystem>

namespace colmap {
namespace mvs {

struct AdvancingFrontMeshingOptions {
  // Maximum edge length constraint for triangles (in world units).
  // Facets with any edge longer than this are discarded.
  // Set to 0 to disable the constraint.
  double max_edge_length = 1.0;

  // Whether to use visibility-based filtering.
  // Requires a dense workspace with fused.ply.vis and sparse/ directory.
  bool visibility_filtering = true;

  // Maximum number of visibility ray intersections before a face is
  // considered to violate free-space and is discarded. Applies to both
  // pre-filtering and post-filtering modes.
  int visibility_filtering_max_intersections = 10;

  // Controls how visibility filtering is applied:
  // - Post-filtering (true): AFSR runs without visibility constraints, then
  //   faces intersected by any visibility ray are removed via AABB tree
  //   queries. Faster, but leaves holes where faces are removed since AFSR
  //   cannot grow alternatives.
  // - Pre-filtering (false): Visibility rays are cast through the Delaunay
  //   triangulation before AFSR. Facets exceeding
  //   visibility_filtering_max_intersections ray intersections are rejected
  //   during surface growing, allowing AFSR to route around violations and
  //   produce a more complete surface. Slower due to Delaunay cell traversal
  //   for ray casting.
  bool visibility_post_filtering = true;

  // Absolute distance (in world units) by which visibility rays are shortened
  // at the target end to avoid self-intersection with the surface near the
  // observed point. Larger values trim more aggressively, reducing false
  // positive face removals but potentially missing actual violations.
  double visibility_ray_trim_offset = 0.1;

  // Block size for block-wise parallel processing (in world units).
  // Set to 0 to disable blocking and process the entire point cloud at once.
  double block_size = 0.0;

  // Overlap margin as a fraction of block_size for seam handling.
  // Only used when block_size > 0.
  double block_overlap = 0.2;

  // The number of threads to use. Default is all threads.
  int num_threads = -1;

  bool Check() const;
};

#if defined(COLMAP_CGAL_ENABLED)

// Advancing front surface reconstruction of dense COLMAP reconstructions.
// This is an implementation of the approach described in:
//
//    D. Cohen-Steiner and F. Da. "A greedy Delaunay-based surface
//    reconstruction algorithm." The Visual Computer, 2004.
//
// The input_path should point to a dense COLMAP workspace folder containing
// fused.ply (and optionally fused.ply.vis + sparse/ for visibility filtering).
void AdvancingFrontMeshing(const AdvancingFrontMeshingOptions& options,
                           const std::filesystem::path& input_path,
                           const std::filesystem::path& output_path);

#endif  // COLMAP_CGAL_ENABLED

}  // namespace mvs
}  // namespace colmap
