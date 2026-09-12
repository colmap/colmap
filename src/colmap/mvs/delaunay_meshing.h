// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <filesystem>

namespace colmap {
namespace mvs {

struct DelaunayMeshingOptions {
  // Unify input points into one cell in the Delaunay triangulation that fall
  // within a reprojected radius of the given pixels.
  double max_proj_dist = 20.0;

  // Maximum relative depth difference between input point and a vertex of an
  // existing cell in the Delaunay triangulation, otherwise a new vertex is
  // created in the triangulation.
  double max_depth_dist = 0.05;

  // The standard deviation of wrt. the number of images seen by each point.
  // Increasing this value decreases the influence of points seen in few images.
  double visibility_sigma = 3.0;

  // The factor that is applied to the computed distance sigma, which is
  // automatically computed as the 25th percentile of edge lengths. A higher
  // value will increase the smoothness of the surface.
  double distance_sigma_factor = 1.0;

  // A higher quality regularization leads to a smoother surface.
  double quality_regularization = 1.0;

  // Filtering thresholds for outlier surface mesh faces. If the longest side of
  // a mesh face (longest out of 3) exceeds the side lengths of all faces at a
  // certain percentile by the given factor, then it is considered an outlier
  // mesh face and discarded.
  double max_side_length_factor = 25.0;
  double max_side_length_percentile = 95.0;

  // The number of threads to use for reconstruction. Default is all threads.
  int num_threads = -1;

  bool Check() const;
};

#if defined(COLMAP_CGAL_ENABLED)

// Delaunay meshing of sparse and dense COLMAP reconstructions. This is an
// implementation of the approach described in:
//
//    P. Labatut, J‐P. Pons, and R. Keriven. "Robust and efficient surface
//    reconstruction from range data". Computer graphics forum, 2009.
//
// In case of sparse input, the path should point to a sparse COLMAP
// reconstruction. In case of dense input, the path should point to a dense
// COLMAP workspace folder, which has been fully processed by the stereo and
// fusion pipeline.
void SparseDelaunayMeshing(const DelaunayMeshingOptions& options,
                           const std::filesystem::path& input_path,
                           const std::filesystem::path& output_path);
void DenseDelaunayMeshing(const DelaunayMeshingOptions& options,
                          const std::filesystem::path& input_path,
                          const std::filesystem::path& output_path);

#endif  // COLMAP_CGAL_ENABLED

}  // namespace mvs
}  // namespace colmap
