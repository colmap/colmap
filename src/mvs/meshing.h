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

#ifndef COLMAP_SRC_MVS_MESHING_H_
#define COLMAP_SRC_MVS_MESHING_H_

#include <string>

namespace colmap {
namespace mvs {

struct PoissonMeshingOptions {
  // This floating point value specifies the importance that interpolation of
  // the point samples is given in the formulation of the screened Poisson
  // equation. The results of the original (unscreened) Poisson Reconstruction
  // can be obtained by setting this value to 0.
  double point_weight = 1.0;

  // This integer is the maximum depth of the tree that will be used for surface
  // reconstruction. Running at depth d corresponds to solving on a voxel grid
  // whose resolution is no larger than 2^d x 2^d x 2^d. Note that since the
  // reconstructor adapts the octree to the sampling density, the specified
  // reconstruction depth is only an upper bound.
  int depth = 13;

  // If specified, the reconstruction code assumes that the input is equipped
  // with colors and will extrapolate the color values to the vertices of the
  // reconstructed mesh. The floating point value specifies the relative
  // importance of finer color estimates over lower ones.
  double color = 32.0;

  // This floating point values specifies the value for mesh trimming. The
  // subset of the mesh with signal value less than the trim value is discarded.
  double trim = 10.0;

  // The number of threads used for the Poisson reconstruction.
  int num_threads = -1;

  bool Check() const;
};

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

// Perform Poisson surface reconstruction and return true if successful.
bool PoissonMeshing(const PoissonMeshingOptions& options,
                    const std::string& input_path,
                    const std::string& output_path);

#ifdef CGAL_ENABLED

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
                           const std::string& input_path,
                           const std::string& output_path);
void DenseDelaunayMeshing(const DelaunayMeshingOptions& options,
                          const std::string& input_path,
                          const std::string& output_path);

#endif  // CGAL_ENABLED

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_MESHING_H_
