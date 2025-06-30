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

#pragma once

#include <string>

namespace colmap {
namespace mvs {

struct PatchMatchOptions {
  // Maximum image size in either dimension.
  int max_image_size = -1;

  // Index of the GPU used for patch match. For multi-GPU usage,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // Depth range in which to randomly sample depth hypotheses.
  double depth_min = -1.0f;
  double depth_max = -1.0f;

  // Half window size to compute NCC photo-consistency cost.
  int window_radius = 5;

  // Number of pixels to skip when computing NCC. For a value of 1, every
  // pixel is used to compute the NCC. For larger values, only every n-th row
  // and column is used and the computation speed thereby increases roughly by
  // a factor of window_step^2. Note that not all combinations of window sizes
  // and steps produce nice results, especially if the step is greather than 2.
  int window_step = 1;

  // Parameters for bilaterally weighted NCC.
  double sigma_spatial = -1;
  double sigma_color = 0.2f;

  // Number of random samples to draw in Monte Carlo sampling.
  int num_samples = 15;

  // Spread of the NCC likelihood function.
  double ncc_sigma = 0.6f;

  // Minimum triangulation angle in degrees.
  double min_triangulation_angle = 1.0f;

  // Spread of the incident angle likelihood function.
  double incident_angle_sigma = 0.9f;

  // Number of coordinate descent iterations. Each iteration consists
  // of four sweeps from left to right, top to bottom, and vice versa.
  int num_iterations = 5;

  // Whether to add a regularized geometric consistency term to the cost
  // function. If true, the `depth_maps` and `normal_maps` must not be null.
  bool geom_consistency = true;

  // The relative weight of the geometric consistency term w.r.t. to
  // the photo-consistency term.
  double geom_consistency_regularizer = 0.3f;

  // Maximum geometric consistency cost in terms of the forward-backward
  // reprojection error in pixels.
  double geom_consistency_max_cost = 3.0f;

  // Whether to enable filtering.
  bool filter = true;

  // Minimum NCC coefficient for pixel to be photo-consistent.
  double filter_min_ncc = 0.1f;

  // Minimum triangulation angle to be stable.
  double filter_min_triangulation_angle = 3.0f;

  // Minimum number of source images have to be consistent
  // for pixel not to be filtered.
  int filter_min_num_consistent = 2;

  // Maximum forward-backward reprojection error for pixel
  // to be geometrically consistent.
  double filter_geom_consistency_max_cost = 1.0f;

  // Cache size in gigabytes for patch match, which keeps the bitmaps, depth
  // maps, and normal maps of this number of images in memory. A higher value
  // leads to less disk access and faster computation, while a lower value
  // leads to reduced memory usage. Note that a single image can consume a lot
  // of memory, if the consistency graph is dense.
  double cache_size = 32.0;

  // Whether to tolerate missing images/maps in the problem setup
  bool allow_missing_files = false;

  // Whether to write the consistency graph.
  bool write_consistency_graph = false;

  void Print() const;
  bool Check() const;
};

}  // namespace mvs
}  // namespace colmap
