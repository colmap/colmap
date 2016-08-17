// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_MVS_PATCH_MATCH_H_
#define COLMAP_SRC_MVS_PATCH_MATCH_H_

#include <iostream>
#include <memory>
#include <vector>

#include "mvs/depth_map.h"
#include "mvs/image.h"
#include "mvs/normal_map.h"

// We must not include "util/math.h" to avoid any Eigen includes here,
// since Visual Studio cannot compile some of the Eigen/Boost expressions.
#ifndef DEG2RAD
#define DEG2RAD(deg) deg * 0.0174532925199432
#endif

namespace colmap {
namespace mvs {

class PatchMatchCuda;

// This is a wrapper class around the actual PatchMatchCuda implementation. This
// class is necessary to hide Cuda code from any boost or Eigen code, since
// NVCC/MSVC cannot compile complex C++ code.
class PatchMatch {
 public:
  // Maximum possible window radius for the photometric consistency cost. This
  // value is equal to THREADS_PER_BLOCK in patch_match_cuda.cu and the limit
  // arises from the shared memory implementation.
  const static size_t kMaxWindowRadius = 32;

  struct Options {
    // Depth range in which to randomly sample depth hypotheses.
    float depth_min = 0.0f;
    float depth_max = 1.0f;

    // Half window size to compute NCC photo-consistency cost.
    int window_radius = 5;

    // Parameters for bilaterally weighted NCC.
    float sigma_spatial = window_radius;
    float sigma_color = 0.2f;

    // Number of random samples to draw in Monte Carlo sampling.
    int num_samples = 15;

    // Spread of the NCC likelihood function.
    float ncc_sigma = 0.6f;

    // Minimum triangulation angle in radians.
    float min_triangulation_angle = DEG2RAD(0.5f);

    // Spread of the incident angle likelihood function.
    float incident_angle_sigma = 0.9f;

    // Number of coordinate descent iterations. Each iteration consists
    // of four sweeps from left to right, top to bottom, and vice versa.
    int num_iterations = 5;

    // Whether to add a regularized geometric consistency term to the cost
    // function. If true, the `depth_maps` and `normal_maps` must not be null.
    bool geom_consistency = false;

    // The relative weight of the geometric consistency term w.r.t. to
    // the photo-consistency term.
    float geom_consistency_regularizer = 0.3f;

    // Maximum geometric consistency cost in terms of the forward-backward
    // reprojection error in pixels.
    float geom_consistency_max_cost = 3.0f;

    // Whether to enable filtering.
    bool filter = true;

    // Minimum selection probability for pixel to be photo-consistent.
    float filter_min_ncc = 0.1f;

    // Minimum selection probability for pixel to be photo-consistent.
    float filter_min_triangulation_angle = DEG2RAD(3.0f);

    // Minimum number of source images have to be consistent
    // for pixel not to be filtered.
    float filter_min_num_consistent = 2;

    // Maximum forward-backward reprojection error for pixel
    // to be geometrically consistent.
    float filter_geom_consistency_max_cost = 1.0f;

    // Print the options to stdout.
    void Print() const;
  };

  struct Problem {
    // Index of the reference image.
    int ref_image_id = -1;

    // Indices of the source images.
    std::vector<int> src_image_ids;

    // Input images for the photometric consistency term.
    std::vector<Image>* images = nullptr;

    // Input depth maps for the geometric consistency term.
    std::vector<DepthMap>* depth_maps = nullptr;

    // Input normal maps for the geometric consistency term.
    std::vector<NormalMap>* normal_maps = nullptr;

    // Print the configuration to stdout.
    void Print() const;
  };

  PatchMatch(const Options& options, const Problem& problem);
  ~PatchMatch();

  // Check the options and the problem for validity.
  void Check() const;

  // Run the patch match algorithm.
  void Run();

  // Get the computed values after running the algorithm.
  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  Mat<float> GetSelProbMap() const;

  // Get a list of geometrically consistent images, in the following format:
  //
  //    r_1, c_1, N_1, i_11, i_12, ..., i_1N_1,
  //    r_2, c_2, N_2, i_21, i_22, ..., i_2N_2, ...
  //
  // where r, c are the row and column image coordinates of the pixel,
  // N is the number of consistent images, followed by the N image identifiers.
  // Note that only pixels are listed which are not filtered.
  std::vector<int> GetConsistentImageIds() const;

 private:
  const Options options_;
  const Problem problem_;
  std::unique_ptr<PatchMatchCuda> patch_match_cuda_;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_PATCH_MATCH_H_
