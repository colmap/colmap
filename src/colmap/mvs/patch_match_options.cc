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

#include "colmap/mvs/patch_match_options.h"

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

namespace colmap {
namespace mvs {

// Maximum possible window radius for the photometric consistency cost. This
// value is equal to THREADS_PER_BLOCK in patch_match_cuda.cu and the limit
// arises from the shared memory implementation.
const static size_t kMaxPatchMatchWindowRadius = 32;

#define PrintOption(option) LOG(INFO) << #option ": " << option

void PatchMatchOptions::Print() const {
  PrintHeading2("PatchMatchOptions");
  PrintOption(max_image_size);
  PrintOption(gpu_index);
  PrintOption(depth_min);
  PrintOption(depth_max);
  PrintOption(window_radius);
  PrintOption(window_step);
  PrintOption(sigma_spatial);
  PrintOption(sigma_color);
  PrintOption(num_samples);
  PrintOption(ncc_sigma);
  PrintOption(min_triangulation_angle);
  PrintOption(incident_angle_sigma);
  PrintOption(num_iterations);
  PrintOption(geom_consistency);
  PrintOption(geom_consistency_regularizer);
  PrintOption(geom_consistency_max_cost);
  PrintOption(filter);
  PrintOption(filter_min_ncc);
  PrintOption(filter_min_triangulation_angle);
  PrintOption(filter_min_num_consistent);
  PrintOption(filter_geom_consistency_max_cost);
  PrintOption(write_consistency_graph);
  PrintOption(allow_missing_files);
}

bool PatchMatchOptions::Check() const {
  if (depth_min != -1.0f || depth_max != -1.0f) {
    CHECK_OPTION_LE(depth_min, depth_max);
    CHECK_OPTION_GE(depth_min, 0.0f);
  }
  CHECK_OPTION_LE(window_radius, static_cast<int>(kMaxPatchMatchWindowRadius));
  CHECK_OPTION_GT(sigma_color, 0.0f);
  CHECK_OPTION_GT(window_radius, 0);
  CHECK_OPTION_GT(window_step, 0);
  CHECK_OPTION_LE(window_step, 2);
  CHECK_OPTION_GT(num_samples, 0);
  CHECK_OPTION_GT(ncc_sigma, 0.0f);
  CHECK_OPTION_GE(min_triangulation_angle, 0.0f);
  CHECK_OPTION_LT(min_triangulation_angle, 180.0f);
  CHECK_OPTION_GT(incident_angle_sigma, 0.0f);
  CHECK_OPTION_GT(num_iterations, 0);
  CHECK_OPTION_GE(geom_consistency_regularizer, 0.0f);
  CHECK_OPTION_GE(geom_consistency_max_cost, 0.0f);
  CHECK_OPTION_GE(filter_min_ncc, -1.0f);
  CHECK_OPTION_LE(filter_min_ncc, 1.0f);
  CHECK_OPTION_GE(filter_min_triangulation_angle, 0.0f);
  CHECK_OPTION_LE(filter_min_triangulation_angle, 180.0f);
  CHECK_OPTION_GE(filter_min_num_consistent, 0);
  CHECK_OPTION_GE(filter_geom_consistency_max_cost, 0.0f);
  CHECK_OPTION_GT(cache_size, 0);
  return true;
}

}  // namespace mvs
}  // namespace colmap
