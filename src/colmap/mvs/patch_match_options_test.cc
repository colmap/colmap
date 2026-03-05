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

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(PatchMatchOptions, DefaultsAreValid) {
  PatchMatchOptions options;
  EXPECT_TRUE(options.Check());
}

TEST(PatchMatchOptions, DefaultValues) {
  PatchMatchOptions options;
  EXPECT_EQ(options.depth_min, -1.0f);
  EXPECT_EQ(options.depth_max, -1.0f);
  EXPECT_EQ(options.sigma_color, 0.2f);
  EXPECT_EQ(options.ncc_sigma, 0.6f);
  EXPECT_EQ(options.min_triangulation_angle, 1.0f);
  EXPECT_EQ(options.incident_angle_sigma, 0.9f);
  EXPECT_EQ(options.geom_consistency_regularizer, 0.3f);
  EXPECT_EQ(options.geom_consistency_max_cost, 3.0f);
  EXPECT_EQ(options.filter_min_ncc, 0.1f);
  EXPECT_EQ(options.filter_min_triangulation_angle, 3.0f);
  EXPECT_EQ(options.filter_geom_consistency_max_cost, 1.0f);
  EXPECT_EQ(options.cache_size, 32.0);
  EXPECT_EQ(options.gpu_index, "-1");
  EXPECT_EQ(options.max_image_size, -1);
  EXPECT_EQ(options.window_radius, 5);
  EXPECT_EQ(options.window_step, 1);
  EXPECT_EQ(options.num_samples, 15);
  EXPECT_EQ(options.num_iterations, 5);
  EXPECT_EQ(options.filter_min_num_consistent, 2);
  EXPECT_EQ(options.num_threads, -1);
  EXPECT_TRUE(options.geom_consistency);
  EXPECT_TRUE(options.filter);
  EXPECT_FALSE(options.allow_missing_files);
  EXPECT_FALSE(options.write_consistency_graph);
}

TEST(PatchMatchOptions, ValidDepthRange) {
  PatchMatchOptions options;
  options.depth_min = 0.5f;
  options.depth_max = 10.0f;
  EXPECT_TRUE(options.Check());
}

TEST(PatchMatchOptions, DepthMinGreaterThanMax) {
  PatchMatchOptions options;
  options.depth_min = 10.0f;
  options.depth_max = 0.5f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, NegativeDepthMin) {
  PatchMatchOptions options;
  options.depth_min = -0.5f;
  options.depth_max = 10.0f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, WindowRadiusZero) {
  PatchMatchOptions options;
  options.window_radius = 0;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, WindowRadiusTooLarge) {
  PatchMatchOptions options;
  options.window_radius = 33;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, WindowRadiusAtMax) {
  PatchMatchOptions options;
  options.window_radius = 32;
  EXPECT_TRUE(options.Check());
}

TEST(PatchMatchOptions, WindowStepZero) {
  PatchMatchOptions options;
  options.window_step = 0;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, WindowStepTooLarge) {
  PatchMatchOptions options;
  options.window_step = 3;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, WindowStepBoundary) {
  PatchMatchOptions options;
  options.window_step = 2;
  EXPECT_TRUE(options.Check());
}

TEST(PatchMatchOptions, SigmaColorZero) {
  PatchMatchOptions options;
  options.sigma_color = 0.0f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, NumSamplesZero) {
  PatchMatchOptions options;
  options.num_samples = 0;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, NccSigmaZero) {
  PatchMatchOptions options;
  options.ncc_sigma = 0.0f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, MinTriangulationAngleNegative) {
  PatchMatchOptions options;
  options.min_triangulation_angle = -1.0f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, MinTriangulationAngleAt180) {
  PatchMatchOptions options;
  options.min_triangulation_angle = 180.0f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, MinTriangulationAngleBoundary) {
  PatchMatchOptions options;
  options.min_triangulation_angle = 0.0f;
  EXPECT_TRUE(options.Check());
}

TEST(PatchMatchOptions, IncidentAngleSigmaZero) {
  PatchMatchOptions options;
  options.incident_angle_sigma = 0.0f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, NumIterationsZero) {
  PatchMatchOptions options;
  options.num_iterations = 0;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, GeomConsistencyRegularizerNegative) {
  PatchMatchOptions options;
  options.geom_consistency_regularizer = -0.1f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, GeomConsistencyMaxCostNegative) {
  PatchMatchOptions options;
  options.geom_consistency_max_cost = -1.0f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, FilterMinNccTooLow) {
  PatchMatchOptions options;
  options.filter_min_ncc = -1.1f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, FilterMinNccTooHigh) {
  PatchMatchOptions options;
  options.filter_min_ncc = 1.1f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, FilterMinNccBoundaries) {
  PatchMatchOptions options;
  options.filter_min_ncc = -1.0f;
  EXPECT_TRUE(options.Check());
  options.filter_min_ncc = 1.0f;
  EXPECT_TRUE(options.Check());
}

TEST(PatchMatchOptions, FilterMinTriangulationAngleNegative) {
  PatchMatchOptions options;
  options.filter_min_triangulation_angle = -0.1f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, FilterMinTriangulationAngleTooHigh) {
  PatchMatchOptions options;
  options.filter_min_triangulation_angle = 180.1f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, FilterMinNumConsistentNegative) {
  PatchMatchOptions options;
  options.filter_min_num_consistent = -1;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, FilterGeomConsistencyMaxCostNegative) {
  PatchMatchOptions options;
  options.filter_geom_consistency_max_cost = -0.1f;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, CacheSizeZero) {
  PatchMatchOptions options;
  options.cache_size = 0.0;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, NumThreadsInvalid) {
  PatchMatchOptions options;
  options.num_threads = -2;
  EXPECT_FALSE(options.Check());
}

TEST(PatchMatchOptions, NumThreadsValid) {
  PatchMatchOptions options;
  options.num_threads = -1;
  EXPECT_TRUE(options.Check());
  options.num_threads = 1;
  EXPECT_TRUE(options.Check());
  options.num_threads = 8;
  EXPECT_TRUE(options.Check());
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
