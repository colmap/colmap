// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_options.h"

#include <initializer_list>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(PatchMatchOptions, WindowMatchesCompiledKernels) {
  PatchMatchOptions options;
  for (const int step : {1, 2}) {
    options.window_step = step;
    for (const int radius : {1, 20}) {
      options.window_radius = radius;
      EXPECT_TRUE(options.Check());
    }
    for (const int radius : {0, 21, 32}) {
      options.window_radius = radius;
      EXPECT_FALSE(options.Check());
    }
  }
  options.window_radius = 5;
  for (const int step : {0, 3}) {
    options.window_step = step;
    EXPECT_FALSE(options.Check());
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
