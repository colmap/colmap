// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/visibility_pyramid.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(VisibilityPyramid, Default) {
  VisibilityPyramid pyramid;
  EXPECT_EQ(pyramid.NumLevels(), 0);
  EXPECT_EQ(pyramid.Width(), 0);
  EXPECT_EQ(pyramid.Height(), 0);
  EXPECT_EQ(pyramid.Score(), 0);
}

TEST(VisibilityPyramid, Score) {
  for (int num_levels = 1; num_levels < 8; ++num_levels) {
    Eigen::VectorXi scores(num_levels);
    size_t max_score = 0;
    for (int i = 1; i <= num_levels; ++i) {
      scores(i - 1) = (1 << i) * (1 << i);
      max_score += scores(i - 1) * scores(i - 1);
    }

    VisibilityPyramid pyramid(static_cast<size_t>(num_levels), 4, 4);
    EXPECT_EQ(pyramid.NumLevels(), num_levels);
    EXPECT_EQ(pyramid.Width(), 4);
    EXPECT_EQ(pyramid.Height(), 4);
    EXPECT_EQ(pyramid.Score(), 0);
    EXPECT_EQ(pyramid.MaxScore(), max_score);

    EXPECT_EQ(pyramid.Score(), 0);
    pyramid.SetPoint(0, 0);
    EXPECT_EQ(pyramid.Score(), scores.sum());
    pyramid.SetPoint(0, 0);
    EXPECT_EQ(pyramid.Score(), scores.sum());
    pyramid.SetPoint(0, 1);
    EXPECT_EQ(pyramid.Score(),
              scores.sum() + scores.tail(scores.size() - 1).sum());
    pyramid.SetPoint(0, 1);
    pyramid.SetPoint(0, 1);
    pyramid.SetPoint(1, 0);
    EXPECT_EQ(pyramid.Score(),
              scores.sum() + 2 * scores.tail(scores.size() - 1).sum());
    pyramid.SetPoint(1, 0);
    pyramid.SetPoint(1, 1);
    EXPECT_EQ(pyramid.Score(),
              scores.sum() + 3 * scores.tail(scores.size() - 1).sum());
    pyramid.ResetPoint(0, 0);
    EXPECT_EQ(pyramid.Score(),
              scores.sum() + 3 * scores.tail(scores.size() - 1).sum());
    pyramid.ResetPoint(0, 0);
    EXPECT_EQ(pyramid.Score(),
              scores.sum() + 2 * scores.tail(scores.size() - 1).sum());
    pyramid.SetPoint(0, 2);
    EXPECT_EQ(pyramid.Score(),
              2 * scores.sum() + 2 * scores.tail(scores.size() - 1).sum());
  }
}

}  // namespace
}  // namespace colmap
