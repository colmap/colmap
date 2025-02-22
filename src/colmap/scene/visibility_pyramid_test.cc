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
