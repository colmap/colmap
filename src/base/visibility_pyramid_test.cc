// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#define TEST_NAME "base/visibility_pyramid"
#include "util/testing.h"

#include "base/visibility_pyramid.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestDefault) {
  VisibilityPyramid pyramid;
  BOOST_CHECK_EQUAL(pyramid.NumLevels(), 0);
  BOOST_CHECK_EQUAL(pyramid.Width(), 0);
  BOOST_CHECK_EQUAL(pyramid.Height(), 0);
  BOOST_CHECK_EQUAL(pyramid.Score(), 0);
}

BOOST_AUTO_TEST_CASE(TestScore) {
  for (int num_levels = 1; num_levels < 8; ++num_levels) {
    Eigen::VectorXi scores(num_levels);
    size_t max_score = 0;
    for (int i = 1; i <= num_levels; ++i) {
      scores(i - 1) = (1 << i) * (1 << i);
      max_score += scores(i - 1) * scores(i - 1);
    }

    VisibilityPyramid pyramid(static_cast<size_t>(num_levels), 4, 4);
    BOOST_CHECK_EQUAL(pyramid.NumLevels(), num_levels);
    BOOST_CHECK_EQUAL(pyramid.Width(), 4);
    BOOST_CHECK_EQUAL(pyramid.Height(), 4);
    BOOST_CHECK_EQUAL(pyramid.Score(), 0);
    BOOST_CHECK_EQUAL(pyramid.MaxScore(), max_score);

    BOOST_CHECK_EQUAL(pyramid.Score(), 0);
    pyramid.SetPoint(0, 0);
    BOOST_CHECK_EQUAL(pyramid.Score(), scores.sum());
    pyramid.SetPoint(0, 0);
    BOOST_CHECK_EQUAL(pyramid.Score(), scores.sum());
    pyramid.SetPoint(0, 1);
    BOOST_CHECK_EQUAL(pyramid.Score(),
                      scores.sum() + scores.tail(scores.size() - 1).sum());
    pyramid.SetPoint(0, 1);
    pyramid.SetPoint(0, 1);
    pyramid.SetPoint(1, 0);
    BOOST_CHECK_EQUAL(pyramid.Score(),
                      scores.sum() + 2 * scores.tail(scores.size() - 1).sum());
    pyramid.SetPoint(1, 0);
    pyramid.SetPoint(1, 1);
    BOOST_CHECK_EQUAL(pyramid.Score(),
                      scores.sum() + 3 * scores.tail(scores.size() - 1).sum());
    pyramid.ResetPoint(0, 0);
    BOOST_CHECK_EQUAL(pyramid.Score(),
                      scores.sum() + 3 * scores.tail(scores.size() - 1).sum());
    pyramid.ResetPoint(0, 0);
    BOOST_CHECK_EQUAL(pyramid.Score(),
                      scores.sum() + 2 * scores.tail(scores.size() - 1).sum());
    pyramid.SetPoint(0, 2);
    BOOST_CHECK_EQUAL(
        pyramid.Score(),
        2 * scores.sum() + 2 * scores.tail(scores.size() - 1).sum());
  }
}
