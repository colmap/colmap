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

#define TEST_NAME "mvs/mat_test"
#include "util/testing.h"

#include "mvs/mat.h"

using namespace colmap::mvs;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  Mat<int> mat;
  BOOST_CHECK_EQUAL(mat.GetWidth(), 0);
  BOOST_CHECK_EQUAL(mat.GetHeight(), 0);
  BOOST_CHECK_EQUAL(mat.GetDepth(), 0);
  BOOST_CHECK_EQUAL(mat.GetNumBytes(), 0);
}

BOOST_AUTO_TEST_CASE(TestNonEmpty) {
  Mat<int> mat(1, 2, 3);
  BOOST_CHECK_EQUAL(mat.GetWidth(), 1);
  BOOST_CHECK_EQUAL(mat.GetHeight(), 2);
  BOOST_CHECK_EQUAL(mat.GetDepth(), 3);
  BOOST_CHECK_EQUAL(mat.GetNumBytes(), 24);
}

BOOST_AUTO_TEST_CASE(TestGetSet) {
  Mat<int> mat(1, 2, 3);

  BOOST_CHECK_EQUAL(mat.GetNumBytes(), 24);

  mat.Set(0, 0, 0, 1);
  mat.Set(0, 0, 1, 2);
  mat.Set(0, 0, 2, 3);
  mat.Set(1, 0, 0, 4);
  mat.Set(1, 0, 1, 5);
  mat.Set(1, 0, 2, 6);

  BOOST_CHECK_EQUAL(mat.Get(0, 0, 0), 1);
  BOOST_CHECK_EQUAL(mat.Get(0, 0, 1), 2);
  BOOST_CHECK_EQUAL(mat.Get(0, 0, 2), 3);
  BOOST_CHECK_EQUAL(mat.Get(1, 0, 0), 4);
  BOOST_CHECK_EQUAL(mat.Get(1, 0, 1), 5);
  BOOST_CHECK_EQUAL(mat.Get(1, 0, 2), 6);

  int slice[3];
  mat.GetSlice(0, 0, slice);
  BOOST_CHECK_EQUAL(slice[0], 1);
  BOOST_CHECK_EQUAL(slice[1], 2);
  BOOST_CHECK_EQUAL(slice[2], 3);
  mat.GetSlice(1, 0, slice);
  BOOST_CHECK_EQUAL(slice[0], 4);
  BOOST_CHECK_EQUAL(slice[1], 5);
  BOOST_CHECK_EQUAL(slice[2], 6);
}

BOOST_AUTO_TEST_CASE(TestFill) {
  Mat<int> mat(1, 2, 3);

  BOOST_CHECK_EQUAL(mat.GetNumBytes(), 24);

  mat.Fill(10);
  mat.Set(0, 0, 0, 10);
  mat.Set(0, 0, 1, 10);
  mat.Set(0, 0, 2, 10);
  mat.Set(1, 0, 0, 10);
  mat.Set(1, 0, 1, 10);
  mat.Set(1, 0, 2, 10);
}
