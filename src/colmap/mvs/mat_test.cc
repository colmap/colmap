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

#define TEST_NAME "mvs/mat_test"
#include "colmap/mvs/mat.h"

#include "colmap/util/testing.h"

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
