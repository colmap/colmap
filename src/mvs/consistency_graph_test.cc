// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#define TEST_NAME "mvs/consistency_graph_test"
#include "util/testing.h"

#include "mvs/consistency_graph.h"

using namespace colmap;
using namespace colmap::mvs;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  const std::vector<int> data;
  ConsistencyGraph consistency_graph(2, 2, data);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      int num_images;
      const int* image_idxs;
      consistency_graph.GetImageIdxs(0, 0, &num_images, &image_idxs);
      BOOST_CHECK_EQUAL(num_images, 0);
      BOOST_CHECK(image_idxs == nullptr);
    }
  }
  BOOST_CHECK_EQUAL(consistency_graph.GetNumBytes(), 16);
}

BOOST_AUTO_TEST_CASE(TestPartial) {
  const std::vector<int> data = {0, 0, 3, 5, 7, 33};
  ConsistencyGraph consistency_graph(2, 1, data);
  int num_images;
  const int* image_idxs;
  consistency_graph.GetImageIdxs(0, 0, &num_images, &image_idxs);
  BOOST_CHECK_EQUAL(num_images, 3);
  BOOST_CHECK_EQUAL(image_idxs[0], 5);
  BOOST_CHECK_EQUAL(image_idxs[1], 7);
  BOOST_CHECK_EQUAL(image_idxs[2], 33);
  consistency_graph.GetImageIdxs(0, 1, &num_images, &image_idxs);
  BOOST_CHECK_EQUAL(num_images, 0);
  BOOST_CHECK(image_idxs == nullptr);
  BOOST_CHECK_EQUAL(consistency_graph.GetNumBytes(), 32);
}

BOOST_AUTO_TEST_CASE(TestZero) {
  const std::vector<int> data = {0, 0, 0};
  ConsistencyGraph consistency_graph(2, 1, data);
  int num_images;
  const int* image_idxs;
  consistency_graph.GetImageIdxs(0, 0, &num_images, &image_idxs);
  BOOST_CHECK_EQUAL(num_images, 0);
  BOOST_CHECK(image_idxs == nullptr);
  consistency_graph.GetImageIdxs(0, 1, &num_images, &image_idxs);
  BOOST_CHECK_EQUAL(num_images, 0);
  BOOST_CHECK(image_idxs == nullptr);
  BOOST_CHECK_EQUAL(consistency_graph.GetNumBytes(), 20);
}

BOOST_AUTO_TEST_CASE(TestFull) {
  const std::vector<int> data = {0, 0, 3, 5, 7, 33, 0, 1, 1, 100};
  ConsistencyGraph consistency_graph(1, 2, data);
  int num_images;
  const int* image_idxs;
  consistency_graph.GetImageIdxs(0, 0, &num_images, &image_idxs);
  BOOST_CHECK_EQUAL(num_images, 3);
  BOOST_CHECK_EQUAL(image_idxs[0], 5);
  BOOST_CHECK_EQUAL(image_idxs[1], 7);
  BOOST_CHECK_EQUAL(image_idxs[2], 33);
  consistency_graph.GetImageIdxs(1, 0, &num_images, &image_idxs);
  BOOST_CHECK_EQUAL(num_images, 1);
  BOOST_CHECK_EQUAL(image_idxs[0], 100);
  BOOST_CHECK_EQUAL(consistency_graph.GetNumBytes(), 48);
}
