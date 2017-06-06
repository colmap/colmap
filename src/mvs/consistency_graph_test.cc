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
      const int* image_ids;
      consistency_graph.GetImageIds(0, 0, &num_images, &image_ids);
      BOOST_CHECK_EQUAL(num_images, 0);
      BOOST_CHECK(image_ids == nullptr);
    }
  }
  BOOST_CHECK_EQUAL(consistency_graph.GetNumBytes(), 16);
}

BOOST_AUTO_TEST_CASE(TestPartial) {
  const std::vector<int> data = {0, 0, 3, 5, 7, 33};
  ConsistencyGraph consistency_graph(2, 1, data);
  int num_images;
  const int* image_ids;
  consistency_graph.GetImageIds(0, 0, &num_images, &image_ids);
  BOOST_CHECK_EQUAL(num_images, 3);
  BOOST_CHECK_EQUAL(image_ids[0], 5);
  BOOST_CHECK_EQUAL(image_ids[1], 7);
  BOOST_CHECK_EQUAL(image_ids[2], 33);
  consistency_graph.GetImageIds(0, 1, &num_images, &image_ids);
  BOOST_CHECK_EQUAL(num_images, 0);
  BOOST_CHECK(image_ids == nullptr);
  BOOST_CHECK_EQUAL(consistency_graph.GetNumBytes(), 32);
}

BOOST_AUTO_TEST_CASE(TestZero) {
  const std::vector<int> data = {0, 0, 0};
  ConsistencyGraph consistency_graph(2, 1, data);
  int num_images;
  const int* image_ids;
  consistency_graph.GetImageIds(0, 0, &num_images, &image_ids);
  BOOST_CHECK_EQUAL(num_images, 0);
  BOOST_CHECK(image_ids == nullptr);
  consistency_graph.GetImageIds(0, 1, &num_images, &image_ids);
  BOOST_CHECK_EQUAL(num_images, 0);
  BOOST_CHECK(image_ids == nullptr);
  BOOST_CHECK_EQUAL(consistency_graph.GetNumBytes(), 20);
}

BOOST_AUTO_TEST_CASE(TestFull) {
  const std::vector<int> data = {0, 0, 3, 5, 7, 33, 0, 1, 1, 100};
  ConsistencyGraph consistency_graph(2, 1, data);
  int num_images;
  const int* image_ids;
  consistency_graph.GetImageIds(0, 0, &num_images, &image_ids);
  BOOST_CHECK_EQUAL(num_images, 3);
  BOOST_CHECK_EQUAL(image_ids[0], 5);
  BOOST_CHECK_EQUAL(image_ids[1], 7);
  BOOST_CHECK_EQUAL(image_ids[2], 33);
  consistency_graph.GetImageIds(0, 1, &num_images, &image_ids);
  BOOST_CHECK_EQUAL(num_images, 1);
  BOOST_CHECK_EQUAL(image_ids[0], 100);
  BOOST_CHECK_EQUAL(consistency_graph.GetNumBytes(), 48);
}
