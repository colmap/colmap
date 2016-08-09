// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "util/cache"
#include <boost/test/unit_test.hpp>

#include "util/cache.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  LRUCache<int, int> cache(5, [](const int key) { return key; });
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
}

BOOST_AUTO_TEST_CASE(TestGet) {
  LRUCache<int, int> cache(5, [](const int key) { return key; });
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
  for (int i = 0; i < 5; ++i) {
    BOOST_CHECK_EQUAL(cache.Get(i), i);
    BOOST_CHECK_EQUAL(cache.NumElems(), i + 1);
    BOOST_CHECK(cache.Exists(i));
  }

  BOOST_CHECK_EQUAL(cache.Get(5), 5);
  BOOST_CHECK_EQUAL(cache.NumElems(), 5);
  BOOST_CHECK(!cache.Exists(0));
  BOOST_CHECK(cache.Exists(5));

  BOOST_CHECK_EQUAL(cache.Get(5), 5);
  BOOST_CHECK_EQUAL(cache.NumElems(), 5);
  BOOST_CHECK(!cache.Exists(0));
  BOOST_CHECK(cache.Exists(5));

  BOOST_CHECK_EQUAL(cache.Get(6), 6);
  BOOST_CHECK_EQUAL(cache.NumElems(), 5);
  BOOST_CHECK(!cache.Exists(0));
  BOOST_CHECK(!cache.Exists(1));
  BOOST_CHECK(cache.Exists(6));
}

BOOST_AUTO_TEST_CASE(TestSet) {
  LRUCache<int, int> cache(5, [](const int key) { return -1; });
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
  for (int i = 0; i < 5; ++i) {
    cache.Set(i, i);
    BOOST_CHECK_EQUAL(cache.NumElems(), i + 1);
    BOOST_CHECK(cache.Exists(i));
  }

  BOOST_CHECK_EQUAL(cache.Get(5), -1);
  BOOST_CHECK_EQUAL(cache.NumElems(), 5);
  BOOST_CHECK(!cache.Exists(0));
  BOOST_CHECK(cache.Exists(5));

  BOOST_CHECK_EQUAL(cache.Get(6), -1);
  BOOST_CHECK_EQUAL(cache.NumElems(), 5);
  BOOST_CHECK(!cache.Exists(0));
  BOOST_CHECK(!cache.Exists(1));
  BOOST_CHECK(cache.Exists(6));
}
