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

#define TEST_NAME "util/cache"
#include "util/testing.h"

#include "util/cache.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestLRUCacheEmpty) {
  LRUCache<int, int> cache(5, [](const int key) { return key; });
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
  BOOST_CHECK_EQUAL(cache.MaxNumElems(), 5);
}

BOOST_AUTO_TEST_CASE(TestLRUCacheGet) {
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

BOOST_AUTO_TEST_CASE(TestLRUCacheGetMutable) {
  LRUCache<int, int> cache(5, [](const int key) { return key; });
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
  for (int i = 0; i < 5; ++i) {
    BOOST_CHECK_EQUAL(cache.GetMutable(i), i);
    BOOST_CHECK_EQUAL(cache.NumElems(), i + 1);
    BOOST_CHECK(cache.Exists(i));
  }

  BOOST_CHECK_EQUAL(cache.GetMutable(5), 5);
  BOOST_CHECK_EQUAL(cache.NumElems(), 5);
  BOOST_CHECK(!cache.Exists(0));
  BOOST_CHECK(cache.Exists(5));

  BOOST_CHECK_EQUAL(cache.GetMutable(5), 5);
  BOOST_CHECK_EQUAL(cache.NumElems(), 5);
  BOOST_CHECK(!cache.Exists(0));
  BOOST_CHECK(cache.Exists(5));

  BOOST_CHECK_EQUAL(cache.GetMutable(6), 6);
  BOOST_CHECK_EQUAL(cache.NumElems(), 5);
  BOOST_CHECK(!cache.Exists(0));
  BOOST_CHECK(!cache.Exists(1));
  BOOST_CHECK(cache.Exists(6));

  cache.GetMutable(6) = 66;
  BOOST_CHECK_EQUAL(cache.GetMutable(6), 66);
  BOOST_CHECK_EQUAL(cache.NumElems(), 5);
  BOOST_CHECK(!cache.Exists(0));
  BOOST_CHECK(!cache.Exists(1));
  BOOST_CHECK(cache.Exists(6));
}

BOOST_AUTO_TEST_CASE(TestLRUCacheSet) {
  LRUCache<int, int> cache(5, [](const int key) { return -1; });
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
  for (int i = 0; i < 5; ++i) {
    cache.Set(i, std::move(i));
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

BOOST_AUTO_TEST_CASE(TestLRUCachePop) {
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

  cache.Pop();
  BOOST_CHECK_EQUAL(cache.NumElems(), 4);
  cache.Pop();
  BOOST_CHECK_EQUAL(cache.NumElems(), 3);
  cache.Pop();
  BOOST_CHECK_EQUAL(cache.NumElems(), 2);
  cache.Pop();
  BOOST_CHECK_EQUAL(cache.NumElems(), 1);
  cache.Pop();
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
  cache.Pop();
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
}

BOOST_AUTO_TEST_CASE(TestLRUCacheClear) {
  LRUCache<int, int> cache(5, [](const int key) { return key; });
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
  for (int i = 0; i < 5; ++i) {
    BOOST_CHECK_EQUAL(cache.Get(i), i);
    BOOST_CHECK_EQUAL(cache.NumElems(), i + 1);
    BOOST_CHECK(cache.Exists(i));
  }

  cache.Clear();
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);

  BOOST_CHECK_EQUAL(cache.Get(0), 0);
  BOOST_CHECK_EQUAL(cache.NumElems(), 1);
  BOOST_CHECK(cache.Exists(0));
}

struct SizedElem {
  SizedElem(const size_t num_bytes_) : num_bytes(num_bytes_) {}
  size_t NumBytes() const { return num_bytes; }
  size_t num_bytes;
};

BOOST_AUTO_TEST_CASE(TestMemoryConstrainedLRUCacheEmpty) {
  MemoryConstrainedLRUCache<int, SizedElem> cache(
      5, [](const int key) { return SizedElem(key); });
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
  BOOST_CHECK_EQUAL(cache.MaxNumElems(), std::numeric_limits<size_t>::max());
  BOOST_CHECK_EQUAL(cache.NumBytes(), 0);
  BOOST_CHECK_EQUAL(cache.MaxNumBytes(), 5);
}

BOOST_AUTO_TEST_CASE(TestMemoryConstrainedLRUCacheGet) {
  MemoryConstrainedLRUCache<int, SizedElem> cache(
      10, [](const int key) { return SizedElem(key); });
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
  for (int i = 0; i < 5; ++i) {
    BOOST_CHECK_EQUAL(cache.Get(i).NumBytes(), i);
    BOOST_CHECK_EQUAL(cache.NumElems(), i + 1);
    BOOST_CHECK(cache.Exists(i));
  }

  BOOST_CHECK_EQUAL(cache.Get(5).NumBytes(), 5);
  BOOST_CHECK_EQUAL(cache.NumElems(), 2);
  BOOST_CHECK_EQUAL(cache.NumBytes(), 9);
  BOOST_CHECK(!cache.Exists(0));
  BOOST_CHECK(cache.Exists(5));

  BOOST_CHECK_EQUAL(cache.Get(5).NumBytes(), 5);
  BOOST_CHECK_EQUAL(cache.NumElems(), 2);
  BOOST_CHECK_EQUAL(cache.NumBytes(), 9);
  BOOST_CHECK(!cache.Exists(0));
  BOOST_CHECK(cache.Exists(5));

  BOOST_CHECK_EQUAL(cache.Get(6).NumBytes(), 6);
  BOOST_CHECK_EQUAL(cache.NumElems(), 1);
  BOOST_CHECK_EQUAL(cache.NumBytes(), 6);
  BOOST_CHECK(!cache.Exists(0));
  BOOST_CHECK(!cache.Exists(1));
  BOOST_CHECK(cache.Exists(6));

  BOOST_CHECK_EQUAL(cache.Get(1).NumBytes(), 1);
  BOOST_CHECK_EQUAL(cache.NumElems(), 2);
  BOOST_CHECK_EQUAL(cache.NumBytes(), 7);
  BOOST_CHECK(!cache.Exists(0));
  BOOST_CHECK(cache.Exists(1));
  BOOST_CHECK(cache.Exists(6));
}

BOOST_AUTO_TEST_CASE(TestMemoryConstrainedLRUCacheClear) {
  MemoryConstrainedLRUCache<int, SizedElem> cache(
      10, [](const int key) { return SizedElem(key); });
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
  for (int i = 0; i < 5; ++i) {
    BOOST_CHECK_EQUAL(cache.Get(i).NumBytes(), i);
    BOOST_CHECK_EQUAL(cache.NumElems(), i + 1);
    BOOST_CHECK(cache.Exists(i));
  }

  cache.Clear();
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
  BOOST_CHECK_EQUAL(cache.NumBytes(), 0);

  BOOST_CHECK_EQUAL(cache.Get(1).NumBytes(), 1);
  BOOST_CHECK_EQUAL(cache.NumBytes(), 1);
  BOOST_CHECK_EQUAL(cache.NumElems(), 1);
  BOOST_CHECK(cache.Exists(1));
}

BOOST_AUTO_TEST_CASE(TestMemoryConstrainedLRUCacheUpdateNumBytes) {
  MemoryConstrainedLRUCache<int, SizedElem> cache(
      50, [](const int key) { return SizedElem(key); });
  BOOST_CHECK_EQUAL(cache.NumElems(), 0);
  for (int i = 0; i < 5; ++i) {
    BOOST_CHECK_EQUAL(cache.Get(i).NumBytes(), i);
    BOOST_CHECK_EQUAL(cache.NumElems(), i + 1);
    BOOST_CHECK(cache.Exists(i));
  }

  BOOST_CHECK_EQUAL(cache.NumBytes(), 10);

  cache.GetMutable(4).num_bytes = 3;
  BOOST_CHECK_EQUAL(cache.NumBytes(), 10);
  cache.UpdateNumBytes(4);
  BOOST_CHECK_EQUAL(cache.NumBytes(), 9);

  cache.GetMutable(2).num_bytes = 3;
  BOOST_CHECK_EQUAL(cache.NumBytes(), 9);
  cache.UpdateNumBytes(2);
  BOOST_CHECK_EQUAL(cache.NumBytes(), 10);

  cache.GetMutable(0).num_bytes = 40;
  BOOST_CHECK_EQUAL(cache.NumBytes(), 10);
  cache.UpdateNumBytes(0);
  BOOST_CHECK_EQUAL(cache.NumBytes(), 50);

  cache.Clear();
  BOOST_CHECK_EQUAL(cache.NumBytes(), 0);
  BOOST_CHECK_EQUAL(cache.Get(2).NumBytes(), 2);
  BOOST_CHECK_EQUAL(cache.NumBytes(), 2);
}
