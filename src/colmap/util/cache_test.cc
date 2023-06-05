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

#define TEST_NAME "util/cache"
#include "colmap/util/cache.h"

#include "colmap/util/testing.h"

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
