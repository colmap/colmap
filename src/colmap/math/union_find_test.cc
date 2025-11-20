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

#include "colmap/math/union_find.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(UnionFind, DefaultConstructor) {
  UnionFind<int> uf;
  // After construction, each element should be its own parent
  EXPECT_EQ(uf.Find(1), 1);
  EXPECT_EQ(uf.Find(2), 2);
  EXPECT_EQ(uf.Find(3), 3);
}

TEST(UnionFind, Reserve) {
  // Test constructor with expected size parameter
  UnionFind<int> uf;
  uf.Reserve(10);
  EXPECT_EQ(uf.Find(1), 1);
  EXPECT_EQ(uf.Find(2), 2);
}

TEST(UnionFind, FindSingleElement) {
  UnionFind<int> uf;
  // Finding an element for the first time should return itself
  EXPECT_EQ(uf.Find(42), 42);
  // Finding it again should still return itself
  EXPECT_EQ(uf.Find(42), 42);
}

TEST(UnionFind, UnionTwoElements) {
  UnionFind<int> uf;
  // Union two elements
  uf.Union(1, 2);
  // Both should have the same root
  EXPECT_EQ(uf.Find(1), uf.Find(2));
}

TEST(UnionFind, UnionMultiplePairs) {
  UnionFind<int> uf;
  // Union multiple pairs
  uf.Union(1, 2);
  uf.Union(3, 4);
  uf.Union(5, 6);

  // Elements in the same set should have the same root
  EXPECT_EQ(uf.Find(1), uf.Find(2));
  EXPECT_EQ(uf.Find(3), uf.Find(4));
  EXPECT_EQ(uf.Find(5), uf.Find(6));

  // Elements in different sets should have different roots
  EXPECT_NE(uf.Find(1), uf.Find(3));
  EXPECT_NE(uf.Find(1), uf.Find(5));
  EXPECT_NE(uf.Find(3), uf.Find(5));
}

TEST(UnionFind, UnionChain) {
  UnionFind<int> uf;
  // Create a chain: 1-2-3-4-5
  uf.Union(1, 2);
  uf.Union(2, 3);
  uf.Union(3, 4);
  uf.Union(4, 5);

  // All elements should have the same root
  const int root = uf.Find(1);
  EXPECT_EQ(uf.Find(2), root);
  EXPECT_EQ(uf.Find(3), root);
  EXPECT_EQ(uf.Find(4), root);
  EXPECT_EQ(uf.Find(5), root);
}

TEST(UnionFind, UnionTwoSets) {
  UnionFind<int> uf;
  // Create two separate sets
  uf.Union(1, 2);
  uf.Union(3, 4);

  // Verify they are separate
  EXPECT_NE(uf.Find(1), uf.Find(3));

  // Union the two sets
  uf.Union(2, 3);

  // Now all elements should have the same root
  const int root = uf.Find(1);
  EXPECT_EQ(uf.Find(2), root);
  EXPECT_EQ(uf.Find(3), root);
  EXPECT_EQ(uf.Find(4), root);
}

TEST(UnionFind, UnionSameElementTwice) {
  UnionFind<int> uf;
  // Union an element with itself
  uf.Union(1, 1);
  EXPECT_EQ(uf.Find(1), 1);

  // Union two elements, then union them again
  uf.Union(2, 3);
  const int root = uf.Find(2);
  uf.Union(2, 3);
  // Should still have the same root
  EXPECT_EQ(uf.Find(2), root);
  EXPECT_EQ(uf.Find(3), root);
}

TEST(UnionFind, PathCompression) {
  UnionFind<int> uf;
  // Create a long chain: 1 -> 2 -> 3 -> 4 -> 5
  uf.Union(1, 2);
  uf.Union(2, 3);
  uf.Union(3, 4);
  uf.Union(4, 5);

  // Find root of element 1
  const int root = uf.Find(1);

  // After path compression, finding 1 again should be efficient
  // The root should remain the same
  EXPECT_EQ(uf.Find(1), root);

  // All elements should still have the same root
  EXPECT_EQ(uf.Find(2), root);
  EXPECT_EQ(uf.Find(3), root);
  EXPECT_EQ(uf.Find(4), root);
  EXPECT_EQ(uf.Find(5), root);
}

TEST(UnionFind, LargeNumberOfElements) {
  constexpr int kNumElements = 1000;
  UnionFind<int> uf;
  uf.Reserve(kNumElements);
  // Union many elements
  for (int i = 0; i < kNumElements; ++i) {
    uf.Union(i, (i + 1) % kNumElements);
  }

  // All elements should be in the same set
  const int root = uf.Find(0);
  for (int i = 1; i < kNumElements; ++i) {
    EXPECT_EQ(uf.Find(i), root);
  }
}

TEST(UnionFind, MultipleDisjointSets) {
  UnionFind<int> uf;
  // Create multiple disjoint sets
  // Set 1: {1, 2, 3}
  uf.Union(1, 2);
  uf.Union(2, 3);

  // Set 2: {10, 20, 30}
  uf.Union(10, 20);
  uf.Union(20, 30);

  // Set 3: {100, 200, 300}
  uf.Union(100, 200);
  uf.Union(200, 300);

  // Verify elements in the same set have the same root
  EXPECT_EQ(uf.Find(1), uf.Find(2));
  EXPECT_EQ(uf.Find(2), uf.Find(3));
  EXPECT_EQ(uf.Find(10), uf.Find(20));
  EXPECT_EQ(uf.Find(20), uf.Find(30));
  EXPECT_EQ(uf.Find(100), uf.Find(200));
  EXPECT_EQ(uf.Find(200), uf.Find(300));

  // Verify elements in different sets have different roots
  EXPECT_NE(uf.Find(1), uf.Find(10));
  EXPECT_NE(uf.Find(1), uf.Find(100));
  EXPECT_NE(uf.Find(10), uf.Find(100));
}

TEST(UnionFind, StringType) {
  UnionFind<std::string> uf;
  // Test with string type
  uf.Union("apple", "apricot");
  uf.Union("banana", "blueberry");
  uf.Union("apricot", "avocado");

  // Elements in the same set should have the same root
  EXPECT_EQ(uf.Find("apple"), uf.Find("apricot"));
  EXPECT_EQ(uf.Find("apple"), uf.Find("avocado"));
  EXPECT_EQ(uf.Find("banana"), uf.Find("blueberry"));

  // Elements in different sets should have different roots
  EXPECT_NE(uf.Find("apple"), uf.Find("banana"));
}

TEST(UnionFind, StarTopology) {
  UnionFind<int> uf;
  // Create a star topology: center = 0, connected to 1, 2, 3, 4, 5
  for (int i = 1; i <= 5; ++i) {
    uf.Union(0, i);
  }

  // All elements should be in the same set
  for (int i = 1; i <= 5; ++i) {
    EXPECT_EQ(uf.Find(0), uf.Find(i));
  }
}

TEST(UnionFind, ReverseUnion) {
  UnionFind<int> uf;
  // Union in reverse order
  uf.Union(5, 4);
  uf.Union(4, 3);
  uf.Union(3, 2);
  uf.Union(2, 1);

  // All should be in the same set
  const int root = uf.Find(5);
  for (int i = 1; i <= 5; ++i) {
    EXPECT_EQ(uf.Find(i), root);
  }
}

}  // namespace
}  // namespace colmap
