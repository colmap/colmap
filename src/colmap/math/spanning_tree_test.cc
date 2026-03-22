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

#include "colmap/math/spanning_tree.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Helper to compute total weight of edges in the spanning tree.
float ComputeTreeWeight(const SpanningTree& tree,
                        const std::vector<std::pair<int, int>>& edges,
                        const std::vector<float>& weights) {
  float total = 0;
  for (size_t i = 0; i < edges.size(); ++i) {
    int u = edges[i].first;
    int v = edges[i].second;
    // Check if this edge is in the tree (either direction).
    if (tree.parents[u] == v || tree.parents[v] == u) {
      total += weights[i];
    }
  }
  return total;
}

TEST(SpanningTree, Nominal) {
  // Triangle: edges with weights 1, 2, 3.
  // Max spanning tree uses edges 2+3=5, min uses 1+2=3.
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {0, 2}};
  const std::vector<float> weights = {1.0f, 2.0f, 3.0f};

  const SpanningTree max_tree = ComputeMaximumSpanningTree(3, edges, weights);
  const SpanningTree min_tree = ComputeMinimumSpanningTree(3, edges, weights);

  EXPECT_EQ(ComputeTreeWeight(max_tree, edges, weights), 5.0f);
  EXPECT_EQ(ComputeTreeWeight(min_tree, edges, weights), 3.0f);
}

TEST(SpanningTree, DisconnectedGraph) {
  // Two components: {0,1} and {2,3}. Only component containing root is
  // included.
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {2, 3}};
  const std::vector<float> weights = {1.0f, 2.0f};

  // Root at 0: includes {0,1}, excludes {2,3}.
  const SpanningTree tree0 = ComputeMaximumSpanningTree(4, edges, weights, 0);
  EXPECT_EQ(tree0.root, 0);
  EXPECT_EQ(tree0.parents[0], 0);
  EXPECT_EQ(tree0.parents[1], 0);
  EXPECT_EQ(tree0.parents[2], -1);
  EXPECT_EQ(tree0.parents[3], -1);

  // Root at 2: includes {2,3}, excludes {0,1}.
  const SpanningTree tree2 = ComputeMaximumSpanningTree(4, edges, weights, 2);
  EXPECT_EQ(tree2.root, 2);
  EXPECT_EQ(tree2.parents[0], -1);
  EXPECT_EQ(tree2.parents[1], -1);
  EXPECT_EQ(tree2.parents[2], 2);
  EXPECT_EQ(tree2.parents[3], 2);
}

TEST(SpanningTree, EmptyGraph) {
  const SpanningTree tree = ComputeMaximumSpanningTree(0, {}, {});
  EXPECT_FALSE(tree.IsValid());
}

}  // namespace
}  // namespace colmap
