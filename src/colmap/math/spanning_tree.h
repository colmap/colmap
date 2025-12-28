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

#pragma once

#include <cstddef>
#include <vector>

namespace colmap {

// Represents a rooted spanning tree as a parent map.
// For each node index i, parents[i] contains the index of its parent.
// The root node has parents[root] == root.
struct SpanningTree {
  int root = -1;
  std::vector<int> parents;

  // Check if tree is valid (has a root and non-empty parents).
  bool IsValid() const { return root >= 0 && !parents.empty(); }

  // Get the number of nodes in the tree.
  size_t NumNodes() const { return parents.size(); }
};

// Compute the maximum spanning tree of an undirected weighted graph.
//
// The graph is specified by:
// - num_nodes: Number of nodes in the graph (nodes are indexed 0 to
// num_nodes-1)
// - edges: List of edges as (node1, node2) pairs
// - weights: Weight for each edge (higher weight = preferred in max spanning
// tree)
// - root: The root node for the resulting tree (default 0)
//
// Returns a SpanningTree with parent pointers rooted at the specified root.
// If the graph is disconnected, only the component containing root is included.
//
// Uses Kruskal's algorithm with negated weights to find maximum spanning tree.
SpanningTree ComputeMaximumSpanningTree(
    int num_nodes,
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<double>& weights,
    int root = 0);

// Compute the minimum spanning tree of an undirected weighted graph.
//
// Same interface as ComputeMaximumSpanningTree, but finds minimum weight tree.
SpanningTree ComputeMinimumSpanningTree(
    int num_nodes,
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<double>& weights,
    int root = 0);

}  // namespace colmap
