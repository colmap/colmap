// SPDX-License-Identifier: BSD-3-Clause

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
    const std::vector<float>& weights,
    int root = 0);

// Compute the minimum spanning tree of an undirected weighted graph.
//
// Same interface as ComputeMaximumSpanningTree, but finds minimum weight tree.
SpanningTree ComputeMinimumSpanningTree(
    int num_nodes,
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<float>& weights,
    int root = 0);

}  // namespace colmap
