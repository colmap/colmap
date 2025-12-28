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

#include <queue>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

namespace colmap {
namespace {

using BoostGraph = boost::adjacency_list<boost::vecS,
                                         boost::vecS,
                                         boost::undirectedS,
                                         boost::no_property,
                                         boost::property<boost::edge_weight_t, double>>;
using EdgeDescriptor = boost::graph_traits<BoostGraph>::edge_descriptor;

// Build parent pointers from adjacency list using BFS from root.
void BuildParentsFromAdjacencyList(
    const std::vector<std::vector<int>>& adjacency_list,
    int root,
    std::vector<int>& parents) {
  const int num_nodes = static_cast<int>(adjacency_list.size());
  parents.assign(num_nodes, -1);
  parents[root] = root;

  std::vector<bool> visited(num_nodes, false);
  visited[root] = true;

  std::queue<int> queue;
  queue.push(root);

  while (!queue.empty()) {
    const int current = queue.front();
    queue.pop();

    for (const int neighbor : adjacency_list[current]) {
      if (!visited[neighbor]) {
        visited[neighbor] = true;
        parents[neighbor] = current;
        queue.push(neighbor);
      }
    }
  }
}

SpanningTree ComputeSpanningTreeInternal(
    int num_nodes,
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<double>& weights,
    int root,
    bool maximize) {
  SpanningTree tree;
  if (num_nodes <= 0) {
    return tree;
  }

  // For maximum spanning tree, we negate weights and find minimum.
  double max_weight = 0;
  if (maximize) {
    for (const double w : weights) {
      max_weight = std::max(max_weight, w);
    }
  }

  // Build boost graph.
  BoostGraph graph(num_nodes);
  auto weight_map = boost::get(boost::edge_weight, graph);

  for (size_t i = 0; i < edges.size(); ++i) {
    const auto& edge = edges[i];
    const double weight = maximize ? (max_weight - weights[i]) : weights[i];
    auto [e, inserted] = boost::add_edge(edge.first, edge.second, graph);
    if (inserted) {
      weight_map[e] = weight;
    }
  }

  // Run Kruskal's algorithm.
  std::vector<EdgeDescriptor> mst_edges;
  boost::kruskal_minimum_spanning_tree(graph, std::back_inserter(mst_edges));

  // Convert MST edges to adjacency list.
  std::vector<std::vector<int>> adjacency_list(num_nodes);
  for (const auto& edge : mst_edges) {
    const int source = static_cast<int>(boost::source(edge, graph));
    const int target = static_cast<int>(boost::target(edge, graph));
    adjacency_list[source].push_back(target);
    adjacency_list[target].push_back(source);
  }

  // Build parent pointers via BFS from specified root.
  tree.root = root;
  BuildParentsFromAdjacencyList(adjacency_list, tree.root, tree.parents);

  return tree;
}

}  // namespace

SpanningTree ComputeMaximumSpanningTree(
    int num_nodes,
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<double>& weights,
    int root) {
  return ComputeSpanningTreeInternal(num_nodes, edges, weights, root, true);
}

SpanningTree ComputeMinimumSpanningTree(
    int num_nodes,
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<double>& weights,
    int root) {
  return ComputeSpanningTreeInternal(num_nodes, edges, weights, root, false);
}

}  // namespace colmap
