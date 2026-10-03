// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/math/spanning_tree.h"

#include <queue>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

namespace colmap {
namespace {

using BoostGraph =
    boost::adjacency_list<boost::vecS,
                          boost::vecS,
                          boost::undirectedS,
                          boost::no_property,
                          boost::property<boost::edge_weight_t, float>>;
using EdgeDescriptor = boost::graph_traits<BoostGraph>::edge_descriptor;

// Build parent pointers from adjacency list using BFS from root.
void BuildParentsFromAdjacencyList(
    const std::vector<std::vector<int>>& adjacency_list,
    int root,
    std::vector<int>& parents) {
  const int num_nodes = static_cast<int>(adjacency_list.size());
  parents.assign(num_nodes, -1);
  parents[root] = root;

  std::vector<char> visited(num_nodes, false);
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
    const std::vector<float>& weights,
    int root,
    bool maximize) {
  SpanningTree tree;
  if (num_nodes <= 0) {
    return tree;
  }

  // For maximum spanning tree, we negate weights and find minimum.
  float max_weight = 0;
  if (maximize) {
    for (const float w : weights) {
      max_weight = std::max(max_weight, w);
    }
  }

  // Build boost graph.
  BoostGraph graph(num_nodes);
  auto weight_map = boost::get(boost::edge_weight, graph);

  for (size_t i = 0; i < edges.size(); ++i) {
    const auto& edge = edges[i];
    const float weight = maximize ? (max_weight - weights[i]) : weights[i];
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
    const std::vector<float>& weights,
    int root) {
  return ComputeSpanningTreeInternal(num_nodes, edges, weights, root, true);
}

SpanningTree ComputeMinimumSpanningTree(
    int num_nodes,
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<float>& weights,
    int root) {
  return ComputeSpanningTreeInternal(num_nodes, edges, weights, root, false);
}

}  // namespace colmap
