// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/math/union_find.h"
#include "colmap/util/hash_containers.h"

#include <vector>

namespace colmap {

// Find all connected components in a graph.
//
// @param nodes   Set of all nodes in the graph.
// @param edges   List of edges as (node1, node2) pairs.
// @return        Vector of components, each component is a vector of nodes.
template <typename T>
std::vector<std::vector<T>> FindConnectedComponents(
    const FlatHashSet<T>& nodes, const std::vector<std::pair<T, T>>& edges) {
  UnionFind<T> uf;
  uf.Reserve(nodes.size());

  for (const auto& [node1, node2] : edges) {
    uf.Union(node1, node2);
  }

  NodeHashMap<T, std::vector<T>> components;
  for (const T& node : nodes) {
    components[uf.Find(node)].push_back(node);
  }

  std::vector<std::vector<T>> result;
  result.reserve(components.size());
  for (auto& [root, members] : components) {
    result.push_back(std::move(members));
  }

  return result;
}

// Find the largest connected component in a graph.
//
// @param nodes   Set of all nodes in the graph.
// @param edges   List of edges as (node1, node2) pairs.
// @return        Vector of nodes in the largest connected component.
template <typename T>
std::vector<T> FindLargestConnectedComponent(
    const FlatHashSet<T>& nodes, const std::vector<std::pair<T, T>>& edges) {
  UnionFind<T> uf;
  uf.Reserve(nodes.size());

  for (const auto& [node1, node2] : edges) {
    uf.Union(node1, node2);
  }

  NodeHashMap<T, std::vector<T>> components;
  for (const T& node : nodes) {
    components[uf.Find(node)].push_back(node);
  }

  T largest_root = T();
  size_t largest_size = 0;
  for (const auto& [root, members] : components) {
    if (members.size() > largest_size) {
      largest_size = members.size();
      largest_root = root;
    }
  }

  return std::move(components[largest_root]);
}

}  // namespace colmap
