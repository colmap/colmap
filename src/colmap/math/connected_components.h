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

#include "colmap/math/union_find.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace colmap {

// Find all connected components in a graph.
//
// @param nodes   Set of all nodes in the graph.
// @param edges   List of edges as (node1, node2) pairs.
// @return        Vector of components, each component is a vector of nodes.
template <typename T>
std::vector<std::vector<T>> FindConnectedComponents(
    const std::unordered_set<T>& nodes,
    const std::vector<std::pair<T, T>>& edges) {
  UnionFind<T> uf;
  uf.Reserve(nodes.size());

  for (const auto& [node1, node2] : edges) {
    uf.Union(node1, node2);
  }

  std::unordered_map<T, std::vector<T>> components;
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
// @return        Set of nodes in the largest connected component.
template <typename T>
std::unordered_set<T> FindLargestConnectedComponent(
    const std::unordered_set<T>& nodes,
    const std::vector<std::pair<T, T>>& edges) {
  UnionFind<T> uf;
  uf.Reserve(nodes.size());

  for (const auto& [node1, node2] : edges) {
    uf.Union(node1, node2);
  }

  std::unordered_map<T, std::vector<T>> components;
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

  const auto& largest = components[largest_root];
  return std::unordered_set<T>(largest.begin(), largest.end());
}

}  // namespace colmap
