// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#ifndef COLMAP_SRC_BASE_GRAPH_CUT_H_
#define COLMAP_SRC_BASE_GRAPH_CUT_H_

#include <unordered_map>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/one_bit_color_map.hpp>

#include "util/logging.h"

namespace colmap {

// Compute the min-cut of a undirected graph using the Stoer Wagner algorithm.
void ComputeMinGraphCutStoerWagner(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights, int* cut_weight,
    std::vector<char>* cut_labels);

// Compute the normalized min-cut of an undirected graph using Graclus.
// Partitions the graph into clusters and returns the cluster labels per vertex.
std::unordered_map<int, int> ComputeNormalizedMinGraphCut(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights, const int num_parts);

// Compute the minimum graph cut of a directed S-T graph using the
// Boykov-Kolmogorov max-flow min-cut algorithm, as descibed in:
//   "An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy
//    Minimization in Vision". Yuri Boykov and Vladimir Kolmogorov. PAMI, 2004.
template <typename node_t, typename value_t>
class MinSTGraphCut {
 public:
  typedef boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                       boost::directedS>
      graph_traits_t;
  typedef graph_traits_t::edge_descriptor edge_descriptor_t;
  typedef graph_traits_t::vertices_size_type vertices_size_t;

  struct Edge {
    value_t capacity;
    value_t residual;
    edge_descriptor_t reverse;
  };

  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                                size_t, Edge>
      graph_t;

  MinSTGraphCut(const size_t num_nodes);

  // Count the number of nodes and edges in the graph.
  size_t NumNodes() const;
  size_t NumEdges() const;

  // Add node to the graph.
  void AddNode(const node_t node_idx, const value_t source_capacity,
               const value_t sink_capacity);

  // Add edge to the graph.
  void AddEdge(const node_t node_idx1, const node_t node_idx2,
               const value_t capacity, const value_t reverse_capacity);

  // Compute the min-cut using the max-flow algorithm. Returns the flow.
  value_t Compute();

  // Check whether node is connected to source or sink after computing the cut.
  bool IsConnectedToSource(const node_t node_idx) const;
  bool IsConnectedToSink(const node_t node_idx) const;

 private:
  const node_t S_node_;
  const node_t T_node_;
  graph_t graph_;
  std::vector<boost::default_color_type> colors_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename node_t, typename value_t>
MinSTGraphCut<node_t, value_t>::MinSTGraphCut(const size_t num_nodes)
    : S_node_(num_nodes), T_node_(num_nodes + 1), graph_(num_nodes + 2) {}

template <typename node_t, typename value_t>
size_t MinSTGraphCut<node_t, value_t>::NumNodes() const {
  return boost::num_vertices(graph_) - 2;
}

template <typename node_t, typename value_t>
size_t MinSTGraphCut<node_t, value_t>::NumEdges() const {
  return boost::num_edges(graph_);
}

template <typename node_t, typename value_t>
void MinSTGraphCut<node_t, value_t>::AddNode(const node_t node_idx,
                                             const value_t source_capacity,
                                             const value_t sink_capacity) {
  CHECK_GE(node_idx, 0);
  CHECK_LE(node_idx, boost::num_vertices(graph_));
  CHECK_GE(source_capacity, 0);
  CHECK_GE(sink_capacity, 0);

  if (source_capacity > 0) {
    const edge_descriptor_t edge =
        boost::add_edge(S_node_, node_idx, graph_).first;
    const edge_descriptor_t edge_reverse =
        boost::add_edge(node_idx, S_node_, graph_).first;
    graph_[edge].capacity = source_capacity;
    graph_[edge].reverse = edge_reverse;
    graph_[edge_reverse].reverse = edge;
  }

  if (sink_capacity > 0) {
    const edge_descriptor_t edge =
        boost::add_edge(node_idx, T_node_, graph_).first;
    const edge_descriptor_t edge_reverse =
        boost::add_edge(T_node_, node_idx, graph_).first;
    graph_[edge].capacity = sink_capacity;
    graph_[edge].reverse = edge_reverse;
    graph_[edge_reverse].reverse = edge;
  }
}

template <typename node_t, typename value_t>
void MinSTGraphCut<node_t, value_t>::AddEdge(const node_t node_idx1,
                                             const node_t node_idx2,
                                             const value_t capacity,
                                             const value_t reverse_capacity) {
  CHECK_GE(node_idx1, 0);
  CHECK_LE(node_idx1, boost::num_vertices(graph_));
  CHECK_GE(node_idx2, 0);
  CHECK_LE(node_idx2, boost::num_vertices(graph_));
  CHECK_GE(capacity, 0);
  CHECK_GE(reverse_capacity, 0);

  const edge_descriptor_t edge =
      boost::add_edge(node_idx1, node_idx2, graph_).first;
  const edge_descriptor_t edge_reverse =
      boost::add_edge(node_idx2, node_idx1, graph_).first;
  graph_[edge].capacity = capacity;
  graph_[edge_reverse].capacity = reverse_capacity;
  graph_[edge].reverse = edge_reverse;
  graph_[edge_reverse].reverse = edge;
}

template <typename node_t, typename value_t>
value_t MinSTGraphCut<node_t, value_t>::Compute() {
  const vertices_size_t num_vertices = boost::num_vertices(graph_);

  colors_.resize(num_vertices);
  std::vector<edge_descriptor_t> predecessors(num_vertices);
  std::vector<vertices_size_t> distances(num_vertices);

  return boost::boykov_kolmogorov_max_flow(
      graph_, boost::get(&Edge::capacity, graph_),
      boost::get(&Edge::residual, graph_), boost::get(&Edge::reverse, graph_),
      predecessors.data(), colors_.data(), distances.data(),
      boost::get(boost::vertex_index, graph_), S_node_, T_node_);
}

template <typename node_t, typename value_t>
bool MinSTGraphCut<node_t, value_t>::IsConnectedToSource(
    const node_t node_idx) const {
  return colors_.at(node_idx) != boost::white_color;
}

template <typename node_t, typename value_t>
bool MinSTGraphCut<node_t, value_t>::IsConnectedToSink(
    const node_t node_idx) const {
  return colors_.at(node_idx) == boost::white_color;
}

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_GRAPH_CUT_H_
