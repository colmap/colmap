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

#include "colmap/base/graph_cut.h"

#include <unordered_map>

#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/typeof/typeof.hpp>

extern "C" {
#include "metis.h"
}

#include "colmap/util/logging.h"

namespace colmap {
namespace {

// Wrapper class for weighted, undirected Metis graph.
class MetisGraph {
 public:
  MetisGraph(const std::vector<std::pair<int, int>>& edges,
             const std::vector<int>& weights) {
    std::unordered_map<int, std::vector<std::pair<int, int>>> adjacency_list;
    for (size_t i = 0; i < edges.size(); ++i) {
      const auto& edge = edges[i];
      const auto weight = weights[i];
      const int vertex_idx1 = GetVertexIdx(edge.first);
      const int vertex_idx2 = GetVertexIdx(edge.second);
      adjacency_list[vertex_idx1].emplace_back(vertex_idx2, weight);
      adjacency_list[vertex_idx2].emplace_back(vertex_idx1, weight);
    }

    xadj_.reserve(vertex_id_to_idx_.size() + 1);
    adjncy_.reserve(2 * edges.size());
    adjwgt_.reserve(2 * edges.size());

    idx_t edge_idx = 0;
    for (size_t i = 0; i < vertex_id_to_idx_.size(); ++i) {
      xadj_.push_back(edge_idx);

      if (adjacency_list.count(i) == 0) {
        continue;
      }

      for (const auto& edge : adjacency_list[i]) {
        edge_idx += 1;
        adjncy_.push_back(edge.first);
        adjwgt_.push_back(edge.second);
      }
    }

    xadj_.push_back(edge_idx);

    CHECK_EQ(edge_idx, 2 * edges.size());
    CHECK_EQ(xadj_.size(), vertex_id_to_idx_.size() + 1);
    CHECK_EQ(adjncy_.size(), 2 * edges.size());
    CHECK_EQ(adjwgt_.size(), 2 * edges.size());

    nvtxs = vertex_id_to_idx_.size();

    xadj = xadj_.data();
    adjncy = adjncy_.data();

    vwgt = nullptr;
    adjwgt = adjwgt_.data();
  }

  int GetVertexIdx(const int id) {
    const auto it = vertex_id_to_idx_.find(id);
    if (it == vertex_id_to_idx_.end()) {
      const int idx = vertex_id_to_idx_.size();
      vertex_id_to_idx_.emplace(id, idx);
      vertex_idx_to_id_.emplace(idx, id);
      return idx;
    } else {
      return it->second;
    }
  }

  int GetVertexId(const int idx) { return vertex_idx_to_id_.at(idx); }

  idx_t nvtxs = 0;
  idx_t* xadj = nullptr;
  idx_t* vwgt = nullptr;
  idx_t* adjncy = nullptr;
  idx_t* adjwgt = nullptr;

 private:
  std::unordered_map<int, int> vertex_id_to_idx_;
  std::unordered_map<int, int> vertex_idx_to_id_;
  std::vector<idx_t> xadj_;
  std::vector<idx_t> adjncy_;
  std::vector<idx_t> adjwgt_;
};

}  // namespace

void ComputeMinGraphCutStoerWagner(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights,
    int* cut_weight,
    std::vector<char>* cut_labels) {
  CHECK_EQ(edges.size(), weights.size());
  CHECK_GE(edges.size(), 2);

  typedef boost::property<boost::edge_weight_t, int> edge_weight_t;
  typedef boost::adjacency_list<boost::vecS,
                                boost::vecS,
                                boost::undirectedS,
                                boost::no_property,
                                edge_weight_t>
      undirected_graph_t;

  int max_vertex_index = 0;
  for (const auto& edge : edges) {
    CHECK_GE(edge.first, 0);
    CHECK_GE(edge.second, 0);
    max_vertex_index = std::max(max_vertex_index, edge.first);
    max_vertex_index = std::max(max_vertex_index, edge.second);
  }

  const undirected_graph_t graph(edges.begin(),
                                 edges.end(),
                                 weights.begin(),
                                 max_vertex_index + 1,
                                 edges.size());

  const auto edge_weight = boost::get(boost::edge_weight, graph);
  const auto parities = boost::make_one_bit_color_map(
      boost::num_vertices(graph), boost::get(boost::vertex_index, graph));
  const auto parity_map = boost::parity_map(parities);

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDelete)
  *cut_weight = boost::stoer_wagner_min_cut(graph, edge_weight, parity_map);

  cut_labels->resize(boost::num_vertices(graph));
  for (size_t i = 0; i < boost::num_vertices(graph); ++i) {
    (*cut_labels)[i] = boost::get(parities, i);
  }
}

std::unordered_map<int, int> ComputeNormalizedMinGraphCut(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights,
    const int num_parts) {
  CHECK(!edges.empty());
  CHECK_EQ(edges.size(), weights.size());
  CHECK_GT(num_parts, 0);

  MetisGraph graph(edges, weights);

  idx_t ncon = 1;
  idx_t edgecut = -1;
  idx_t nparts = num_parts;

  idx_t metisOptions[METIS_NOPTIONS];
  METIS_SetDefaultOptions(metisOptions);

  std::vector<idx_t> cut_labels(graph.nvtxs, -1);
  const int metisResult = METIS_PartGraphKway(&graph.nvtxs,
                                              /*ncon=*/&ncon,
                                              graph.xadj,
                                              graph.adjncy,
                                              /*vwgt=*/nullptr,
                                              /*vsize=*/nullptr,
                                              graph.adjwgt,
                                              &nparts,
                                              /*tpwgts=*/nullptr,
                                              /*ubvec=*/nullptr,
                                              metisOptions,
                                              &edgecut,
                                              cut_labels.data());

  if (metisResult == METIS_ERROR_INPUT) {
    LOG(FATAL) << "INTERNAL: Metis input error";
  } else if (metisResult == METIS_ERROR_MEMORY) {
    LOG(FATAL) << "INTERNAL: Metis memory error";
  } else if (metisResult == METIS_ERROR) {
    LOG(FATAL) << "INTERNAL: Metis 'some other type of error'";
  }

  std::unordered_map<int, int> labels;
  for (size_t idx = 0; idx < cut_labels.size(); ++idx) {
    labels.emplace(graph.GetVertexId(idx), cut_labels[idx]);
  }

  return labels;
}

}  // namespace colmap
