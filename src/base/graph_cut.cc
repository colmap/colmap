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

#include "base/graph_cut.h"

#include <unordered_map>

#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/typeof/typeof.hpp>

extern "C" {
#include "Graclus/metisLib/metis.h"
}

#include "util/logging.h"

namespace colmap {
namespace {

// Wrapper class for weighted, undirected Graclus graph.
class GraclusGraph {
 public:
  GraclusGraph(const std::vector<std::pair<int, int>>& edges,
               const std::vector<int>& weights) {
    CHECK_EQ(edges.size(), weights.size());

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

    idxtype edge_idx = 0;
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

    data.gdata = data.rdata = nullptr;

    data.nvtxs = vertex_id_to_idx_.size();
    data.nedges = 2 * edges.size();
    data.mincut = data.minvol = -1;

    data.xadj = xadj_.data();
    data.adjncy = adjncy_.data();

    data.vwgt = nullptr;
    data.adjwgt = adjwgt_.data();

    data.adjwgtsum = nullptr;
    data.label = nullptr;
    data.cmap = nullptr;

    data.where = data.pwgts = nullptr;
    data.id = data.ed = nullptr;
    data.bndptr = data.bndind = nullptr;
    data.rinfo = nullptr;
    data.vrinfo = nullptr;
    data.nrinfo = nullptr;

    data.ncon = 1;
    data.nvwgt = nullptr;
    data.npwgts = nullptr;

    data.vsize = nullptr;

    data.coarser = data.finer = nullptr;
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

  GraphType data;

 private:
  std::unordered_map<int, int> vertex_id_to_idx_;
  std::unordered_map<int, int> vertex_idx_to_id_;
  std::vector<idxtype> xadj_;
  std::vector<idxtype> adjncy_;
  std::vector<idxtype> adjwgt_;
};

}  // namespace

void ComputeMinGraphCutStoerWagner(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights, int* cut_weight,
    std::vector<char>* cut_labels) {
  CHECK_EQ(edges.size(), weights.size());
  CHECK_GE(edges.size(), 2);

  typedef boost::property<boost::edge_weight_t, int> edge_weight_t;
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                                boost::no_property, edge_weight_t>
      undirected_graph_t;

  int max_vertex_index = 0;
  for (const auto& edge : edges) {
    CHECK_GE(edge.first, 0);
    CHECK_GE(edge.second, 0);
    max_vertex_index = std::max(max_vertex_index, edge.first);
    max_vertex_index = std::max(max_vertex_index, edge.second);
  }

  const undirected_graph_t graph(edges.begin(), edges.end(), weights.begin(),
                                 max_vertex_index + 1, edges.size());

  const auto parities = boost::make_one_bit_color_map(
      boost::num_vertices(graph), boost::get(boost::vertex_index, graph));

  *cut_weight =
      boost::stoer_wagner_min_cut(graph, boost::get(boost::edge_weight, graph),
                                  boost::parity_map(parities));

  cut_labels->resize(boost::num_vertices(graph));
  for (size_t i = 0; i < boost::num_vertices(graph); ++i) {
    (*cut_labels)[i] = boost::get(parities, i);
  }
}

std::unordered_map<int, int> ComputeNormalizedMinGraphCut(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights, const int num_parts) {
  GraclusGraph graph(edges, weights);

  const int levels =
      amax((graph.data.nvtxs) / (40 * log2_metis(num_parts)), 20 * (num_parts));

  std::vector<idxtype> cut_labels(graph.data.nvtxs);

  int options[11];
  options[0] = 0;
  int wgtflag = 1;
  int numflag = 0;
  int chain_length = 0;
  int edgecut;
  int var_num_parts = num_parts;

  MLKKM_PartGraphKway(&graph.data.nvtxs, graph.data.xadj, graph.data.adjncy,
                      graph.data.vwgt, graph.data.adjwgt, &wgtflag, &numflag,
                      &var_num_parts, &chain_length, options, &edgecut,
                      cut_labels.data(), levels);

  float lbvec[MAXNCON];
  ComputePartitionBalance(&graph.data, num_parts, cut_labels.data(), lbvec);

  ComputeNCut(&graph.data, &cut_labels[0], num_parts);

  std::unordered_map<int, int> labels;
  for (size_t idx = 0; idx < cut_labels.size(); ++idx) {
    labels.emplace(graph.GetVertexId(idx), cut_labels[idx]);
  }

  return labels;
}

}  // namespace colmap
