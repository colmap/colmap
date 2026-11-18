// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/math/graph_cut.h"

#include "colmap/util/hash_containers.h"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/property_map/property_map.hpp>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning(push)
// Disable warning C4005: 'PRId32': macro redefinition
#pragma warning(disable : 4005)
#endif
extern "C" {
#include <metis.h>
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "colmap/util/logging.h"

namespace colmap {
namespace {

// Wrapper class for weighted, undirected Metis graph.
class MetisGraph {
 public:
  MetisGraph(const std::vector<std::pair<int, int>>& edges,
             const std::vector<int>& weights) {
    // Preserve first-seen vertex order while counting degrees.
    xadj_.push_back(0);
    for (const auto& [vertex_id1, vertex_id2] : edges) {
      const int vertex_idx1 = GetVertexIdx(vertex_id1);
      const int vertex_idx2 = GetVertexIdx(vertex_id2);
      ++xadj_[static_cast<size_t>(vertex_idx1) + 1];
      ++xadj_[static_cast<size_t>(vertex_idx2) + 1];
    }
    for (size_t i = 1; i < xadj_.size(); ++i) {
      xadj_[i] += xadj_[i - 1];
    }

    adjncy_.resize(2 * edges.size());
    adjwgt_.resize(2 * edges.size());
    std::vector<idx_t> cursors(xadj_.begin(), xadj_.end() - 1);
    // Replay input order within every row, retaining both directions, parallel
    // edges and both entries for a self-loop exactly as before.
    for (size_t i = 0; i < edges.size(); ++i) {
      const int vertex_idx1 = vertex_id_to_idx_.at(edges[i].first);
      const int vertex_idx2 = vertex_id_to_idx_.at(edges[i].second);
      const idx_t edge_idx1 = cursors[vertex_idx1]++;
      const idx_t edge_idx2 = cursors[vertex_idx2]++;
      adjncy_[edge_idx1] = vertex_idx2;
      adjwgt_[edge_idx1] = weights[i];
      adjncy_[edge_idx2] = vertex_idx1;
      adjwgt_[edge_idx2] = weights[i];
    }

    THROW_CHECK_EQ(xadj_.back(), 2 * edges.size());
    THROW_CHECK_EQ(xadj_.size(), vertex_id_to_idx_.size() + 1);

    nvtxs = vertex_id_to_idx_.size();

    xadj = xadj_.data();
    adjncy = adjncy_.data();
    adjwgt = adjwgt_.data();
  }

  int GetVertexIdx(const int id) {
    const auto it = vertex_id_to_idx_.find(id);
    if (it == vertex_id_to_idx_.end()) {
      const int idx = vertex_id_to_idx_.size();
      vertex_id_to_idx_.emplace(id, idx);
      vertex_idx_to_id_.push_back(id);
      xadj_.push_back(0);
      return idx;
    } else {
      return it->second;
    }
  }

  int GetVertexId(const int idx) { return vertex_idx_to_id_.at(idx); }

  idx_t nvtxs = 0;
  idx_t* xadj = nullptr;
  idx_t* adjncy = nullptr;
  idx_t* adjwgt = nullptr;

 private:
  FlatHashMap<int, int> vertex_id_to_idx_;
  std::vector<int> vertex_idx_to_id_;
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
  THROW_CHECK_EQ(edges.size(), weights.size());
  THROW_CHECK_GE(edges.size(), 2);

  using edge_weight_t = boost::property<boost::edge_weight_t, int>;
  using undirected_graph_t = boost::adjacency_list<boost::vecS,
                                                   boost::vecS,
                                                   boost::undirectedS,
                                                   boost::no_property,
                                                   edge_weight_t>;

  int max_vertex_index = 0;
  for (const auto& [v1, v2] : edges) {
    THROW_CHECK_GE(v1, 0);
    THROW_CHECK_GE(v2, 0);
    max_vertex_index = std::max(max_vertex_index, v1);
    max_vertex_index = std::max(max_vertex_index, v2);
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

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDelete,clang-analyzer-core.uninitialized.Assign)
  *cut_weight = boost::stoer_wagner_min_cut(graph, edge_weight, parity_map);

  cut_labels->resize(boost::num_vertices(graph));
  for (size_t i = 0; i < boost::num_vertices(graph); ++i) {
    (*cut_labels)[i] = boost::get(parities, i);
  }
}

NodeHashMap<int, int> ComputeNormalizedMinGraphCut(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights,
    const int num_parts) {
  THROW_CHECK(!edges.empty());
  THROW_CHECK_EQ(edges.size(), weights.size());
  THROW_CHECK_GT(num_parts, 0);

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
    LOG(FATAL_THROW) << "INTERNAL: Metis input error";
  } else if (metisResult == METIS_ERROR_MEMORY) {
    LOG(FATAL_THROW) << "INTERNAL: Metis memory error";
  } else if (metisResult == METIS_ERROR) {
    LOG(FATAL_THROW) << "INTERNAL: Metis 'some other type of error'";
  }

  NodeHashMap<int, int> labels;
  for (size_t idx = 0; idx < cut_labels.size(); ++idx) {
    labels.emplace(graph.GetVertexId(idx), cut_labels[idx]);
  }

  return labels;
}

}  // namespace colmap
