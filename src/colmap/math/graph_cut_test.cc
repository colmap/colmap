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

#include "colmap/math/graph_cut.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(GraphCut, ComputeMinGraphCutStoerWagner) {
  const std::vector<std::pair<int, int>> edges = {{3, 4},
                                                  {3, 6},
                                                  {3, 5},
                                                  {0, 4},
                                                  {0, 1},
                                                  {0, 6},
                                                  {0, 7},
                                                  {0, 5},
                                                  {0, 2},
                                                  {4, 1},
                                                  {1, 6},
                                                  {1, 5},
                                                  {6, 7},
                                                  {7, 5},
                                                  {5, 2},
                                                  {3, 4}};
  const std::vector<int> weights = {
      0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1, 1, 4};
  int cut_weight;
  std::vector<char> cut_labels;
  ComputeMinGraphCutStoerWagner(edges, weights, &cut_weight, &cut_labels);
  EXPECT_EQ(cut_weight, 7);
  EXPECT_EQ(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    EXPECT_GE(label, 0);
    EXPECT_LT(label, 2);
  }
}

TEST(GraphCut, ComputeMinGraphCutStoerWagnerDuplicateEdge) {
  const std::vector<std::pair<int, int>> edges = {{3, 4},
                                                  {3, 6},
                                                  {3, 5},
                                                  {0, 4},
                                                  {0, 1},
                                                  {0, 6},
                                                  {0, 7},
                                                  {0, 5},
                                                  {0, 2},
                                                  {4, 1},
                                                  {1, 6},
                                                  {1, 5},
                                                  {6, 7},
                                                  {7, 5},
                                                  {5, 2},
                                                  {3, 4},
                                                  {3, 4}};
  const std::vector<int> weights = {
      0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1, 1, 4, 4};
  int cut_weight;
  std::vector<char> cut_labels;
  ComputeMinGraphCutStoerWagner(edges, weights, &cut_weight, &cut_labels);
  EXPECT_EQ(cut_weight, 7);
  EXPECT_EQ(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    EXPECT_GE(label, 0);
    EXPECT_LT(label, 2);
  }
}

TEST(GraphCut, ComputeMinGraphCutStoerWagnerMissingVertex) {
  const std::vector<std::pair<int, int>> edges = {{3, 4},
                                                  {3, 6},
                                                  {3, 5},
                                                  {0, 1},
                                                  {0, 6},
                                                  {0, 7},
                                                  {0, 5},
                                                  {0, 2},
                                                  {4, 1},
                                                  {1, 6},
                                                  {1, 5},
                                                  {6, 7},
                                                  {7, 5},
                                                  {5, 2}};
  const std::vector<int> weights = {0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1};
  int cut_weight;
  std::vector<char> cut_labels;
  ComputeMinGraphCutStoerWagner(edges, weights, &cut_weight, &cut_labels);
  EXPECT_EQ(cut_weight, 2);
  EXPECT_EQ(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    EXPECT_GE(label, 0);
    EXPECT_LT(label, 2);
  }
}

TEST(GraphCut, ComputeMinGraphCutStoerWagnerDisconnected) {
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {3, 4}};
  const std::vector<int> weights = {1, 3, 1};
  int cut_weight;
  std::vector<char> cut_labels;
  ComputeMinGraphCutStoerWagner(edges, weights, &cut_weight, &cut_labels);
  EXPECT_EQ(cut_weight, 0);
  EXPECT_EQ(cut_labels.size(), 5);
  for (const auto& label : cut_labels) {
    EXPECT_GE(label, 0);
    EXPECT_LT(label, 2);
  }
}

TEST(GraphCut, ComputeNormalizedMinGraphCut) {
  const std::vector<std::pair<int, int>> edges = {{3, 4},
                                                  {3, 6},
                                                  {3, 5},
                                                  {0, 4},
                                                  {0, 1},
                                                  {0, 6},
                                                  {0, 7},
                                                  {0, 5},
                                                  {0, 2},
                                                  {4, 1},
                                                  {1, 6},
                                                  {1, 5},
                                                  {6, 7},
                                                  {7, 5},
                                                  {5, 2},
                                                  {3, 4}};
  const std::vector<int> weights = {
      0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1, 1, 4};
  const auto cut_labels = ComputeNormalizedMinGraphCut(edges, weights, 2);
  EXPECT_EQ(cut_labels.size(), 8);
  size_t num_labels[2] = {0};
  for (const auto& label : cut_labels) {
    EXPECT_GE(label.second, 0);
    EXPECT_LT(label.second, 2);
    num_labels[label.second] += 1;
  }
  EXPECT_GT(num_labels[0], 0);
  EXPECT_GT(num_labels[1], 0);
}

TEST(GraphCut, ComputeNormalizedMinGraphCutDuplicateEdge) {
  const std::vector<std::pair<int, int>> edges = {{3, 4},
                                                  {3, 6},
                                                  {3, 5},
                                                  {0, 4},
                                                  {0, 1},
                                                  {0, 6},
                                                  {0, 7},
                                                  {0, 5},
                                                  {0, 2},
                                                  {4, 1},
                                                  {1, 6},
                                                  {1, 5},
                                                  {6, 7},
                                                  {7, 5},
                                                  {5, 2},
                                                  {3, 4},
                                                  {3, 4}};
  const std::vector<int> weights = {
      0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1, 1, 4, 4};
  const auto cut_labels = ComputeNormalizedMinGraphCut(edges, weights, 2);
  EXPECT_EQ(cut_labels.size(), 8);
  size_t num_labels[2] = {0};
  for (const auto& label : cut_labels) {
    EXPECT_GE(label.second, 0);
    EXPECT_LT(label.second, 2);
    num_labels[label.second] += 1;
  }
  EXPECT_GT(num_labels[0], 0);
  EXPECT_GT(num_labels[1], 0);
}

TEST(GraphCut, ComputeNormalizedMinGraphCutMissingVertex) {
  const std::vector<std::pair<int, int>> edges = {{3, 4},
                                                  {3, 6},
                                                  {3, 5},
                                                  {0, 1},
                                                  {0, 6},
                                                  {0, 7},
                                                  {0, 5},
                                                  {0, 2},
                                                  {4, 1},
                                                  {1, 6},
                                                  {1, 5},
                                                  {6, 7},
                                                  {7, 5},
                                                  {5, 2}};
  const std::vector<int> weights = {0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1};
  const auto cut_labels = ComputeNormalizedMinGraphCut(edges, weights, 2);
  EXPECT_EQ(cut_labels.size(), 8);
  size_t num_labels[2] = {0};
  for (const auto& label : cut_labels) {
    EXPECT_GE(label.second, 0);
    EXPECT_LT(label.second, 2);
    num_labels[label.second] += 1;
  }
  EXPECT_GT(num_labels[0], 0);
  EXPECT_GT(num_labels[1], 0);
}

TEST(GraphCut, ComputeNormalizedMinGraphCutDisconnected) {
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {3, 4}};
  const std::vector<int> weights = {1, 3, 1};
  const auto cut_labels = ComputeNormalizedMinGraphCut(edges, weights, 2);
  EXPECT_EQ(cut_labels.size(), 5);
  EXPECT_EQ(cut_labels.at(0), cut_labels.at(1));
  EXPECT_EQ(cut_labels.at(1), cut_labels.at(2));
  EXPECT_NE(cut_labels.at(2), cut_labels.at(3));
  EXPECT_EQ(cut_labels.at(3), cut_labels.at(4));
}

TEST(GraphCut, MinSTGraphCut1) {
  MinSTGraphCut<int, int> graph(2);
  EXPECT_EQ(graph.NumNodes(), 2);
  EXPECT_EQ(graph.NumEdges(), 0);
  graph.AddNode(0, 5, 1);
  graph.AddNode(1, 2, 6);
  graph.AddEdge(0, 1, 3, 4);
  EXPECT_EQ(graph.NumEdges(), 10);
  EXPECT_EQ(graph.Compute(), 6);
  EXPECT_TRUE(graph.IsConnectedToSource(0));
  EXPECT_TRUE(graph.IsConnectedToSink(1));
}

TEST(GraphCut, MinSTGraphCut2) {
  MinSTGraphCut<int, int> graph(2);
  graph.AddNode(0, 1, 5);
  graph.AddNode(1, 2, 6);
  graph.AddEdge(0, 1, 3, 4);
  EXPECT_EQ(graph.NumEdges(), 10);
  EXPECT_EQ(graph.Compute(), 3);
  EXPECT_TRUE(graph.IsConnectedToSink(0));
  EXPECT_TRUE(graph.IsConnectedToSink(1));
}

TEST(GraphCut, MinSTGraphCut3) {
  MinSTGraphCut<int, int> graph(3);
  graph.AddNode(0, 6, 4);
  graph.AddNode(2, 3, 6);
  graph.AddEdge(0, 1, 2, 4);
  graph.AddEdge(1, 2, 3, 5);
  EXPECT_EQ(graph.NumEdges(), 12);
  EXPECT_EQ(graph.Compute(), 9);
  EXPECT_TRUE(graph.IsConnectedToSource(0));
  EXPECT_TRUE(graph.IsConnectedToSink(1));
  EXPECT_TRUE(graph.IsConnectedToSink(2));
}

}  // namespace
}  // namespace colmap
