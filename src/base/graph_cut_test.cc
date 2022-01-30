// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

#define TEST_NAME "base/graph_cut"
#include "util/testing.h"

#include "base/graph_cut.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestComputeMinGraphCutStoerWagner) {
  const std::vector<std::pair<int, int>> edges = {
      {3, 4}, {3, 6}, {3, 5}, {0, 4}, {0, 1}, {0, 6}, {0, 7}, {0, 5},
      {0, 2}, {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}, {3, 4}};
  const std::vector<int> weights = {0, 3, 1, 3,  1, 2, 6, 1,
                                    8, 1, 1, 80, 2, 1, 1, 4};
  int cut_weight;
  std::vector<char> cut_labels;
  ComputeMinGraphCutStoerWagner(edges, weights, &cut_weight, &cut_labels);
  BOOST_CHECK_EQUAL(cut_weight, 7);
  BOOST_CHECK_EQUAL(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label, 0);
    BOOST_CHECK_LT(label, 2);
  }
}

BOOST_AUTO_TEST_CASE(TestComputeMinGraphCutStoerWagnerDuplicateEdge) {
  const std::vector<std::pair<int, int>> edges = {
      {3, 4}, {3, 6}, {3, 5}, {0, 4}, {0, 1}, {0, 6}, {0, 7}, {0, 5}, {0, 2},
      {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}, {3, 4}, {3, 4}};
  const std::vector<int> weights = {0, 3, 1,  3, 1, 2, 6, 1, 8,
                                    1, 1, 80, 2, 1, 1, 4, 4};
  int cut_weight;
  std::vector<char> cut_labels;
  ComputeMinGraphCutStoerWagner(edges, weights, &cut_weight, &cut_labels);
  BOOST_CHECK_EQUAL(cut_weight, 7);
  BOOST_CHECK_EQUAL(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label, 0);
    BOOST_CHECK_LT(label, 2);
  }
}

BOOST_AUTO_TEST_CASE(TestComputeMinGraphCutStoerWagnerMissingVertex) {
  const std::vector<std::pair<int, int>> edges = {
      {3, 4}, {3, 6}, {3, 5}, {0, 1}, {0, 6}, {0, 7}, {0, 5},
      {0, 2}, {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}};
  const std::vector<int> weights = {0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1};
  int cut_weight;
  std::vector<char> cut_labels;
  ComputeMinGraphCutStoerWagner(edges, weights, &cut_weight, &cut_labels);
  BOOST_CHECK_EQUAL(cut_weight, 2);
  BOOST_CHECK_EQUAL(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label, 0);
    BOOST_CHECK_LT(label, 2);
  }
}

BOOST_AUTO_TEST_CASE(TestComputeMinGraphCutStoerWagnerDisconnected) {
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {3, 4}};
  const std::vector<int> weights = {1, 3, 1};
  int cut_weight;
  std::vector<char> cut_labels;
  ComputeMinGraphCutStoerWagner(edges, weights, &cut_weight, &cut_labels);
  BOOST_CHECK_EQUAL(cut_weight, 0);
  BOOST_CHECK_EQUAL(cut_labels.size(), 5);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label, 0);
    BOOST_CHECK_LT(label, 2);
  }
}

BOOST_AUTO_TEST_CASE(TestComputeNormalizedMinGraphCut) {
  const std::vector<std::pair<int, int>> edges = {
      {3, 4}, {3, 6}, {3, 5}, {0, 4}, {0, 1}, {0, 6}, {0, 7}, {0, 5},
      {0, 2}, {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}, {3, 4}};
  const std::vector<int> weights = {0, 3, 1, 3,  1, 2, 6, 1,
                                    8, 1, 1, 80, 2, 1, 1, 4};
  const auto cut_labels = ComputeNormalizedMinGraphCut(edges, weights, 2);
  BOOST_CHECK_EQUAL(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label.second, 0);
    BOOST_CHECK_LT(label.second, 2);
  }
}

BOOST_AUTO_TEST_CASE(TestComputeNormalizedMinGraphCutDuplicateEdge) {
  const std::vector<std::pair<int, int>> edges = {
      {3, 4}, {3, 6}, {3, 5}, {0, 4}, {0, 1}, {0, 6}, {0, 7}, {0, 5}, {0, 2},
      {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}, {3, 4}, {3, 4}};
  const std::vector<int> weights = {0, 3, 1,  3, 1, 2, 6, 1, 8,
                                    1, 1, 80, 2, 1, 1, 4, 4};
  const auto cut_labels = ComputeNormalizedMinGraphCut(edges, weights, 2);
  BOOST_CHECK_EQUAL(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label.second, 0);
    BOOST_CHECK_LT(label.second, 2);
  }
}

BOOST_AUTO_TEST_CASE(TestComputeNormalizedMinGraphCutMissingVertex) {
  const std::vector<std::pair<int, int>> edges = {
      {3, 4}, {3, 6}, {3, 5}, {0, 1}, {0, 6}, {0, 7}, {0, 5},
      {0, 2}, {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}};
  const std::vector<int> weights = {0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1};
  const auto cut_labels = ComputeNormalizedMinGraphCut(edges, weights, 2);
  BOOST_CHECK_EQUAL(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label.second, 0);
    BOOST_CHECK_LT(label.second, 2);
  }
}

BOOST_AUTO_TEST_CASE(TestComputeNormalizedMinGraphCutDisconnected) {
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {3, 4}};
  const std::vector<int> weights = {1, 3, 1};
  const auto cut_labels = ComputeNormalizedMinGraphCut(edges, weights, 2);
  BOOST_CHECK_EQUAL(cut_labels.size(), 5);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label.second, 0);
    BOOST_CHECK_LT(label.second, 2);
  }
}

BOOST_AUTO_TEST_CASE(TestMinSTGraphCut1) {
  MinSTGraphCut<int, int> graph(2);
  BOOST_CHECK_EQUAL(graph.NumNodes(), 2);
  BOOST_CHECK_EQUAL(graph.NumEdges(), 0);
  graph.AddNode(0, 5, 1);
  graph.AddNode(1, 2, 6);
  graph.AddEdge(0, 1, 3, 4);
  BOOST_CHECK_EQUAL(graph.NumEdges(), 10);
  BOOST_CHECK_EQUAL(graph.Compute(), 6);
  BOOST_CHECK(graph.IsConnectedToSource(0));
  BOOST_CHECK(graph.IsConnectedToSink(1));
}

BOOST_AUTO_TEST_CASE(TestMinSTGraphCut2) {
  MinSTGraphCut<int, int> graph(2);
  graph.AddNode(0, 1, 5);
  graph.AddNode(1, 2, 6);
  graph.AddEdge(0, 1, 3, 4);
  BOOST_CHECK_EQUAL(graph.NumEdges(), 10);
  BOOST_CHECK_EQUAL(graph.Compute(), 3);
  BOOST_CHECK(graph.IsConnectedToSink(0));
  BOOST_CHECK(graph.IsConnectedToSink(1));
}

BOOST_AUTO_TEST_CASE(TestMinSTGraphCut3) {
  MinSTGraphCut<int, int> graph(3);
  graph.AddNode(0, 6, 4);
  graph.AddNode(2, 3, 6);
  graph.AddEdge(0, 1, 2, 4);
  graph.AddEdge(1, 2, 3, 5);
  BOOST_CHECK_EQUAL(graph.NumEdges(), 12);
  BOOST_CHECK_EQUAL(graph.Compute(), 9);
  BOOST_CHECK(graph.IsConnectedToSource(0));
  BOOST_CHECK(graph.IsConnectedToSink(1));
  BOOST_CHECK(graph.IsConnectedToSink(2));
}
