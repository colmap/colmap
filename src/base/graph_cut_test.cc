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
// Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

#define TEST_NAME "base/graph_cut"
#include "util/testing.h"

#include "base/graph_cut.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestComputeMinGraphCut) {
  const std::vector<std::pair<int, int>> edges = {
      {3, 4}, {3, 6}, {3, 5}, {0, 4}, {0, 1}, {0, 6}, {0, 7}, {0, 5},
      {0, 2}, {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}, {3, 4}};
  const std::vector<int> weights = {0, 3, 1, 3,  1, 2, 6, 1,
                                    8, 1, 1, 80, 2, 1, 1, 4};
  int cut_weight;
  std::vector<char> cut_labels;
  ComputeMinGraphCut(edges, weights, &cut_weight, &cut_labels);
  BOOST_CHECK_EQUAL(cut_weight, 7);
  BOOST_CHECK_EQUAL(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label, 0);
    BOOST_CHECK_LT(label, 2);
  }
}

BOOST_AUTO_TEST_CASE(TestComputeMinGraphCutDuplicateEdge) {
  const std::vector<std::pair<int, int>> edges = {
      {3, 4}, {3, 6}, {3, 5}, {0, 4}, {0, 1}, {0, 6}, {0, 7}, {0, 5},
      {0, 2}, {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}, {3, 4}, {3, 4}};
  const std::vector<int> weights = {0, 3, 1, 3,  1, 2, 6, 1,
                                    8, 1, 1, 80, 2, 1, 1, 4, 4};
  int cut_weight;
  std::vector<char> cut_labels;
  ComputeMinGraphCut(edges, weights, &cut_weight, &cut_labels);
  BOOST_CHECK_EQUAL(cut_weight, 7);
  BOOST_CHECK_EQUAL(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label, 0);
    BOOST_CHECK_LT(label, 2);
  }
}

BOOST_AUTO_TEST_CASE(TestComputeMinGraphCutMissingVertex) {
  const std::vector<std::pair<int, int>> edges = {
      {3, 4}, {3, 6}, {3, 5}, {0, 1}, {0, 6}, {0, 7}, {0, 5},
      {0, 2}, {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}};
  const std::vector<int> weights = {0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1};
  int cut_weight;
  std::vector<char> cut_labels;
  ComputeMinGraphCut(edges, weights, &cut_weight, &cut_labels);
  BOOST_CHECK_EQUAL(cut_weight, 2);
  BOOST_CHECK_EQUAL(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label, 0);
    BOOST_CHECK_LT(label, 2);
  }
}

BOOST_AUTO_TEST_CASE(TestComputeMinGraphCutDisconnected) {
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {3, 4}};
  const std::vector<int> weights = {1, 3, 1};
  int cut_weight;
  std::vector<char> cut_labels;
  ComputeMinGraphCut(edges, weights, &cut_weight, &cut_labels);
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
  const auto cut_labels =
      ComputeNormalizedMinGraphCut(edges, weights, 2);
  BOOST_CHECK_EQUAL(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label.second, 0);
    BOOST_CHECK_LT(label.second, 2);
  }
}

BOOST_AUTO_TEST_CASE(TestComputeNormalizedMinGraphCutDuplicateEdge) {
  const std::vector<std::pair<int, int>> edges = {
      {3, 4}, {3, 6}, {3, 5}, {0, 4}, {0, 1}, {0, 6}, {0, 7}, {0, 5},
      {0, 2}, {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}, {3, 4}, {3, 4}};
  const std::vector<int> weights = {0, 3, 1, 3,  1, 2, 6, 1,
                                    8, 1, 1, 80, 2, 1, 1, 4, 4};
  const auto cut_labels =
      ComputeNormalizedMinGraphCut(edges, weights, 2);
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
  const auto cut_labels =
      ComputeNormalizedMinGraphCut(edges, weights, 2);
  BOOST_CHECK_EQUAL(cut_labels.size(), 8);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label.second, 0);
    BOOST_CHECK_LT(label.second, 2);
  }
}

BOOST_AUTO_TEST_CASE(TestComputeNormalizedMinGraphCutDisconnected) {
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {3, 4}};
  const std::vector<int> weights = {1, 3, 1};
  const auto cut_labels =
      ComputeNormalizedMinGraphCut(edges, weights, 2);
  BOOST_CHECK_EQUAL(cut_labels.size(), 5);
  for (const auto& label : cut_labels) {
    BOOST_CHECK_GE(label.second, 0);
    BOOST_CHECK_LT(label.second, 2);
  }
}
