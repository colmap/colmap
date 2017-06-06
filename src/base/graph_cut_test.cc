// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
