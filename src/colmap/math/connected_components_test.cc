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

#include "colmap/math/connected_components.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

using ::testing::UnorderedElementsAre;

TEST(FindConnectedComponents, Empty) {
  std::unordered_set<int> nodes;
  std::vector<std::pair<int, int>> edges;
  auto components = FindConnectedComponents(nodes, edges);
  EXPECT_TRUE(components.empty());
}

TEST(FindConnectedComponents, SingleNode) {
  std::unordered_set<int> nodes = {1};
  std::vector<std::pair<int, int>> edges;
  auto components = FindConnectedComponents(nodes, edges);
  ASSERT_EQ(components.size(), 1);
  EXPECT_EQ(components[0].size(), 1);
  EXPECT_EQ(components[0][0], 1);
}

TEST(FindConnectedComponents, TwoConnectedNodes) {
  std::unordered_set<int> nodes = {1, 2};
  std::vector<std::pair<int, int>> edges = {{1, 2}};
  auto components = FindConnectedComponents(nodes, edges);
  ASSERT_EQ(components.size(), 1);
  EXPECT_EQ(components[0].size(), 2);
}

TEST(FindConnectedComponents, TwoDisconnectedNodes) {
  std::unordered_set<int> nodes = {1, 2};
  std::vector<std::pair<int, int>> edges;
  auto components = FindConnectedComponents(nodes, edges);
  ASSERT_EQ(components.size(), 2);
  EXPECT_EQ(components[0].size(), 1);
  EXPECT_EQ(components[1].size(), 1);
}

TEST(FindConnectedComponents, ThreeComponents) {
  std::unordered_set<int> nodes = {1, 2, 3, 4, 5, 6};
  std::vector<std::pair<int, int>> edges = {{1, 2}, {3, 4}, {5, 6}};
  auto components = FindConnectedComponents(nodes, edges);
  ASSERT_EQ(components.size(), 3);
  for (const auto& comp : components) {
    EXPECT_EQ(comp.size(), 2);
  }
}

TEST(FindConnectedComponents, Chain) {
  std::unordered_set<int> nodes = {1, 2, 3, 4, 5};
  std::vector<std::pair<int, int>> edges = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};
  auto components = FindConnectedComponents(nodes, edges);
  ASSERT_EQ(components.size(), 1);
  EXPECT_EQ(components[0].size(), 5);
}

TEST(FindLargestConnectedComponent, Empty) {
  std::unordered_set<int> nodes;
  std::vector<std::pair<int, int>> edges;
  auto largest = FindLargestConnectedComponent(nodes, edges);
  EXPECT_TRUE(largest.empty());
}

TEST(FindLargestConnectedComponent, SingleNode) {
  std::unordered_set<int> nodes = {42};
  std::vector<std::pair<int, int>> edges;
  auto largest = FindLargestConnectedComponent(nodes, edges);
  EXPECT_THAT(largest, UnorderedElementsAre(42));
}

TEST(FindLargestConnectedComponent, AllConnected) {
  std::unordered_set<int> nodes = {1, 2, 3, 4};
  std::vector<std::pair<int, int>> edges = {{1, 2}, {2, 3}, {3, 4}};
  auto largest = FindLargestConnectedComponent(nodes, edges);
  EXPECT_THAT(largest, UnorderedElementsAre(1, 2, 3, 4));
}

TEST(FindLargestConnectedComponent, TwoComponentsDifferentSizes) {
  // Component 1: {1, 2, 3} (size 3)
  // Component 2: {10, 20} (size 2)
  std::unordered_set<int> nodes = {1, 2, 3, 10, 20};
  std::vector<std::pair<int, int>> edges = {{1, 2}, {2, 3}, {10, 20}};
  auto largest = FindLargestConnectedComponent(nodes, edges);
  EXPECT_THAT(largest, UnorderedElementsAre(1, 2, 3));
}

TEST(FindLargestConnectedComponent, ManySmallOnelarger) {
  // 5 isolated nodes + 1 component of 3
  std::unordered_set<int> nodes = {1, 2, 3, 4, 5, 100, 200, 300};
  std::vector<std::pair<int, int>> edges = {{100, 200}, {200, 300}};
  auto largest = FindLargestConnectedComponent(nodes, edges);
  EXPECT_THAT(largest, UnorderedElementsAre(100, 200, 300));
}

TEST(FindLargestConnectedComponent, StringType) {
  std::unordered_set<std::string> nodes = {"a", "b", "c", "x", "y"};
  std::vector<std::pair<std::string, std::string>> edges = {{"a", "b"},
                                                            {"b", "c"}};
  auto largest = FindLargestConnectedComponent(nodes, edges);
  EXPECT_THAT(largest, UnorderedElementsAre("a", "b", "c"));
}

}  // namespace
}  // namespace colmap
