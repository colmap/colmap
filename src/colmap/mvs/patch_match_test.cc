// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_internal.h"

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace internal {
namespace {

TEST(SelectPatchMatchSources, ConfiguredOrderAndFirstOccurrence) {
  const std::vector<int> configured = {3, 1, 3, 0, 4, 2, 1};
  const std::vector<int> expected = {3, 1, 4, 2};
  for (const std::vector<int>& available :
       {std::vector<int>{0, 1, 2, 3, 4}, std::vector<int>{4, 3, 2, 1, 0}}) {
    EXPECT_EQ(SelectPatchMatchSources(
                  0,
                  configured,
                  FlatHashSet<int>(available.begin(), available.end())),
              expected);
    EXPECT_EQ(SelectPatchMatchSources(
                  0,
                  {2, 4, 1, 3},
                  FlatHashSet<int>(available.begin(), available.end())),
              (std::vector<int>{2, 4, 1, 3}));
  }
  EXPECT_EQ(configured, (std::vector<int>{3, 1, 3, 0, 4, 2, 1}));
}

TEST(SelectPatchMatchSources, MissingSourcesAndRepeatedSelection) {
  const std::vector<int> configured = {3, 1, 3, 0, 4, 2, 1};
  const auto selected =
      SelectPatchMatchSources(0, configured, {0, 1, 2, 3, 99});
  EXPECT_EQ(selected, (std::vector<int>{3, 1, 2}));
  EXPECT_EQ(SelectPatchMatchSources(0, selected, {0, 1, 2, 3, 99}), selected);
  EXPECT_EQ(SelectPatchMatchSources(0, selected, {0, 1, 3}),
            (std::vector<int>{3, 1}));
}

TEST(SelectPatchMatchSources, EmptyAndReferenceOnly) {
  EXPECT_TRUE(SelectPatchMatchSources(0, {}, {0, 1}).empty());
  EXPECT_TRUE(SelectPatchMatchSources(0, {0, 0}, {0}).empty());
  EXPECT_TRUE(SelectPatchMatchSources(0, {1, 2}, {}).empty());
  EXPECT_EQ(SelectPatchMatchSources(0, {2, 0, 1, 2}, {1, 2}),
            (std::vector<int>{2, 1}));
}

}  // namespace
}  // namespace internal
}  // namespace mvs
}  // namespace colmap
