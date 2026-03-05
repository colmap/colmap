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

#include "colmap/controllers/feature_matching_utils.h"

#include "colmap/controllers/matcher_cache.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

struct TestData {
  std::filesystem::path test_dir;
  std::shared_ptr<Database> database;
  std::shared_ptr<FeatureMatcherCache> cache;
  std::vector<image_t> image_ids;
};

TestData CreateTestData(int num_images) {
  TestData data;
  data.test_dir = CreateTestDir();
  const auto database_path = data.test_dir / "database.db";
  data.database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = num_images;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 1;
  options.num_points3D = 20;
  options.num_points2D_without_point3D = 3;
  SynthesizeDataset(options, &reconstruction, data.database.get());

  data.cache = std::make_shared<FeatureMatcherCache>(100, data.database);
  data.image_ids = data.cache->GetImageIds();
  return data;
}

FeatureMatchingOptions DefaultMatchingOptions() {
  FeatureMatchingOptions options;
  options.use_gpu = false;
  options.num_threads = 1;
  return options;
}

std::vector<std::pair<image_t, image_t>> AllPairs(
    const std::vector<image_t>& image_ids) {
  std::vector<std::pair<image_t, image_t>> pairs;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    for (size_t j = i + 1; j < image_ids.size(); ++j) {
      pairs.emplace_back(image_ids[i], image_ids[j]);
    }
  }
  return pairs;
}

// Match pairs without geometric verification, then clear TVGs.
// Leaves matches in the database ready for a GeometricVerifierController.
void MatchPairsWithoutVerification(
    TestData& data, const std::vector<std::pair<image_t, image_t>>& pairs) {
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();
  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  matching_options.skip_geometric_verification = true;
  TwoViewGeometryOptions geometry_options;
  FeatureMatcherController matcher(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(matcher.Setup());
  matcher.Match(pairs);
  data.database->ClearTwoViewGeometries();
}

TEST(FeatureMatcherController, MatchEmptyPairs) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // Matching empty pairs should return without error
  controller.Match({});

  EXPECT_EQ(data.database->ReadAllMatches().size(), 0);
}

TEST(FeatureMatcherController, MatchSkipsSelfMatches) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // Self-match pairs should be skipped
  std::vector<std::pair<image_t, image_t>> pairs;
  pairs.reserve(data.image_ids.size());
  for (const auto id : data.image_ids) {
    pairs.emplace_back(id, id);
  }
  controller.Match(pairs);

  EXPECT_EQ(data.database->ReadAllMatches().size(), 0);
}

TEST(FeatureMatcherController, MatchSkipsDuplicatePairs) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 2);
  const image_t id1 = data.image_ids[0];
  const image_t id2 = data.image_ids[1];

  // Submit same pair multiple times — should only process once
  controller.Match({{id1, id2}, {id1, id2}, {id1, id2}});

  const auto matches = data.database->ReadAllMatches();
  EXPECT_EQ(matches.size(), 1);
}

TEST(FeatureMatcherController, MatchSkipsExistingResults) {
  auto data = CreateTestData(3);

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 2);
  const image_t id1 = data.image_ids[0];
  const image_t id2 = data.image_ids[1];

  // Clear and match once
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();
  controller.Match({{id1, id2}});

  const auto matches_before = data.database->ReadAllMatches();
  const auto tvg_before = data.database->ReadTwoViewGeometries();
  EXPECT_EQ(matches_before.size(), 1);
  EXPECT_EQ(tvg_before.size(), 1);

  // Match same pair again — should skip since both matches and TVG exist
  controller.Match({{id1, id2}});

  const auto matches_after = data.database->ReadAllMatches();
  const auto tvg_after = data.database->ReadTwoViewGeometries();
  EXPECT_EQ(matches_after.size(), matches_before.size());
  EXPECT_EQ(tvg_after.size(), tvg_before.size());

  // Match with reversed pair — should also be skipped
  controller.Match({{id2, id1}});

  const auto matches_reversed = data.database->ReadAllMatches();
  const auto tvg_reversed = data.database->ReadTwoViewGeometries();
  EXPECT_EQ(matches_reversed.size(), matches_before.size());
  EXPECT_EQ(tvg_reversed.size(), tvg_before.size());
}

TEST(FeatureMatcherController, MatchMultiplePairs) {
  auto data = CreateTestData(4);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // Match all pairs
  const auto pairs = AllPairs(data.image_ids);
  controller.Match(pairs);

  // 4 choose 2 = 6 pairs
  EXPECT_EQ(data.database->ReadAllMatches().size(), 6);
  EXPECT_EQ(data.database->ReadTwoViewGeometries().size(), 6);
}

TEST(FeatureMatcherController, MatchSkipGeometricVerification) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  matching_options.skip_geometric_verification = true;
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 2);
  controller.Match({{data.image_ids[0], data.image_ids[1]}});

  // Matches should be written even without geometric verification
  EXPECT_EQ(data.database->ReadAllMatches().size(), 1);

  // Verify geometric verification was skipped: TVG should have UNDEFINED config
  const auto tvg =
      data.database->ReadTwoViewGeometry(data.image_ids[0], data.image_ids[1]);
  EXPECT_EQ(tvg.config, TwoViewGeometry::UNDEFINED);
  EXPECT_TRUE(tvg.inlier_matches.empty());
}

TEST(GeometricVerifierController, OptionsAccessor) {
  auto data = CreateTestData(3);

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);

  EXPECT_EQ(controller.Options().num_threads, 1);
  controller.Options().num_threads = 2;
  EXPECT_EQ(controller.Options().num_threads, 2);
}

TEST(GeometricVerifierController, VerifyEmptyPairs) {
  auto data = CreateTestData(3);
  data.database->ClearTwoViewGeometries();

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // Verifying empty pairs should return without error
  controller.Verify({});

  EXPECT_EQ(data.database->ReadTwoViewGeometries().size(), 0);
}

TEST(GeometricVerifierController, VerifySkipsSelfMatches) {
  auto data = CreateTestData(3);
  data.database->ClearTwoViewGeometries();

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  std::vector<std::pair<image_t, image_t>> pairs;
  pairs.reserve(data.image_ids.size());
  for (const auto id : data.image_ids) {
    pairs.emplace_back(id, id);
  }
  controller.Verify(pairs);

  EXPECT_EQ(data.database->ReadTwoViewGeometries().size(), 0);
}

TEST(GeometricVerifierController, VerifySkipsDuplicatePairs) {
  auto data = CreateTestData(3);
  ASSERT_GE(data.image_ids.size(), 2);
  MatchPairsWithoutVerification(data, {{data.image_ids[0], data.image_ids[1]}});

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  const image_t id1 = data.image_ids[0];
  const image_t id2 = data.image_ids[1];

  // Submit same pair multiple times — should only process once
  controller.Verify({{id1, id2}, {id1, id2}, {id1, id2}});

  const auto tvgs = data.database->ReadTwoViewGeometries();
  EXPECT_EQ(tvgs.size(), 1);
}

TEST(GeometricVerifierController, VerifyWithExistingMatches) {
  auto data = CreateTestData(4);
  const auto pairs = AllPairs(data.image_ids);
  MatchPairsWithoutVerification(data, pairs);

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Verify(pairs);

  // All 6 pairs should now have TVGs
  EXPECT_EQ(data.database->ReadTwoViewGeometries().size(), 6);
}

}  // namespace
}  // namespace colmap
