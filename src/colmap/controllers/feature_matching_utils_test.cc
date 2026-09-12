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
#include "colmap/sensor/models.h"
#include "colmap/util/testing.h"

#include <random>

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

// A pair with 40 exact translation inliers among 100 matches (true ratio
// 0.4): with min_inlier_ratio=0.5 the ratio recheck -- not RANSAC failure --
// rejects it, so the estimator returns DEGENERATE *with* inliers.
struct LowRatioPairData {
  TestData data;
  image_t image_id1;
  image_t image_id2;
};

LowRatioPairData CreateLowRatioPairData() {
  LowRatioPairData pair_data;
  TestData& data = pair_data.data;
  data.test_dir = CreateTestDir();
  data.database = Database::Open(data.test_dir / "database.db");

  // Two distinct uncalibrated pinhole cameras -> uncalibrated path.
  Camera camera1 = Camera::CreateFromModelId(
      kInvalidCameraId, SimplePinholeCameraModel::model_id, 1000, 1000, 1000);
  camera1.camera_id = data.database->WriteCamera(camera1);
  Camera camera2 = Camera::CreateFromModelId(
      kInvalidCameraId, SimplePinholeCameraModel::model_id, 1000, 1000, 1000);
  camera2.camera_id = data.database->WriteCamera(camera2);

  Image image1;
  image1.SetName("image1");
  image1.SetCameraId(camera1.camera_id);
  const image_t image_id1 = data.database->WriteImage(image1);
  Image image2;
  image2.SetName("image2");
  image2.SetCameraId(camera2.camera_id);
  const image_t image_id2 = data.database->WriteImage(image2);

  std::mt19937 prng(42);
  std::uniform_real_distribution<float> uniform(100.f, 900.f);
  constexpr int kNumInliers = 40;
  constexpr int kNumMatches = 100;
  FeatureKeypoints keypoints1;
  FeatureKeypoints keypoints2;
  FeatureMatches matches;
  for (int i = 0; i < kNumMatches; ++i) {
    const float x1 = uniform(prng);
    const float y1 = uniform(prng);
    float x2 = x1 + 20.f;
    float y2 = y1 + 10.f;
    if (i >= kNumInliers) {
      x2 = uniform(prng);
      y2 = uniform(prng);
    }
    keypoints1.emplace_back(x1, y1);
    keypoints2.emplace_back(x2, y2);
    matches.emplace_back(i, i);
  }
  data.database->WriteKeypoints(image_id1, keypoints1);
  data.database->WriteKeypoints(image_id2, keypoints2);
  data.database->WriteMatches(image_id1, image_id2, matches);

  data.cache = std::make_shared<FeatureMatcherCache>(100, data.database);
  pair_data.image_id1 = image_id1;
  pair_data.image_id2 = image_id2;
  return pair_data;
}

TwoViewGeometryOptions LowRatioGeometryOptions() {
  TwoViewGeometryOptions options;
  options.min_inlier_ratio = 0.5;
  options.detect_watermark = false;
  options.ransac_options.random_seed = 42;
  options.ransac_options.min_num_trials = 1000;
  return options;
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

  // Geometric verification was skipped: no TVG row is stored at all.
  EXPECT_FALSE(data.database->ExistsTwoViewGeometry(data.image_ids[0],
                                                    data.image_ids[1]));
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

TEST(GeometricVerifierController, VerifyClearsDegenerateGeometry) {
  auto pair_data = CreateLowRatioPairData();
  auto& data = pair_data.data;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  GeometricVerifierController controller(
      verifier_options, LowRatioGeometryOptions(), data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Verify({{pair_data.image_id1, pair_data.image_id2}});

  // The ratio recheck rejects the pair: the DEGENERATE diagnosis is kept
  // but nothing else may be stored.
  const auto tvg = data.database->ReadTwoViewGeometry(pair_data.image_id1,
                                                      pair_data.image_id2);
  EXPECT_EQ(tvg.config, TwoViewGeometry::DEGENERATE);
  EXPECT_TRUE(tvg.inlier_matches.empty());
  EXPECT_FALSE(tvg.F.has_value());
  EXPECT_FALSE(tvg.H.has_value());
  EXPECT_FALSE(tvg.E.has_value());
  // Raw matches are preserved.
  EXPECT_EQ(data.database->ReadMatches(pair_data.image_id1, pair_data.image_id2)
                .size(),
            100);
}

TEST(FeatureMatcherController, MatchClearsDegenerateGeometry) {
  auto pair_data = CreateLowRatioPairData();
  auto& data = pair_data.data;

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  FeatureMatcherController controller(
      matching_options, LowRatioGeometryOptions(), data.cache);
  ASSERT_TRUE(controller.Setup());

  // Matches exist without TVG, so Match takes the verification leg.
  controller.Match({{pair_data.image_id1, pair_data.image_id2}});

  const auto tvg = data.database->ReadTwoViewGeometry(pair_data.image_id1,
                                                      pair_data.image_id2);
  EXPECT_EQ(tvg.config, TwoViewGeometry::DEGENERATE);
  EXPECT_TRUE(tvg.inlier_matches.empty());
  EXPECT_FALSE(tvg.F.has_value());
  EXPECT_FALSE(tvg.H.has_value());
  EXPECT_FALSE(tvg.E.has_value());
  EXPECT_EQ(data.database->ReadMatches(pair_data.image_id1, pair_data.image_id2)
                .size(),
            100);
}

TEST(FeatureMatcherController, MatchSkippedPairsStoreNoTwoViewGeometry) {
  auto pair_data = CreateLowRatioPairData();
  auto& data = pair_data.data;
  // No descriptors and no pre-existing matches: matching is skipped.
  data.database->ClearMatches();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;
  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Match({{pair_data.image_id1, pair_data.image_id2}});

  // Skipped pairs leave no row: absence means never verified.
  EXPECT_FALSE(data.database->ExistsTwoViewGeometry(pair_data.image_id1,
                                                    pair_data.image_id2));
}

TEST(GeometricVerifierController, VerifySkippedPairsStoreNoTwoViewGeometry) {
  auto pair_data = CreateLowRatioPairData();
  auto& data = pair_data.data;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 200;  // Above the 100 stored matches.
  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Verify({{pair_data.image_id1, pair_data.image_id2}});

  // Skipped pairs leave no row: absence means never verified.
  EXPECT_FALSE(data.database->ExistsTwoViewGeometry(pair_data.image_id1,
                                                    pair_data.image_id2));
  // Raw matches are preserved.
  EXPECT_EQ(data.database->ReadMatches(pair_data.image_id1, pair_data.image_id2)
                .size(),
            100);
}

}  // namespace
}  // namespace colmap
