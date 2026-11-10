// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/model.h"

#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(Model, ReadCOLMAP) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 10;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const auto test_dir = CreateTestDir();
  const auto sparse_path = test_dir / "sparse";
  std::filesystem::create_directories(sparse_path);
  reconstruction.WriteBinary(sparse_path);

  Model model;
  model.ReadFromCOLMAP(test_dir);
  EXPECT_EQ(model.images.size(), reconstruction.NumRegImages());
  EXPECT_EQ(model.points.size(), reconstruction.NumPoints3D());

  Model model_lower;
  model_lower.Read(test_dir, "colmap");
  EXPECT_EQ(model_lower.images.size(), reconstruction.NumRegImages());
  EXPECT_EQ(model_lower.points.size(), reconstruction.NumPoints3D());

  // Verify case insensitivity.
  Model model_upper;
  model_upper.Read(test_dir, "COLMAP");
  EXPECT_EQ(model_upper.images.size(), reconstruction.NumRegImages());
  EXPECT_EQ(model_upper.points.size(), reconstruction.NumPoints3D());

  const std::string name = model.GetImageName(0);
  EXPECT_FALSE(name.empty());
  EXPECT_EQ(model.GetImageIdx(name), 0);

  EXPECT_THROW(model.GetImageIdx("nonexistent"), std::exception);
  EXPECT_THROW(model.GetImageName(-1), std::exception);
  EXPECT_THROW(model.GetImageName(model.images.size()), std::exception);
}

TEST(Model, ComputeSharedPoints) {
  Model model;
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  // Projection center = -R^T * T = -T (for R=I).
  const float T1[3] = {0, 0, 0};
  const float T2[3] = {-1, 0, 0};
  const float T3[3] = {-2, 0, 0};
  model.images.emplace_back("img0.jpg", 100, 100, K, R, T1);
  model.images.emplace_back("img1.jpg", 100, 100, K, R, T2);
  model.images.emplace_back("img2.jpg", 100, 100, K, R, T3);
  model.points.emplace_back(Model::Point{5.0f, 0.0f, 10.0f, {0, 1}});

  // Point seen by images 0, 1, and 2.
  model.points.emplace_back(Model::Point{6.0f, 0.0f, 10.0f, {0, 1, 2}});

  const std::vector<std::map<int, int>> shared = model.ComputeSharedPoints();
  EXPECT_EQ(shared.size(), 3);
  EXPECT_EQ(shared[0].at(1), 2);  // images 0,1 share 2 points
  EXPECT_EQ(shared[1].at(0), 2);
  EXPECT_EQ(shared[0].at(2), 1);  // images 0,2 share 1 point
  EXPECT_EQ(shared[2].at(0), 1);
  EXPECT_EQ(shared[1].at(2), 1);  // images 1,2 share 1 point
  EXPECT_EQ(shared[2].at(1), 1);
}

TEST(Model, ComputeDepthRanges) {
  Model model;
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  model.images.emplace_back("img0.jpg", 100, 100, K, R, T);

  for (int i = 1; i <= 100; ++i) {
    model.points.emplace_back(
        Model::Point{0.0f, 0.0f, static_cast<float>(i), {0}});
  }

  const std::vector<std::pair<float, float>> depth_ranges =
      model.ComputeDepthRanges();
  EXPECT_EQ(depth_ranges.size(), 1);

  // The points span depths in the range [1..100] and
  // the range is computed at percentiles 1% and 99%
  // with some additional padding.
  EXPECT_FLOAT_EQ(depth_ranges[0].first, 1.5f);
  EXPECT_FLOAT_EQ(depth_ranges[0].second, 125.0f);
}

TEST(Model, ComputeDepthRangesNoPoints) {
  Model model;
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  model.images.emplace_back("img0.jpg", 100, 100, K, R, T);

  const std::vector<std::pair<float, float>> depth_ranges =
      model.ComputeDepthRanges();
  EXPECT_EQ(depth_ranges.size(), 1);
  EXPECT_FLOAT_EQ(depth_ranges[0].first, -1.0f);
  EXPECT_FLOAT_EQ(depth_ranges[0].second, -1.0f);
}

TEST(Model, ComputeTriangulationAngles) {
  Model model;
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  // Projection center = -R^T * T = -T (for R=I).
  // Place cameras at (0,0,0) and (1,0,0).
  const float T1[3] = {0, 0, 0};
  const float T2[3] = {-1, 0, 0};
  model.images.emplace_back("img0.jpg", 100, 100, K, R, T1);
  model.images.emplace_back("img1.jpg", 100, 100, K, R, T2);

  // Point at midpoint between cameras, at depth 10.
  model.points.emplace_back(Model::Point{0.5f, 0.0f, 10.0f, {0, 1}});

  const std::vector<std::map<int, float>> angles =
      model.ComputeTriangulationAngles(50);
  EXPECT_EQ(angles.size(), 2);
  const float expected_angle = 2.0f * std::atan(0.5f / 10.0f);
  EXPECT_NEAR(angles[0].at(1), expected_angle, 1e-5f);
  // Angles should be symmetric.
  EXPECT_FLOAT_EQ(angles[0].at(1), angles[1].at(0));
}

TEST(Model, GetMaxOverlappingImages) {
  Model model;
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  // Projection center = -R^T * T = -T (for R=I).
  const float T1[3] = {0, 0, 0};
  const float T2[3] = {-1, 0, 0};
  const float T3[3] = {-2, 0, 0};
  model.images.emplace_back("img0.jpg", 100, 100, K, R, T1);
  model.images.emplace_back("img1.jpg", 100, 100, K, R, T2);
  model.images.emplace_back("img2.jpg", 100, 100, K, R, T3);
  for (int i = 0; i < 10; ++i) {
    model.points.emplace_back(Model::Point{0.5f, 0.0f, 5.0f + i, {0, 1}});
  }
  model.points.emplace_back(Model::Point{1.0f, 0.0f, 10.0f, {0, 2}});

  const std::vector<std::vector<int>> overlapping =
      model.GetMaxOverlappingImages(2, 0.0);
  EXPECT_EQ(overlapping.size(), 3);
  // Image 0 should have image 1 as top overlap.
  EXPECT_FALSE(overlapping[0].empty());
  EXPECT_EQ(overlapping[0][0], 1);
}

std::pair<float, float> SortedDepthRange(std::vector<float> depths) {
  if (depths.empty()) {
    return {-1.0f, -1.0f};
  }
  std::sort(depths.begin(), depths.end());
  return {depths[static_cast<size_t>(depths.size() * 0.01f)] * 0.75f,
          depths[static_cast<size_t>(depths.size() * 0.99f)] * 1.25f};
}

TEST(Model, ComputeDepthRangesRanksAndTies) {
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  std::mt19937 rng(0);
  for (const size_t num_points : {0, 1, 2, 99, 100, 101, 1024, 10001}) {
    for (const bool tied : {false, true}) {
      Model model;
      model.images.emplace_back("img.jpg", 100, 100, K, R, T);
      std::vector<float> depths;
      for (size_t i = 0; i < num_points; ++i) {
        const float depth = tied ? 7.0f : static_cast<float>(i + 1);
        depths.push_back(depth);
        model.points.push_back(Model::Point{0, 0, depth, {0}});
      }
      // These observations must not change the population used for the ranks.
      model.points.push_back(Model::Point{0, 0, 0, {0}});
      model.points.push_back(Model::Point{0, 0, -100, {0}});
      const auto expected = SortedDepthRange(depths);
      for (int permutation = 0; permutation < 3; ++permutation) {
        SCOPED_TRACE(::testing::Message()
                     << "num_points=" << num_points << " tied=" << tied
                     << " permutation=" << permutation);
        const auto ranges = model.ComputeDepthRanges();
        ASSERT_EQ(ranges.size(), 1);
        EXPECT_EQ(ranges[0], expected);
        if (permutation == 0) {
          std::reverse(model.points.begin(), model.points.end());
        } else {
          std::shuffle(model.points.begin(), model.points.end(), rng);
        }
      }
    }
  }
}

TEST(Model, ComputeDepthRangesAcrossPosesAndTracks) {
  Model model;
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R0[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float R1[9] = {0, 0, 1, 0, 1, 0, -1, 0, 0};
  const float R2[9] = {1, 0, 0, 0, 0, -1, 0, 1, 0};
  const float T0[3] = {0, 0, 0};
  const float T1[3] = {0, 0, 3};
  const float T2[3] = {0, 0, 2};
  model.images.emplace_back("img0.jpg", 100, 100, K, R0, T0);
  model.images.emplace_back("img1.jpg", 100, 100, K, R1, T1);
  model.images.emplace_back("img2.jpg", 100, 100, K, R2, T2);
  model.images.emplace_back("unobserved.jpg", 100, 100, K, R0, T0);
  std::vector<std::vector<float>> depths(4);
  for (int i = 0; i < 257; ++i) {
    Model::Point point{static_cast<float>(i % 41 - 20),
                       static_cast<float>(i % 23 - 11),
                       static_cast<float>(i % 31 - 15),
                       {0}};
    if (i % 5 != 0) {
      point.track.push_back(1);
    }
    if (i % 7 != 0) {
      point.track.push_back(2);
    }
    // These axis-aligned rotations give independent, exact expected depths.
    const float camera_depths[] = {point.z, 3.0f - point.x, point.y + 2.0f};
    for (const int image_idx : point.track) {
      if (camera_depths[image_idx] > 0) {
        depths[image_idx].push_back(camera_depths[image_idx]);
      }
    }
    model.points.push_back(point);
  }
  std::vector<std::pair<float, float>> expected;
  expected.reserve(depths.size());
  for (const auto& image_depths : depths) {
    expected.push_back(SortedDepthRange(image_depths));
  }
  EXPECT_EQ(model.ComputeDepthRanges(), expected);
  std::reverse(model.points.begin(), model.points.end());
  EXPECT_EQ(model.ComputeDepthRanges(), expected);
}

TEST(Model, ComputeDepthRangesMixedDepthsAndInvalidTracks) {
  Model model;
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  model.images.emplace_back("img.jpg", 100, 100, K, R, T);
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (const float depth : {0.0f, -0.0f, -1.0f, -infinity, nan}) {
    model.points.push_back(Model::Point{0, 0, depth, {0}});
  }
  const std::vector<std::pair<float, float>> empty_ranges = {{-1, -1}};
  EXPECT_EQ(model.ComputeDepthRanges(), empty_ranges);

  const std::vector<float> positive_depths = {
      0.125f, 0.5f, 1.0f, 2.0f, 4.0f, 16.0f, 1e-20f, 1e30f, 1.0f};
  std::vector<float> expected_depths;
  for (size_t i = 0; i < 257; ++i) {
    const float depth = positive_depths[i % positive_depths.size()];
    Model::Point point{0, 0, depth, {0}};
    expected_depths.push_back(depth);
    // The owning method counts duplicate track indices as observations.
    if (i % 7 == 0) {
      point.track.push_back(0);
      expected_depths.push_back(depth);
    }
    model.points.push_back(std::move(point));
  }
  for (int i = 0; i < 8; ++i) {
    model.points.push_back(Model::Point{0, 0, infinity, {0}});
    expected_depths.push_back(infinity);
  }
  const std::vector<std::pair<float, float>> expected = {
      SortedDepthRange(expected_depths)};
  EXPECT_EQ(model.ComputeDepthRanges(), expected);
  std::reverse(model.points.begin(), model.points.end());
  EXPECT_EQ(model.ComputeDepthRanges(), expected);

  // A repeated nearest observation changes the 1% rank from depth 2 to 1.
  model.points.clear();
  expected_depths.clear();
  for (int depth = 1; depth <= 100; ++depth) {
    Model::Point point{0, 0, static_cast<float>(depth), {0}};
    expected_depths.push_back(point.z);
    if (depth == 1) {
      point.track.push_back(0);
      expected_depths.push_back(point.z);
    }
    model.points.push_back(std::move(point));
  }
  const std::vector<std::pair<float, float>> weighted_expected = {
      SortedDepthRange(expected_depths)};
  ASSERT_EQ(weighted_expected[0].first, 0.75f);
  EXPECT_EQ(model.ComputeDepthRanges(), weighted_expected);

  // Lookup still rejects invalid indices before excluding negative depths.
  for (const int image_idx : {-1, 1}) {
    model.points.push_back(Model::Point{0, 0, -1, {image_idx}});
    EXPECT_THROW(model.ComputeDepthRanges(), std::out_of_range);
    model.points.pop_back();
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
