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

#include "colmap/controllers/matcher_cache.h"

#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

struct TestData {
  std::shared_ptr<Database> database;
  Reconstruction reconstruction;
};

TestData CreateTestData(int num_images,
                        bool with_priors = false,
                        int num_cameras_per_rig = 1) {
  TestData data;
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  data.database = Database::Open(database_path);

  SyntheticDatasetOptions options;
  options.num_rigs = num_images / num_cameras_per_rig;
  options.num_cameras_per_rig = num_cameras_per_rig;
  options.num_frames_per_rig = 1;
  options.num_points3D = 20;
  options.num_points2D_without_point3D = 3;
  options.prior_position = with_priors;
  SynthesizeDataset(options, &data.reconstruction, data.database.get());

  return data;
}

TEST(FeatureMatcherCache, GetCamera) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const std::vector<Camera> cameras = data.database->ReadAllCameras();
  ASSERT_FALSE(cameras.empty());

  for (const Camera& camera : cameras) {
    EXPECT_EQ(cache.GetCamera(camera.camera_id), camera);
  }
}

TEST(FeatureMatcherCache, GetFrame) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const std::vector<Frame> frames = data.database->ReadAllFrames();
  ASSERT_FALSE(frames.empty());

  for (const Frame& frame : frames) {
    EXPECT_EQ(cache.GetFrame(frame.FrameId()), frame);
  }
}

TEST(FeatureMatcherCache, GetImage) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const std::vector<Image> images = data.database->ReadAllImages();
  ASSERT_FALSE(images.empty());

  for (const Image& image : images) {
    EXPECT_EQ(cache.GetImage(image.ImageId()), image);
  }
}

TEST(FeatureMatcherCache, GetImageIds) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const std::vector<Image> images = data.database->ReadAllImages();
  std::vector<image_t> expected_ids;
  expected_ids.reserve(images.size());
  for (const Image& image : images) {
    expected_ids.push_back(image.ImageId());
  }
  EXPECT_THAT(cache.GetImageIds(),
              ::testing::UnorderedElementsAreArray(expected_ids));
}

TEST(FeatureMatcherCache, GetFrameIds) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const std::vector<Frame> frames = data.database->ReadAllFrames();
  std::vector<frame_t> expected_ids;
  expected_ids.reserve(frames.size());
  for (const Frame& frame : frames) {
    expected_ids.push_back(frame.FrameId());
  }
  EXPECT_THAT(cache.GetFrameIds(),
              ::testing::UnorderedElementsAreArray(expected_ids));
}

TEST(FeatureMatcherCache, FindImagePosePriorOrNullWithPriors) {
  auto data = CreateTestData(4, /*with_priors=*/true);
  FeatureMatcherCache cache(5, data.database);

  const std::vector<Image> images = data.database->ReadAllImages();
  ASSERT_FALSE(images.empty());

  for (const Image& image : images) {
    const PosePrior* prior = cache.FindImagePosePriorOrNull(image.ImageId());
    ASSERT_NE(prior, nullptr);
    EXPECT_TRUE(prior->HasPosition());
  }
}

TEST(FeatureMatcherCache, FindImagePosePriorOrNullWithoutPriors) {
  auto data = CreateTestData(4, /*with_priors=*/false);
  FeatureMatcherCache cache(5, data.database);

  const std::vector<Image> images = data.database->ReadAllImages();
  ASSERT_FALSE(images.empty());

  for (const Image& image : images) {
    const PosePrior* prior = cache.FindImagePosePriorOrNull(image.ImageId());
    EXPECT_EQ(prior, nullptr);
  }
}

TEST(FeatureMatcherCache, Features) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const std::vector<Image> images = data.database->ReadAllImages();
  ASSERT_FALSE(images.empty());

  for (const Image& image : images) {
    EXPECT_TRUE(cache.ExistsKeypoints(image.ImageId()));
    EXPECT_TRUE(cache.ExistsDescriptors(image.ImageId()));
    EXPECT_EQ(*cache.GetKeypoints(image.ImageId()),
              data.database->ReadKeypoints(image.ImageId()));
    auto cached_descriptors = cache.GetDescriptors(image.ImageId());
    FeatureDescriptors db_descriptors =
        data.database->ReadDescriptors(image.ImageId());
    EXPECT_EQ(cached_descriptors->type, db_descriptors.type);
    EXPECT_EQ(cached_descriptors->data, db_descriptors.data);
  }
}

TEST(FeatureMatcherCache, Matches) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const auto all_matches = data.database->ReadAllMatches();
  ASSERT_FALSE(all_matches.empty());

  for (const auto& [pair_id, matches] : all_matches) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    EXPECT_TRUE(cache.ExistsMatches(image_id1, image_id2));
    EXPECT_EQ(cache.GetMatches(image_id1, image_id2), matches);
  }
}

TEST(FeatureMatcherCache, TwoViewGeometry) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const auto all_tvg = data.database->ReadTwoViewGeometries();
  ASSERT_FALSE(all_tvg.empty());

  for (const auto& [pair_id, tvg] : all_tvg) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    EXPECT_TRUE(cache.ExistsTwoViewGeometry(image_id1, image_id2));
    EXPECT_EQ(cache.ExistsInlierMatches(image_id1, image_id2),
              !tvg.inlier_matches.empty());
    EXPECT_EQ(cache.GetTwoViewGeometry(image_id1, image_id2).inlier_matches,
              tvg.inlier_matches);
  }
}

TEST(FeatureMatcherCache, WriteAndGetMatches) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const std::vector<Image> images = data.database->ReadAllImages();
  ASSERT_GE(images.size(), 2);

  const image_t id1 = images[0].ImageId();
  const image_t id2 = images[1].ImageId();

  // Delete existing matches first.
  cache.DeleteMatches(id1, id2);
  EXPECT_FALSE(cache.ExistsMatches(id1, id2));

  // Write new matches.
  FeatureMatches matches(5);
  for (size_t i = 0; i < matches.size(); ++i) {
    matches[i].point2D_idx1 = i;
    matches[i].point2D_idx2 = i;
  }
  cache.WriteMatches(id1, id2, matches);
  EXPECT_TRUE(cache.ExistsMatches(id1, id2));

  FeatureMatches read_matches = cache.GetMatches(id1, id2);
  EXPECT_EQ(read_matches.size(), 5);
}

TEST(FeatureMatcherCache, WriteAndGetTwoViewGeometry) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const std::vector<Image> images = data.database->ReadAllImages();
  ASSERT_GE(images.size(), 2);

  const image_t id1 = images[0].ImageId();
  const image_t id2 = images[1].ImageId();

  // Delete existing two-view geometry first.
  cache.DeleteTwoViewGeometry(id1, id2);
  EXPECT_FALSE(cache.ExistsTwoViewGeometry(id1, id2));

  // Write new two-view geometry.
  TwoViewGeometry tvg;
  tvg.config = TwoViewGeometry::CALIBRATED;
  tvg.inlier_matches.resize(10);
  cache.WriteTwoViewGeometry(id1, id2, tvg);
  EXPECT_TRUE(cache.ExistsTwoViewGeometry(id1, id2));

  TwoViewGeometry read_tvg = cache.GetTwoViewGeometry(id1, id2);
  EXPECT_EQ(read_tvg.config, TwoViewGeometry::CALIBRATED);
  EXPECT_EQ(read_tvg.inlier_matches.size(), 10);
}

TEST(FeatureMatcherCache, UpdateTwoViewGeometry) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const auto all_tvg = data.database->ReadTwoViewGeometries();
  ASSERT_FALSE(all_tvg.empty());

  const auto& [pair_id, original_tvg] = all_tvg.front();
  const auto [id1, id2] = PairIdToImagePair(pair_id);

  TwoViewGeometry updated_tvg;
  updated_tvg.config = TwoViewGeometry::UNCALIBRATED;
  updated_tvg.inlier_matches.resize(7);
  cache.UpdateTwoViewGeometry(id1, id2, updated_tvg);

  TwoViewGeometry read_tvg = cache.GetTwoViewGeometry(id1, id2);
  EXPECT_EQ(read_tvg.config, TwoViewGeometry::UNCALIBRATED);
  EXPECT_EQ(read_tvg.inlier_matches.size(), 7);
}

TEST(FeatureMatcherCache, DeleteMatches) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const auto all_matches = data.database->ReadAllMatches();
  ASSERT_FALSE(all_matches.empty());

  const auto& [pair_id, _] = all_matches.front();
  const auto [id1, id2] = PairIdToImagePair(pair_id);

  EXPECT_TRUE(cache.ExistsMatches(id1, id2));
  cache.DeleteMatches(id1, id2);
  EXPECT_FALSE(cache.ExistsMatches(id1, id2));
}

TEST(FeatureMatcherCache, DeleteTwoViewGeometry) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const auto all_tvg = data.database->ReadTwoViewGeometries();
  ASSERT_FALSE(all_tvg.empty());

  const auto& [pair_id, _] = all_tvg.front();
  const auto [id1, id2] = PairIdToImagePair(pair_id);

  EXPECT_TRUE(cache.ExistsTwoViewGeometry(id1, id2));
  cache.DeleteTwoViewGeometry(id1, id2);
  EXPECT_FALSE(cache.ExistsTwoViewGeometry(id1, id2));
}

TEST(FeatureMatcherCache, DeleteInlierMatches) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const auto all_tvg = data.database->ReadTwoViewGeometries();
  ASSERT_FALSE(all_tvg.empty());

  // Find a pair with inlier matches.
  image_t image_id1 = 0, image_id2 = 0;
  bool found = false;
  for (const auto& [pair_id, tvg] : all_tvg) {
    if (!tvg.inlier_matches.empty()) {
      std::tie(image_id1, image_id2) = PairIdToImagePair(pair_id);
      found = true;
      break;
    }
  }
  ASSERT_TRUE(found);

  EXPECT_TRUE(cache.ExistsInlierMatches(image_id1, image_id2));
  cache.DeleteInlierMatches(image_id1, image_id2);
  EXPECT_FALSE(cache.ExistsInlierMatches(image_id1, image_id2));
  // The two-view geometry entry should still exist.
  EXPECT_TRUE(cache.ExistsTwoViewGeometry(image_id1, image_id2));
}

TEST(FeatureMatcherCache, MaxNumKeypoints) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  const size_t max_num_keypoints = cache.MaxNumKeypoints();
  EXPECT_GT(max_num_keypoints, 0);

  // Calling again should return the cached value.
  EXPECT_EQ(cache.MaxNumKeypoints(), max_num_keypoints);
}

TEST(FeatureMatcherCache, AccessDatabase) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  size_t num_images = 0;
  cache.AccessDatabase([&num_images](Database& database) {
    num_images = database.ReadAllImages().size();
  });
  EXPECT_EQ(num_images, 4);
}

TEST(FeatureMatcherCache, GetFeatureDescriptorIndexCache) {
  auto data = CreateTestData(4);
  FeatureMatcherCache cache(5, data.database);

  auto& index_cache = cache.GetFeatureDescriptorIndexCache();

  const std::vector<Image> images = data.database->ReadAllImages();
  ASSERT_FALSE(images.empty());

  // Access descriptor index for the first image to trigger build.
  auto index = index_cache.Get(images[0].ImageId());
  ASSERT_NE(index, nullptr);
}

}  // namespace
}  // namespace colmap
