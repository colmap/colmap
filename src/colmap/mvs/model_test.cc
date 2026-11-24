// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/model.h"

#include "colmap/scene/synthetic.h"
#include "colmap/util/string.h"
#include "colmap/util/testing.h"

#include <cmath>
#include <fstream>

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

void WriteCOLMAPModel(const std::filesystem::path& path,
                      int num_images,
                      int num_points,
                      const std::string& name_prefix) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = num_images;
  options.num_points3D = num_points;
  SynthesizeDataset(options, &reconstruction);
  for (const auto& [image_id, image] : reconstruction.Images()) {
    reconstruction.Image(image_id).SetName(name_prefix + image.Name());
  }
  std::filesystem::create_directories(path / "sparse");
  reconstruction.WriteBinary(path / "sparse");
}

TEST(Model, ReadCOLMAPReplacesPreviousModel) {
  const auto test_dir = CreateTestDir();
  const auto initial_path = test_dir / "initial";
  const auto replacement_path = test_dir / "replacement";
  WriteCOLMAPModel(initial_path, 3, 10, "initial/");
  WriteCOLMAPModel(replacement_path, 2, 4, "replacement/");

  Model model;
  model.ReadFromCOLMAP(initial_path);
  const std::string initial_name = model.GetImageName(0);
  model.Read(initial_path, "COLMAP");
  EXPECT_EQ(model.images.size(), 3);
  EXPECT_EQ(model.points.size(), 10);
  for (const auto& shared : model.ComputeSharedPoints()) {
    for (const auto& [_, count] : shared) {
      EXPECT_EQ(count, 10);
    }
  }

  model.ReadFromCOLMAP(replacement_path);
  ASSERT_EQ(model.images.size(), 2);
  ASSERT_EQ(model.points.size(), 4);
  EXPECT_THROW(model.GetImageIdx(initial_name), std::exception);
  for (int image_idx = 0; image_idx < 2; ++image_idx) {
    const std::string name = model.GetImageName(image_idx);
    EXPECT_EQ(model.GetImageIdx(name), image_idx);
    EXPECT_EQ(model.images[image_idx].GetPath(),
              replacement_path / "images" / name);
  }
  for (const auto& point : model.points) {
    for (const int image_idx : point.track) {
      EXPECT_GE(image_idx, 0);
      EXPECT_LT(image_idx, model.images.size());
    }
  }

  const std::string replacement_name = model.GetImageName(0);
  Reconstruction empty_reconstruction;
  empty_reconstruction.WriteBinary(replacement_path / "sparse");
  model.ReadFromCOLMAP(replacement_path);
  EXPECT_TRUE(model.images.empty());
  EXPECT_TRUE(model.points.empty());
  EXPECT_TRUE(model.GetMaxOverlappingImagesFromPMVS().empty());
  EXPECT_THROW(model.GetImageIdx(replacement_name), std::exception);
  EXPECT_THROW(model.GetImageName(0), std::exception);
}

void WritePMVSModel(const std::filesystem::path& path,
                    int num_images,
                    bool bundler) {
  std::filesystem::create_directories(path / "visualize");
  std::filesystem::create_directories(path / "txt");
  Bitmap bitmap(10, 10, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(12, 34, 56));
  for (int image_idx = 0; image_idx < num_images; ++image_idx) {
    ASSERT_TRUE(
        bitmap.Write(path / "visualize" / StringPrintf("%08d.jpg", image_idx)));
    std::ofstream projection(path / "txt" /
                             StringPrintf("%08d.txt", image_idx));
    projection << "CONTOUR\n100 0 5 " << -100 * image_idx
               << "\n0 100 5 0\n0 0 1 0\n";
  }
  if (bundler) {
    std::ofstream bundle(path / "bundle.rd.out");
    bundle << "# Bundle file v0.3\n"
           << num_images << " " << (num_images > 0 ? 1 : 0) << "\n";
    for (int image_idx = 0; image_idx < num_images; ++image_idx) {
      bundle << "100 0 0\n1 0 0\n0 1 0\n0 0 1\n" << image_idx << " 0 0\n";
    }
    if (num_images > 0) {
      bundle << "0 0 -10\n255 255 255\n" << num_images;
      for (int image_idx = 0; image_idx < num_images; ++image_idx) {
        bundle << " " << image_idx << " 0 0 0";
      }
      bundle << "\n";
    }
  } else {
    std::ofstream visibility(path / "vis.dat");
    visibility << "VISDATA\n" << num_images << "\n";
    for (int image_idx = 0; image_idx < num_images; ++image_idx) {
      visibility << image_idx << " " << num_images - 1;
      for (int other_idx = 0; other_idx < num_images; ++other_idx) {
        if (other_idx != image_idx) visibility << " " << other_idx;
      }
      visibility << "\n";
    }
  }
}

class ModelReadPMVSTest : public ::testing::TestWithParam<bool> {};

TEST_P(ModelReadPMVSTest, ReplacesPreviousModel) {
  const auto test_dir = CreateTestDir();
  const auto initial_path = test_dir / "initial";
  const auto replacement_path = test_dir / "replacement";
  WritePMVSModel(initial_path, 3, /*bundler=*/false);
  WritePMVSModel(replacement_path, 2, /*bundler=*/GetParam());

  Model model;
  model.ReadFromPMVS(initial_path);
  ASSERT_EQ(model.GetMaxOverlappingImagesFromPMVS().size(), 3);
  ASSERT_NO_THROW(model.ReadFromPMVS(replacement_path));
  ASSERT_EQ(model.images.size(), 2);
  EXPECT_EQ(model.points.size(), GetParam() ? 1 : 0);
  EXPECT_THROW(model.GetImageIdx("00000002.jpg"), std::exception);
  for (int image_idx = 0; image_idx < 2; ++image_idx) {
    const std::string name = StringPrintf("%08d.jpg", image_idx);
    EXPECT_EQ(model.GetImageName(image_idx), name);
    EXPECT_EQ(model.GetImageIdx(name), image_idx);
    EXPECT_EQ(model.images[image_idx].GetPath(),
              replacement_path / "visualize" / name);
  }
  if (GetParam()) {
    EXPECT_TRUE(model.GetMaxOverlappingImagesFromPMVS().empty());
  } else {
    EXPECT_EQ(model.GetMaxOverlappingImagesFromPMVS(),
              (std::vector<std::vector<int>>{{1}, {0}}));
  }
  ASSERT_NO_THROW(model.Read(replacement_path, "PMVS"));
  EXPECT_EQ(model.images.size(), 2);
  EXPECT_EQ(model.points.size(), GetParam() ? 1 : 0);

  WriteCOLMAPModel(test_dir, 2, 4, "colmap/");
  model.ReadFromCOLMAP(test_dir);
  EXPECT_EQ(model.images.size(), 2);
  EXPECT_EQ(model.points.size(), 4);
  EXPECT_TRUE(model.GetMaxOverlappingImagesFromPMVS().empty());
  EXPECT_THROW(model.GetImageIdx("00000000.jpg"), std::exception);

  const std::string previous_name = model.GetImageName(0);
  const auto empty_path = test_dir / "empty";
  WritePMVSModel(empty_path, 0, /*bundler=*/GetParam());
  model.ReadFromPMVS(empty_path);
  EXPECT_TRUE(model.images.empty());
  EXPECT_TRUE(model.points.empty());
  EXPECT_TRUE(model.GetMaxOverlappingImagesFromPMVS().empty());
  EXPECT_THROW(model.GetImageIdx(previous_name), std::exception);
  EXPECT_THROW(model.GetImageName(0), std::exception);
}

INSTANTIATE_TEST_SUITE_P(Model, ModelReadPMVSTest, ::testing::Bool());

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

}  // namespace
}  // namespace mvs
}  // namespace colmap
