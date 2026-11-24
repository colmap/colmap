// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/workspace.h"

#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/testing.h"

#include <filesystem>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

class ParameterizedWorkspaceTests
    : public ::testing::TestWithParam<std::function<std::unique_ptr<Workspace>(
          const Workspace::Options&)>> {
 protected:
  void SetUp() override {
    temp_dir_ = CreateTestDir();
    CreateDirIfNotExists(temp_dir_ / "sparse");
    CreateDirIfNotExists(temp_dir_ / "images");
    CreateDirIfNotExists(temp_dir_ / "stereo");
    CreateDirIfNotExists(temp_dir_ / "stereo" / "depth_maps");
    CreateDirIfNotExists(temp_dir_ / "stereo" / "normal_maps");

    SyntheticDatasetOptions options;
    options.num_rigs = 1;
    options.num_cameras_per_rig = 1;
    options.num_frames_per_rig = 1;
    options.camera_width = 10;
    options.camera_height = 5;
    Reconstruction reconstruction;
    SynthesizeDataset(options, &reconstruction);
    reconstruction.Write(temp_dir_ / "sparse");

    image_name_ = reconstruction.Image(1).Name();

    Mat<float> depth_map(options.camera_width, options.camera_height, 1);
    depth_map.Fill(1.0f);
    depth_map.Write(temp_dir_ / "stereo" / "depth_maps" /
                    (image_name_ + ".geometric.bin"));

    Mat<float> normal_map(options.camera_width, options.camera_height, 3);
    normal_map.Fill(1.0f);
    normal_map.Write(temp_dir_ / "stereo" / "normal_maps" /
                     (image_name_ + ".geometric.bin"));

    Bitmap bitmap(options.camera_width, options.camera_height, true);
    bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
    bitmap.Write(temp_dir_ / "images" / image_name_);
  }

  Workspace::Options GetOptions() {
    Workspace::Options options;
    options.workspace_path = temp_dir_;
    options.workspace_format = "COLMAP";
    options.input_type = "geometric";
    return options;
  }

  std::filesystem::path temp_dir_;
  std::string image_name_;
};

TEST_P(ParameterizedWorkspaceTests, GetData) {
  auto workspace = GetParam()(GetOptions());
  const auto& model = workspace->GetModel();
  EXPECT_EQ(model.images.size(), 1);
  workspace->Load({image_name_});
  EXPECT_TRUE(workspace->HasBitmap(0));
  EXPECT_THAT(workspace->GetBitmapPath(0).string(),
              testing::HasSubstr(image_name_));
  EXPECT_FALSE(workspace->GetBitmap(0).IsEmpty());
  EXPECT_TRUE(workspace->HasDepthMap(0));
  EXPECT_THAT(workspace->GetDepthMapPath(0).string(),
              testing::HasSubstr(image_name_));
  EXPECT_GT(workspace->GetDepthMap(0).GetNumBytes(), 0);
  EXPECT_TRUE(workspace->HasNormalMap(0));
  EXPECT_THAT(workspace->GetNormalMapPath(0).string(),
              testing::HasSubstr(image_name_));
  EXPECT_GT(workspace->GetNormalMap(0).GetNumBytes(), 0);
}

TEST_P(ParameterizedWorkspaceTests, MaxImageSize) {
  Workspace::Options options = GetOptions();
  options.max_image_size = 4;
  auto workspace = GetParam()(options);
  workspace->Load({image_name_});
  ASSERT_EQ(workspace->GetModel().images.size(), 1);
  EXPECT_EQ(workspace->GetModel().images[0].GetWidth(), 4);
  EXPECT_EQ(workspace->GetModel().images[0].GetHeight(), 2);
  EXPECT_EQ(workspace->GetBitmap(0).Width(), 4);
  EXPECT_EQ(workspace->GetBitmap(0).Height(), 2);
  EXPECT_EQ(workspace->GetDepthMap(0).GetWidth(), 4);
  EXPECT_EQ(workspace->GetDepthMap(0).GetHeight(), 2);
  EXPECT_EQ(workspace->GetNormalMap(0).GetWidth(), 4);
  EXPECT_EQ(workspace->GetNormalMap(0).GetHeight(), 2);
}

TEST_P(ParameterizedWorkspaceTests, Load) {
  auto workspace = GetParam()(GetOptions());
  workspace->Load({image_name_});
}

INSTANTIATE_TEST_SUITE_P(WorkspaceTests,
                         ParameterizedWorkspaceTests,
                         ::testing::Values(
                             [](const Workspace::Options& options) {
                               return std::make_unique<Workspace>(options);
                             },
                             [](const Workspace::Options& options) {
                               return std::make_unique<CachedWorkspace>(
                                   options);
                             }));

TEST(Workspace, ImportPMVSClusterVisibility) {
  const auto path = CreateTestDir();
  CreateDirIfNotExists(path / "visualize");
  CreateDirIfNotExists(path / "txt");
  for (int i = 0; i < 4; ++i) {
    Bitmap bitmap(32, 24, true);
    bitmap.Fill(BitmapColor<uint8_t>(10, 20, 30));
    ASSERT_TRUE(bitmap.Write(path / "visualize" / StringPrintf("%08d.jpg", i)));
    std::ofstream projection(path / "txt" / StringPrintf("%08d.txt", i));
    projection << "CONTOUR\n32 0 16 " << -32 * i << "\n0 32 12 0\n0 0 1 0\n";
  }
  {
    std::ofstream visibility(path / "vis.dat");
    visibility << "VISDATA\n4\n0 1 1\n1 2 0 2\n2 2 1 3\n3 1 2\n";
  }
  Workspace::Options options;
  options.workspace_path = path;
  options.workspace_format = "PMVS";
  Workspace workspace(options);
  const auto& model = workspace.GetModel();
  ASSERT_EQ(model.images.size(), 4);
  const std::vector<std::vector<int>> visible = {{1}, {0, 2}, {1, 3}, {2}};
  ASSERT_EQ(model.GetMaxOverlappingImagesFromPMVS(), visible);

  const std::vector<std::pair<std::string, std::vector<int>>> selections = {
      {"timages 4 0 1 2 3", {0, 1, 2, 3}},
      {"timages 4 3 1 0 2", {3, 1, 0, 2}},
      {"timages 2 1 3", {1, 3}},
      {"timages -1 2 4", {2, 3}}};
  for (const auto& [selection, image_idxs] : selections) {
    SCOPED_TRACE(selection);
    {
      std::ofstream option(path / "option-cluster");
      option << selection << '\n';
    }
    ImportPMVSWorkspace(workspace, "option-cluster");
    const auto config = ReadTextFileLines(path / "stereo" / "patch-match.cfg");
    const auto fusion = ReadTextFileLines(path / "stereo" / "fusion.cfg");
    ASSERT_EQ(config.size(), 2 * image_idxs.size());
    ASSERT_EQ(fusion.size(), image_idxs.size());
    for (size_t i = 0; i < image_idxs.size(); ++i) {
      const int image_idx = image_idxs[i];
      EXPECT_EQ(config[2 * i], model.GetImageName(image_idx));
      EXPECT_EQ(fusion[i], model.GetImageName(image_idx));
      std::vector<std::string> sources;
      sources.reserve(visible[image_idx].size());
      for (const int source_idx : visible[image_idx]) {
        sources.push_back(model.GetImageName(source_idx));
      }
      EXPECT_EQ(CSVToVector<std::string>(config[2 * i + 1]), sources);
    }
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
