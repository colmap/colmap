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

#include "colmap/mvs/workspace.h"

#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <filesystem>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

// Shared helper that creates a single-image COLMAP workspace on disk.
// Returns the image name used by the synthetic dataset.
struct WorkspaceFixture {
  std::filesystem::path temp_dir;
  std::string image_name;
  int camera_width = 10;
  int camera_height = 5;

  void Create() { Create(camera_width, camera_height); }

  void Create(int width, int height) {
    camera_width = width;
    camera_height = height;
    temp_dir = CreateTestDir();
    CreateDirIfNotExists(temp_dir / "sparse");
    CreateDirIfNotExists(temp_dir / "images");
    CreateDirIfNotExists(temp_dir / "stereo");
    CreateDirIfNotExists(temp_dir / "stereo" / "depth_maps");
    CreateDirIfNotExists(temp_dir / "stereo" / "normal_maps");

    SyntheticDatasetOptions syn_opts;
    syn_opts.num_rigs = 1;
    syn_opts.num_cameras_per_rig = 1;
    syn_opts.num_frames_per_rig = 1;
    syn_opts.camera_width = camera_width;
    syn_opts.camera_height = camera_height;
    Reconstruction reconstruction;
    SynthesizeDataset(syn_opts, &reconstruction);
    reconstruction.Write(temp_dir / "sparse");

    image_name = reconstruction.Image(1).Name();

    WriteDepthMap(image_name, 1.0f);
    WriteNormalMap(image_name, 1.0f);
    WriteBitmap(image_name);
  }

  void WriteDepthMap(const std::string& name, float fill_value) {
    Mat<float> dm(camera_width, camera_height, 1);
    dm.Fill(fill_value);
    dm.Write(temp_dir / "stereo" / "depth_maps" / (name + ".geometric.bin"));
  }

  void WriteNormalMap(const std::string& name, float fill_value) {
    Mat<float> nm(camera_width, camera_height, 3);
    nm.Fill(fill_value);
    nm.Write(temp_dir / "stereo" / "normal_maps" / (name + ".geometric.bin"));
  }

  void WriteBitmap(const std::string& name) {
    Bitmap bmp(camera_width, camera_height, true);
    bmp.Fill(BitmapColor<uint8_t>(0, 0, 0));
    bmp.Write(temp_dir / "images" / name);
  }

  Workspace::Options GetOptions() const {
    Workspace::Options opts;
    opts.workspace_path = temp_dir;
    opts.workspace_format = "COLMAP";
    opts.input_type = "geometric";
    return opts;
  }
};

// --------------------------------------------------------------------------
// Parameterized tests (existing + expanded)
// --------------------------------------------------------------------------

class ParameterizedWorkspaceTests
    : public ::testing::TestWithParam<std::function<std::unique_ptr<Workspace>(
          const Workspace::Options&)>> {
 protected:
  void SetUp() override { fixture_.Create(); }

  Workspace::Options GetOptions() { return fixture_.GetOptions(); }

  WorkspaceFixture fixture_;
  std::string& image_name_ = fixture_.image_name;
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

// --------------------------------------------------------------------------
// Non-parameterized tests for specific code paths
// --------------------------------------------------------------------------

// Verify that input_type is normalized to lowercase in the constructor
// and that GetFileName produces the expected file name.
TEST(WorkspaceTest, InputTypeCaseNormalization) {
  WorkspaceFixture f;
  f.Create();
  Workspace::Options opts = f.GetOptions();
  opts.input_type = "GeoMetRIC";
  // Depth and normal maps were written with ".geometric.bin" suffix,
  // so the lowercased input_type should match.
  Workspace ws(opts);
  EXPECT_TRUE(ws.HasDepthMap(0));
  EXPECT_TRUE(ws.HasNormalMap(0));
  // The path should contain the lowercased input type.
  EXPECT_THAT(ws.GetDepthMapPath(0).string(),
              testing::HasSubstr(".geometric.bin"));
  EXPECT_THAT(ws.GetNormalMapPath(0).string(),
              testing::HasSubstr(".geometric.bin"));
}

// Verify that GetDepthMapPath and GetNormalMapPath build paths under the
// configured stereo_folder.
TEST(WorkspaceTest, DepthNormalMapPaths) {
  WorkspaceFixture f;
  f.Create();
  Workspace::Options opts = f.GetOptions();
  Workspace ws(opts);
  const auto depth_path = ws.GetDepthMapPath(0);
  const auto normal_path = ws.GetNormalMapPath(0);
  // Paths should be under <workspace>/stereo/depth_maps and normal_maps.
  EXPECT_THAT(depth_path.string(), testing::HasSubstr("stereo/depth_maps"));
  EXPECT_THAT(normal_path.string(), testing::HasSubstr("stereo/normal_maps"));
  // Both should end with ".geometric.bin".
  EXPECT_THAT(depth_path.filename().string(),
              testing::HasSubstr(".geometric.bin"));
  EXPECT_THAT(normal_path.filename().string(),
              testing::HasSubstr(".geometric.bin"));
}

// When bitmap or depth map file is missing, Load should skip that image
// without crashing.
TEST(WorkspaceTest, LoadSkipsMissingFiles) {
  WorkspaceFixture f;
  f.Create();
  // Remove the depth map file so HasDepthMap returns false.
  std::filesystem::remove(f.temp_dir / "stereo" / "depth_maps" /
                          (f.image_name + ".geometric.bin"));
  Workspace::Options opts = f.GetOptions();
  Workspace ws(opts);
  // Load should print a warning but not crash.
  EXPECT_NO_THROW(ws.Load({f.image_name}));
  // The image data should NOT have been loaded (vectors remain empty/null).
  EXPECT_FALSE(ws.HasDepthMap(0));
}

// Verify HasBitmap/HasDepthMap/HasNormalMap return false when files are absent.
TEST(WorkspaceTest, HasMethodsReturnFalseWhenFileMissing) {
  WorkspaceFixture f;
  f.Create();
  // Remove the normal map file.
  std::filesystem::remove(f.temp_dir / "stereo" / "normal_maps" /
                          (f.image_name + ".geometric.bin"));
  Workspace ws(f.GetOptions());
  EXPECT_TRUE(ws.HasBitmap(0));
  EXPECT_TRUE(ws.HasDepthMap(0));
  EXPECT_FALSE(ws.HasNormalMap(0));
}

// Verify CachedWorkspace::Load is a no-op (data loaded on demand via Get*).
TEST(CachedWorkspaceTest, LoadIsNoOp) {
  WorkspaceFixture f;
  f.Create();
  Workspace::Options opts = f.GetOptions();
  CachedWorkspace ws(opts);
  // Load should do nothing and not crash.
  ws.Load({f.image_name});
  // Data is loaded on demand, verify we can still access it.
  EXPECT_FALSE(ws.GetBitmap(0).IsEmpty());
  EXPECT_GT(ws.GetDepthMap(0).GetNumBytes(), 0);
  EXPECT_GT(ws.GetNormalMap(0).GetNumBytes(), 0);
}

// Verify CachedWorkspace rescales on Get when max_image_size is set.
TEST(CachedWorkspaceTest, RescalesOnGetWithMaxImageSize) {
  WorkspaceFixture f;
  f.Create(10, 5);
  Workspace::Options opts = f.GetOptions();
  opts.max_image_size = 4;
  CachedWorkspace ws(opts);
  // Bitmap should be rescaled to 4x2 on first access.
  const auto& bmp = ws.GetBitmap(0);
  EXPECT_EQ(bmp.Width(), 4);
  EXPECT_EQ(bmp.Height(), 2);
  // Depth map should be downsized to 4x2.
  const auto& dm = ws.GetDepthMap(0);
  EXPECT_EQ(dm.GetWidth(), 4);
  EXPECT_EQ(dm.GetHeight(), 2);
  // Normal map should be downsized to 4x2.
  const auto& nm = ws.GetNormalMap(0);
  EXPECT_EQ(nm.GetWidth(), 4);
  EXPECT_EQ(nm.GetHeight(), 2);
}

// ClearCache should allow data to be reloaded from disk.
TEST(CachedWorkspaceTest, ClearCacheReloadsData) {
  WorkspaceFixture f;
  f.Create();
  Workspace::Options opts = f.GetOptions();
  CachedWorkspace ws(opts);
  // Load data into cache.
  EXPECT_GT(ws.GetDepthMap(0).GetNumBytes(), 0);
  // Clear and reload.
  ws.ClearCache();
  EXPECT_GT(ws.GetDepthMap(0).GetNumBytes(), 0);
}

// Verify CachedWorkspace without max_image_size returns original dimensions.
TEST(CachedWorkspaceTest, NoRescaleWithoutMaxImageSize) {
  WorkspaceFixture f;
  f.Create(10, 5);
  Workspace::Options opts = f.GetOptions();
  // max_image_size defaults to -1 (no rescaling).
  CachedWorkspace ws(opts);
  EXPECT_EQ(ws.GetBitmap(0).Width(), 10);
  EXPECT_EQ(ws.GetBitmap(0).Height(), 5);
  EXPECT_EQ(ws.GetDepthMap(0).GetWidth(), 10);
  EXPECT_EQ(ws.GetDepthMap(0).GetHeight(), 5);
  EXPECT_EQ(ws.GetNormalMap(0).GetWidth(), 10);
  EXPECT_EQ(ws.GetNormalMap(0).GetHeight(), 5);
}

// --------------------------------------------------------------------------
// ImportPMVSWorkspace tests
// --------------------------------------------------------------------------

// Helper to create a minimal COLMAP workspace with multiple images for PMVS
// import testing.
struct MultiImageWorkspaceFixture {
  std::filesystem::path temp_dir;
  std::vector<std::string> image_names;
  int camera_width = 10;
  int camera_height = 5;

  void Create(int num_images) {
    temp_dir = CreateTestDir();
    CreateDirIfNotExists(temp_dir / "sparse");
    CreateDirIfNotExists(temp_dir / "images");
    CreateDirIfNotExists(temp_dir / "stereo");
    CreateDirIfNotExists(temp_dir / "stereo" / "depth_maps");
    CreateDirIfNotExists(temp_dir / "stereo" / "normal_maps");

    SyntheticDatasetOptions syn_opts;
    syn_opts.num_rigs = 1;
    syn_opts.num_cameras_per_rig = 1;
    syn_opts.num_frames_per_rig = num_images;
    syn_opts.camera_width = camera_width;
    syn_opts.camera_height = camera_height;
    Reconstruction reconstruction;
    SynthesizeDataset(syn_opts, &reconstruction);
    reconstruction.Write(temp_dir / "sparse");

    image_names.clear();
    for (const image_t image_id : reconstruction.RegImageIds()) {
      const std::string& name = reconstruction.Image(image_id).Name();
      image_names.push_back(name);

      Mat<float> dm(camera_width, camera_height, 1);
      dm.Fill(1.0f);
      dm.Write(temp_dir / "stereo" / "depth_maps" /
               (name + ".geometric.bin"));

      Mat<float> nm(camera_width, camera_height, 3);
      nm.Fill(1.0f);
      nm.Write(temp_dir / "stereo" / "normal_maps" /
               (name + ".geometric.bin"));

      Bitmap bmp(camera_width, camera_height, true);
      bmp.Fill(BitmapColor<uint8_t>(0, 0, 0));
      bmp.Write(temp_dir / "images" / name);
    }
  }

  Workspace::Options GetOptions() const {
    Workspace::Options opts;
    opts.workspace_path = temp_dir;
    opts.workspace_format = "COLMAP";
    opts.input_type = "geometric";
    return opts;
  }
};

// ImportPMVSWorkspace with explicit image list in the option file:
//   timages N idx0 idx1 ... idxN-1
TEST(ImportPMVSWorkspaceTest, ExplicitImageList) {
  MultiImageWorkspaceFixture f;
  f.Create(3);
  Workspace::Options opts = f.GetOptions();
  Workspace ws(opts);

  // Write a PMVS option file with explicit image indices.
  {
    std::ofstream opt_file(f.temp_dir / "option-all");
    opt_file << "timages 3 0 1 2\n";
  }

  ImportPMVSWorkspace(ws, "option-all");

  // Verify that patch-match.cfg and fusion.cfg were created.
  const auto pm_path = f.temp_dir / "stereo" / "patch-match.cfg";
  const auto fusion_path = f.temp_dir / "stereo" / "fusion.cfg";
  ASSERT_TRUE(ExistsFile(pm_path));
  ASSERT_TRUE(ExistsFile(fusion_path));

  // Read and verify patch-match.cfg has entries for all 3 images.
  auto pm_lines = ReadTextFileLines(pm_path);
  // Each image produces 2 lines: image name + source images.
  // Filter out empty lines.
  std::vector<std::string> non_empty_pm;
  for (const auto& line : pm_lines) {
    if (!line.empty()) non_empty_pm.push_back(line);
  }
  EXPECT_EQ(non_empty_pm.size(), 6);

  // Read and verify fusion.cfg has entries for all 3 images.
  auto fusion_lines = ReadTextFileLines(fusion_path);
  std::vector<std::string> non_empty_fusion;
  for (const auto& line : fusion_lines) {
    if (!line.empty()) non_empty_fusion.push_back(line);
  }
  EXPECT_EQ(non_empty_fusion.size(), 3);
}

// ImportPMVSWorkspace with range format:
//   timages -1 lower upper
TEST(ImportPMVSWorkspaceTest, RangeFormat) {
  MultiImageWorkspaceFixture f;
  f.Create(3);
  Workspace::Options opts = f.GetOptions();
  Workspace ws(opts);

  // Write a PMVS option file with range format (images 0 to 2).
  {
    std::ofstream opt_file(f.temp_dir / "option-range");
    opt_file << "some_other_option value\n";
    opt_file << "timages -1 0 2\n";
  }

  ImportPMVSWorkspace(ws, "option-range");

  const auto pm_path = f.temp_dir / "stereo" / "patch-match.cfg";
  const auto fusion_path = f.temp_dir / "stereo" / "fusion.cfg";
  ASSERT_TRUE(ExistsFile(pm_path));
  ASSERT_TRUE(ExistsFile(fusion_path));

  // Range is [0, 2), so 2 images.
  auto fusion_lines = ReadTextFileLines(fusion_path);
  std::vector<std::string> non_empty;
  for (const auto& line : fusion_lines) {
    if (!line.empty()) non_empty.push_back(line);
  }
  EXPECT_EQ(non_empty.size(), 2);
}

// ImportPMVSWorkspace creates the stereo subdirectory structure.
TEST(ImportPMVSWorkspaceTest, CreatesDirectories) {
  MultiImageWorkspaceFixture f;
  f.Create(1);

  // Remove the stereo subdirectories to verify they get created.
  std::filesystem::remove_all(f.temp_dir / "stereo");

  Workspace::Options opts = f.GetOptions();
  Workspace ws(opts);

  {
    std::ofstream opt_file(f.temp_dir / "option-dirs");
    opt_file << "timages 1 0\n";
  }

  ImportPMVSWorkspace(ws, "option-dirs");

  EXPECT_TRUE(std::filesystem::exists(f.temp_dir / "stereo" / "depth_maps"));
  EXPECT_TRUE(std::filesystem::exists(f.temp_dir / "stereo" / "normal_maps"));
  EXPECT_TRUE(
      std::filesystem::exists(f.temp_dir / "stereo" / "consistency_graphs"));
}

// ImportPMVSWorkspace skips non-timages lines in the option file.
TEST(ImportPMVSWorkspaceTest, SkipsNonTimagesLines) {
  MultiImageWorkspaceFixture f;
  f.Create(1);
  Workspace::Options opts = f.GetOptions();
  Workspace ws(opts);

  // Write an option file where timages is NOT the first line.
  {
    std::ofstream opt_file(f.temp_dir / "option-skip");
    opt_file << "level 1\n";
    opt_file << "csize 2\n";
    opt_file << "threshold 0.7\n";
    opt_file << "timages 1 0\n";
    opt_file << "oimages 0\n";
  }

  ImportPMVSWorkspace(ws, "option-skip");

  const auto pm_path = f.temp_dir / "stereo" / "patch-match.cfg";
  ASSERT_TRUE(ExistsFile(pm_path));
  auto pm_lines = ReadTextFileLines(pm_path);
  std::vector<std::string> non_empty;
  for (const auto& line : pm_lines) {
    if (!line.empty()) non_empty.push_back(line);
  }
  // 1 image = 2 lines in patch-match.cfg (image name + __auto__).
  EXPECT_EQ(non_empty.size(), 2);
}

// Verify the patch-match.cfg contains "__auto__, 20" when no PMVS overlapping
// images are available (which is the case for COLMAP-format workspaces).
TEST(ImportPMVSWorkspaceTest, AutoSourceImagesForColmapWorkspace) {
  MultiImageWorkspaceFixture f;
  f.Create(2);
  Workspace::Options opts = f.GetOptions();
  Workspace ws(opts);

  {
    std::ofstream opt_file(f.temp_dir / "option-auto");
    opt_file << "timages 2 0 1\n";
  }

  ImportPMVSWorkspace(ws, "option-auto");

  const auto pm_path = f.temp_dir / "stereo" / "patch-match.cfg";
  auto pm_lines = ReadTextFileLines(pm_path);
  // Every other line should be "__auto__, 20".
  bool found_auto = false;
  for (const auto& line : pm_lines) {
    if (line.find("__auto__, 20") != std::string::npos) {
      found_auto = true;
      break;
    }
  }
  EXPECT_TRUE(found_auto);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
