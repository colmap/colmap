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
    CreateDirIfNotExists(JoinPaths(temp_dir_, "sparse"));
    CreateDirIfNotExists(JoinPaths(temp_dir_, "images"));
    CreateDirIfNotExists(JoinPaths(temp_dir_, "stereo"));
    CreateDirIfNotExists(JoinPaths(temp_dir_, "stereo", "depth_maps"));
    CreateDirIfNotExists(JoinPaths(temp_dir_, "stereo", "normal_maps"));

    SyntheticDatasetOptions options;
    options.num_rigs = 1;
    options.num_cameras_per_rig = 1;
    options.num_frames_per_rig = 1;
    options.camera_width = 10;
    options.camera_height = 5;
    Reconstruction reconstruction;
    SynthesizeDataset(options, &reconstruction);
    reconstruction.Write(JoinPaths(temp_dir_, "sparse"));

    image_name_ = reconstruction.Image(1).Name();

    Mat<float> depth_map(options.camera_width, options.camera_height, 1);
    depth_map.Fill(1.0f);
    depth_map.Write(JoinPaths(
        temp_dir_, "stereo", "depth_maps", image_name_ + ".geometric.bin"));

    Mat<float> normal_map(options.camera_width, options.camera_height, 3);
    normal_map.Fill(1.0f);
    normal_map.Write(JoinPaths(
        temp_dir_, "stereo", "normal_maps", image_name_ + ".geometric.bin"));

    Bitmap bitmap(options.camera_width, options.camera_height, true);
    bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
    bitmap.Write(JoinPaths(temp_dir_, "images", image_name_));
  }

  Workspace::Options GetOptions() {
    Workspace::Options options;
    options.workspace_path = temp_dir_;
    options.workspace_format = "COLMAP";
    options.input_type = "geometric";
    return options;
  }

  std::string temp_dir_;
  std::string image_name_;
};

TEST_P(ParameterizedWorkspaceTests, GetData) {
  auto workspace = GetParam()(GetOptions());
  const auto& model = workspace->GetModel();
  EXPECT_EQ(model.images.size(), 1);
  EXPECT_TRUE(workspace->HasBitmap(0));
  EXPECT_EQ(workspace->GetBitmapPath(0),
            JoinPaths(temp_dir_, "images", image_name_));
  EXPECT_TRUE(workspace->HasDepthMap(0));
  EXPECT_EQ(
      workspace->GetDepthMapPath(0),
      JoinPaths(
          temp_dir_, "stereo", "depth_maps", image_name_ + ".geometric.bin"));
  EXPECT_TRUE(workspace->HasNormalMap(0));
  EXPECT_EQ(
      workspace->GetNormalMapPath(0),
      JoinPaths(
          temp_dir_, "stereo", "normal_maps", image_name_ + ".geometric.bin"));
}

TEST_P(ParameterizedWorkspaceTests, MaxImageSize) {
  Workspace::Options options = GetOptions();
  options.max_image_size = 4;
  auto workspace = GetParam()(options);
  ASSERT_EQ(workspace->GetModel().images.size(), 1);
  EXPECT_EQ(workspace->GetModel().images[0].GetWidth(), 4);
  EXPECT_EQ(workspace->GetModel().images[0].GetHeight(), 2);
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

}  // namespace
}  // namespace mvs
}  // namespace colmap
