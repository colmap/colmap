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

#include "colmap/mvs/fusion.h"

#include "colmap/mvs/consistency_graph.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(StereoFusion, Integration) {
  std::string temp_dir = CreateTestDir();
  CreateDirIfNotExists(JoinPaths(temp_dir, "sparse"));
  CreateDirIfNotExists(JoinPaths(temp_dir, "images"));
  CreateDirIfNotExists(JoinPaths(temp_dir, "stereo"));
  CreateDirIfNotExists(JoinPaths(temp_dir, "stereo", "depth_maps"));
  CreateDirIfNotExists(JoinPaths(temp_dir, "stereo", "normal_maps"));

  // Create synthetic reconstruction with 2 overlapping images.
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.camera_width = 30;
  synthetic_dataset_options.camera_height = 20;
  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  reconstruction.Write(JoinPaths(temp_dir, "sparse"));

  // Create depth maps, normal maps, and consistency graphs for both images.
  std::vector<std::string> image_names;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    image_names.push_back(image.Name());

    // Create depth map with constant depth.
    Mat<float> depth_map(synthetic_dataset_options.camera_width,
                         synthetic_dataset_options.camera_height,
                         1);
    depth_map.Fill(5.0f);
    depth_map.Write(JoinPaths(
        temp_dir, "stereo", "depth_maps", image.Name() + ".geometric.bin"));

    // Create normal map pointing in z direction.
    Mat<float> normal_map(synthetic_dataset_options.camera_width,
                          synthetic_dataset_options.camera_height,
                          3);
    for (int i = 0; i < normal_map.GetHeight() * normal_map.GetWidth(); ++i) {
      normal_map.GetPtr()[3 * i + 0] = 0.0f;  // nx
      normal_map.GetPtr()[3 * i + 1] = 0.0f;  // ny
      normal_map.GetPtr()[3 * i + 2] = 1.0f;  // nz
    }
    normal_map.Write(JoinPaths(
        temp_dir, "stereo", "normal_maps", image.Name() + ".geometric.bin"));

    // Create bitmap.
    Bitmap bitmap(synthetic_dataset_options.camera_width,
                  synthetic_dataset_options.camera_height,
                  true);
    bitmap.Fill(BitmapColor<uint8_t>(128, 128, 128));
    bitmap.Write(JoinPaths(temp_dir, "images", image.Name()));
  }

  // Write fusion config
  std::ofstream fusion_cfg(JoinPaths(temp_dir, "stereo", "fusion.cfg"));
  for (const auto& name : image_names) {
    fusion_cfg << name << "\n";
  }
  fusion_cfg.close();

  // Run fusion
  StereoFusionOptions options;
  options.min_num_pixels = 1;
  options.max_num_pixels = 100;
  options.max_traversal_depth = 10;
  options.check_num_images = 10;
  options.use_cache = false;

  StereoFusion fusion(options, temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  // Verify that some points were fused
  const auto& fused_points = fusion.GetFusedPoints();
  const auto& visibility = fusion.GetFusedPointsVisibility();

  EXPECT_GT(fused_points.size(), 0);
  EXPECT_EQ(fused_points.size(), visibility.size());

  for (const auto& point : fused_points) {
    EXPECT_GT(point.x, -10.0f);
    EXPECT_LT(point.x, 10.0f);
    EXPECT_GT(point.y, -10.0f);
    EXPECT_LT(point.y, 10.0f);
    EXPECT_GT(point.z, -10.0f);
    EXPECT_LT(point.z, 10.0f);
    EXPECT_FLOAT_EQ(
        point.nx * point.nx + point.ny * point.ny + point.nz * point.nz, 1.0f);
  }

  for (const auto& vis : visibility) {
    EXPECT_GT(vis.size(), 0);
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
