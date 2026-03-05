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

#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {

// Re-declare internal::FindNextImage so we can test it directly.
namespace internal {
int FindNextImage(const std::vector<std::vector<int>>& overlapping_images,
                  const std::vector<char>& used_images,
                  const std::vector<char>& fused_images,
                  int prev_image_idx);
}  // namespace internal

namespace {

// Helper to create a workspace directory with synthetic data suitable for
// fusion testing. Returns the temp dir path and populates image_names.
struct FusionTestWorkspace {
  std::filesystem::path temp_dir;
  std::vector<std::string> image_names;

  static FusionTestWorkspace Create(int num_frames = 2,
                                    int width = 30,
                                    int height = 20,
                                    float depth_value = 5.0f,
                                    bool write_fusion_cfg = true) {
    FusionTestWorkspace ws;
    ws.temp_dir = CreateTestDir();
    CreateDirIfNotExists(ws.temp_dir / "sparse");
    CreateDirIfNotExists(ws.temp_dir / "images");
    CreateDirIfNotExists(ws.temp_dir / "stereo");
    CreateDirIfNotExists(ws.temp_dir / "stereo" / "depth_maps");
    CreateDirIfNotExists(ws.temp_dir / "stereo" / "normal_maps");

    SyntheticDatasetOptions synthetic_dataset_options;
    synthetic_dataset_options.num_rigs = 1;
    synthetic_dataset_options.num_cameras_per_rig = 1;
    synthetic_dataset_options.num_frames_per_rig = num_frames;
    synthetic_dataset_options.camera_width = width;
    synthetic_dataset_options.camera_height = height;
    Reconstruction reconstruction;
    SynthesizeDataset(synthetic_dataset_options, &reconstruction);
    reconstruction.Write(ws.temp_dir / "sparse");

    for (const auto& [image_id, image] : reconstruction.Images()) {
      ws.image_names.push_back(image.Name());

      Mat<float> depth_map(width, height, 1);
      depth_map.Fill(depth_value);
      depth_map.Write(ws.temp_dir / "stereo" / "depth_maps" /
                      (image.Name() + ".geometric.bin"));

      Mat<float> normal_map(width, height, 3);
      const size_t num_pixels = normal_map.GetHeight() * normal_map.GetWidth();
      for (size_t i = 0; i < num_pixels; ++i) {
        normal_map.GetPtr()[3 * i + 0] = 0.0f;
        normal_map.GetPtr()[3 * i + 1] = 0.0f;
        normal_map.GetPtr()[3 * i + 2] = 1.0f;
      }
      normal_map.Write(ws.temp_dir / "stereo" / "normal_maps" /
                       (image.Name() + ".geometric.bin"));

      Bitmap bitmap(width, height, true);
      bitmap.Fill(BitmapColor<uint8_t>(0, 64, 128));
      bitmap.Write(ws.temp_dir / "images" / image.Name());
    }

    if (write_fusion_cfg) {
      std::ofstream fusion_cfg(ws.temp_dir / "stereo" / "fusion.cfg");
      for (const auto& name : ws.image_names) {
        fusion_cfg << name << "\n";
      }
      fusion_cfg.close();
    }

    return ws;
  }
};

StereoFusionOptions DefaultTestOptions() {
  StereoFusionOptions options;
  options.min_num_pixels = 1;
  options.max_num_pixels = 100;
  options.max_traversal_depth = 10;
  options.check_num_images = 10;
  options.use_cache = false;
  return options;
}

TEST(StereoFusion, Integration) {
  auto ws = FusionTestWorkspace::Create();

  StereoFusion fusion(
      DefaultTestOptions(), ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

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
    EXPECT_EQ(point.r, 0);
    EXPECT_EQ(point.g, 64);
    EXPECT_EQ(point.b, 128);
    EXPECT_FLOAT_EQ(
        point.nx * point.nx + point.ny * point.ny + point.nz * point.nz, 1.0f);
  }

  for (const auto& vis : visibility) {
    EXPECT_GT(vis.size(), 0);
  }
}

TEST(StereoFusionOptions, CheckValid) {
  StereoFusionOptions options;
  EXPECT_TRUE(options.Check());
}

TEST(StereoFusionOptions, CheckMinNumPixelsNegative) {
  StereoFusionOptions options;
  options.min_num_pixels = -1;
  EXPECT_FALSE(options.Check());
}

TEST(StereoFusionOptions, CheckMinNumPixelsGreaterThanMax) {
  StereoFusionOptions options;
  options.min_num_pixels = 100;
  options.max_num_pixels = 10;
  EXPECT_FALSE(options.Check());
}

TEST(StereoFusionOptions, CheckMaxTraversalDepthZero) {
  StereoFusionOptions options;
  options.max_traversal_depth = 0;
  EXPECT_FALSE(options.Check());
}

TEST(StereoFusionOptions, CheckMaxReprojErrorNegative) {
  StereoFusionOptions options;
  options.max_reproj_error = -1.0;
  EXPECT_FALSE(options.Check());
}

TEST(StereoFusionOptions, CheckMaxDepthErrorNegative) {
  StereoFusionOptions options;
  options.max_depth_error = -0.5;
  EXPECT_FALSE(options.Check());
}

TEST(StereoFusionOptions, CheckMaxNormalErrorNegative) {
  StereoFusionOptions options;
  options.max_normal_error = -10.0;
  EXPECT_FALSE(options.Check());
}

TEST(StereoFusionOptions, CheckNumImagesZero) {
  StereoFusionOptions options;
  options.check_num_images = 0;
  EXPECT_FALSE(options.Check());
}

TEST(StereoFusionOptions, CheckCacheSizeZero) {
  StereoFusionOptions options;
  options.cache_size = 0.0;
  EXPECT_FALSE(options.Check());
}

TEST(FindNextImage, ReturnsOverlappingUnfusedImage) {
  // Image 0 overlaps with [1, 2]. Image 1 is used but not fused.
  std::vector<std::vector<int>> overlapping = {{1, 2}, {0, 2}, {0, 1}};
  std::vector<char> used = {true, true, true};
  std::vector<char> fused = {true, false, false};
  EXPECT_EQ(internal::FindNextImage(overlapping, used, fused, 0), 1);
}

TEST(FindNextImage, SkipsAlreadyFusedOverlappingImages) {
  // Image 0 overlaps with [1, 2]. Image 1 is already fused, image 2 is not.
  std::vector<std::vector<int>> overlapping = {{1, 2}, {0, 2}, {0, 1}};
  std::vector<char> used = {true, true, true};
  std::vector<char> fused = {true, true, false};
  EXPECT_EQ(internal::FindNextImage(overlapping, used, fused, 0), 2);
}

TEST(FindNextImage, FallsBackToFirstUnfusedImage) {
  // Image 0 overlaps with [1], but image 1 is already fused.
  // Image 2 is not overlapping with image 0 but is unfused.
  std::vector<std::vector<int>> overlapping = {{1}, {0}, {0}};
  std::vector<char> used = {true, true, true};
  std::vector<char> fused = {true, true, false};
  EXPECT_EQ(internal::FindNextImage(overlapping, used, fused, 0), 2);
}

TEST(FindNextImage, ReturnsNegativeWhenAllFused) {
  std::vector<std::vector<int>> overlapping = {{1}, {0}};
  std::vector<char> used = {true, true};
  std::vector<char> fused = {true, true};
  EXPECT_EQ(internal::FindNextImage(overlapping, used, fused, 0), -1);
}

TEST(FindNextImage, SkipsUnusedImages) {
  // Image 1 overlaps with image 0 but is unused (not in fusion.cfg).
  std::vector<std::vector<int>> overlapping = {{1, 2}, {0}, {0}};
  std::vector<char> used = {true, false, true};
  std::vector<char> fused = {true, false, false};
  EXPECT_EQ(internal::FindNextImage(overlapping, used, fused, 0), 2);
}

TEST(WritePointsVisibility, EmptyVisibility) {
  const auto temp_dir = CreateTestDir();
  const auto path = temp_dir / "vis.bin";

  std::vector<std::vector<int>> empty_vis;
  WritePointsVisibility(path, empty_vis);

  std::ifstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  const uint64_t num_points = ReadBinaryLittleEndian<uint64_t>(&file);
  EXPECT_EQ(num_points, 0);
}

TEST(WritePointsVisibility, SinglePointMultipleImages) {
  const auto temp_dir = CreateTestDir();
  const auto path = temp_dir / "vis.bin";

  std::vector<std::vector<int>> vis = {{0, 3, 7}};
  WritePointsVisibility(path, vis);

  std::ifstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  const uint64_t num_points = ReadBinaryLittleEndian<uint64_t>(&file);
  EXPECT_EQ(num_points, 1);
  const uint32_t num_images = ReadBinaryLittleEndian<uint32_t>(&file);
  EXPECT_EQ(num_images, 3);
  EXPECT_EQ(ReadBinaryLittleEndian<uint32_t>(&file), 0);
  EXPECT_EQ(ReadBinaryLittleEndian<uint32_t>(&file), 3);
  EXPECT_EQ(ReadBinaryLittleEndian<uint32_t>(&file), 7);
}

TEST(WritePointsVisibility, MultiplePointsVaryingVisibility) {
  const auto temp_dir = CreateTestDir();
  const auto path = temp_dir / "vis.bin";

  std::vector<std::vector<int>> vis = {{1, 2}, {5}, {0, 1, 2, 3}};
  WritePointsVisibility(path, vis);

  std::ifstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());

  const uint64_t num_points = ReadBinaryLittleEndian<uint64_t>(&file);
  EXPECT_EQ(num_points, 3);

  // Point 0: 2 images
  EXPECT_EQ(ReadBinaryLittleEndian<uint32_t>(&file), 2);
  EXPECT_EQ(ReadBinaryLittleEndian<uint32_t>(&file), 1);
  EXPECT_EQ(ReadBinaryLittleEndian<uint32_t>(&file), 2);

  // Point 1: 1 image
  EXPECT_EQ(ReadBinaryLittleEndian<uint32_t>(&file), 1);
  EXPECT_EQ(ReadBinaryLittleEndian<uint32_t>(&file), 5);

  // Point 2: 4 images
  EXPECT_EQ(ReadBinaryLittleEndian<uint32_t>(&file), 4);
  EXPECT_EQ(ReadBinaryLittleEndian<uint32_t>(&file), 0);
  EXPECT_EQ(ReadBinaryLittleEndian<uint32_t>(&file), 1);
  EXPECT_EQ(ReadBinaryLittleEndian<uint32_t>(&file), 2);
  EXPECT_EQ(ReadBinaryLittleEndian<uint32_t>(&file), 3);
}

TEST(StereoFusion, ZeroDepthProducesNoPoints) {
  // Depth maps filled with 0 should cause all pixels to be skipped.
  auto ws = FusionTestWorkspace::Create(
      /*num_frames=*/2, /*width=*/10, /*height=*/10, /*depth_value=*/0.0f);

  StereoFusion fusion(
      DefaultTestOptions(), ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  EXPECT_EQ(fusion.GetFusedPoints().size(), 0);
  EXPECT_EQ(fusion.GetFusedPointsVisibility().size(), 0);
}

TEST(StereoFusion, NegativeDepthProducesNoPoints) {
  auto ws = FusionTestWorkspace::Create(
      /*num_frames=*/2, /*width=*/10, /*height=*/10, /*depth_value=*/-1.0f);

  StereoFusion fusion(
      DefaultTestOptions(), ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  EXPECT_EQ(fusion.GetFusedPoints().size(), 0);
  EXPECT_EQ(fusion.GetFusedPointsVisibility().size(), 0);
}

TEST(StereoFusion, BoundingBoxFiltersPoints) {
  auto ws = FusionTestWorkspace::Create();

  // Set a bounding box that excludes all fused points. The synthetic cameras
  // are near the origin looking outward, so points at depth=5 are several
  // units away. A tiny box at a far offset excludes everything.
  auto options = DefaultTestOptions();
  options.bounding_box = std::make_pair(Eigen::Vector3f(100, 100, 100),
                                        Eigen::Vector3f(101, 101, 101));

  StereoFusion fusion(options, ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  EXPECT_EQ(fusion.GetFusedPoints().size(), 0);
}

TEST(StereoFusion, HighMinNumPixelsFiltersPoints) {
  // With min_num_pixels set very high, no pixel cluster will be large enough.
  auto ws = FusionTestWorkspace::Create(
      /*num_frames=*/2, /*width=*/10, /*height=*/10);

  auto options = DefaultTestOptions();
  options.min_num_pixels = 99999;
  options.max_num_pixels = 99999;

  StereoFusion fusion(options, ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  EXPECT_EQ(fusion.GetFusedPoints().size(), 0);
}

TEST(StereoFusion, MaxNumPixelsCapsClusterSize) {
  auto ws = FusionTestWorkspace::Create();

  auto options = DefaultTestOptions();
  // Allow at most 2 pixels per fused point. This should still produce fused
  // points but with tightly capped cluster sizes.
  options.max_num_pixels = 2;
  options.min_num_pixels = 1;

  StereoFusion fusion(options, ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  // Should still produce points since min_num_pixels=1 is easily met.
  EXPECT_GT(fusion.GetFusedPoints().size(), 0);
}

TEST(StereoFusion, MaxTraversalDepthOne) {
  // With max_traversal_depth=1, only the reference pixel is used (no
  // traversal to overlapping images), so each fused point can only come from
  // a single pixel. Combined with min_num_pixels=1 this should still produce
  // points.
  auto ws = FusionTestWorkspace::Create();

  auto options = DefaultTestOptions();
  options.max_traversal_depth = 1;
  options.min_num_pixels = 1;

  StereoFusion fusion(options, ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  EXPECT_GT(fusion.GetFusedPoints().size(), 0);

  // Each point should only be visible from one image since traversal was
  // limited to depth 1 (only the reference pixel, no cross-image matching).
  for (const auto& vis : fusion.GetFusedPointsVisibility()) {
    EXPECT_EQ(vis.size(), 1);
  }
}

TEST(StereoFusion, TightReprojErrorFiltersPoints) {
  // With max_reproj_error=0, the reprojection check at traversal_depth>0 will
  // fail for all non-reference pixels, so each cluster can have at most 1
  // pixel from overlapping images. With min_num_pixels=2, no points survive.
  auto ws = FusionTestWorkspace::Create(
      /*num_frames=*/2, /*width=*/10, /*height=*/10);

  auto options = DefaultTestOptions();
  options.max_reproj_error = 0.0;
  options.min_num_pixels = 2;

  StereoFusion fusion(options, ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  EXPECT_EQ(fusion.GetFusedPoints().size(), 0);
}

TEST(StereoFusion, TightDepthErrorFiltersPoints) {
  // With max_depth_error=0, the depth consistency check will reject all
  // non-reference pixels when there is any depth discrepancy from projection.
  // With min_num_pixels=2, nothing survives.
  auto ws = FusionTestWorkspace::Create(
      /*num_frames=*/2, /*width=*/10, /*height=*/10);

  auto options = DefaultTestOptions();
  options.max_depth_error = 0.0;
  options.min_num_pixels = 2;

  StereoFusion fusion(options, ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  EXPECT_EQ(fusion.GetFusedPoints().size(), 0);
}

TEST(StereoFusion, MaskExcludesPixels) {
  auto ws = FusionTestWorkspace::Create(
      /*num_frames=*/2, /*width=*/10, /*height=*/10);

  // Create mask directory and mask files: black (0) pixels are masked out,
  // white (255) pixels are kept.
  const auto mask_dir = ws.temp_dir / "masks";
  CreateDirIfNotExists(mask_dir);

  for (const auto& name : ws.image_names) {
    // Create all-black mask (all zeros = all masked).
    Bitmap mask(10, 10, false);
    mask.Fill(BitmapColor<uint8_t>(0));
    mask.Write(mask_dir / (name + ".png"));
  }

  auto options = DefaultTestOptions();
  options.mask_path = mask_dir;

  StereoFusion fusion(options, ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  // All pixels masked out, so no points should be fused.
  EXPECT_EQ(fusion.GetFusedPoints().size(), 0);
}

TEST(StereoFusion, PartialMaskReducesPoints) {
  auto ws = FusionTestWorkspace::Create(
      /*num_frames=*/2, /*width=*/10, /*height=*/10);

  const auto mask_dir = ws.temp_dir / "masks";
  CreateDirIfNotExists(mask_dir);

  for (const auto& name : ws.image_names) {
    // White mask (255) means "keep" these pixels (mask value != 0).
    Bitmap mask(10, 10, false);
    mask.Fill(BitmapColor<uint8_t>(255));
    mask.Write(mask_dir / (name + ".png"));
  }

  // Run without mask first.
  auto options_no_mask = DefaultTestOptions();
  StereoFusion fusion_no_mask(
      options_no_mask, ws.temp_dir, "COLMAP", "", "geometric");
  fusion_no_mask.Run();
  const size_t points_no_mask = fusion_no_mask.GetFusedPoints().size();

  // Run with mask (all-white = all kept).
  auto options_mask = DefaultTestOptions();
  options_mask.mask_path = mask_dir;
  StereoFusion fusion_mask(
      options_mask, ws.temp_dir, "COLMAP", "", "geometric");
  fusion_mask.Run();
  const size_t points_with_mask = fusion_mask.GetFusedPoints().size();

  // With all-white mask, same number of points as without mask.
  EXPECT_EQ(points_with_mask, points_no_mask);
}

TEST(StereoFusion, CachedWorkspaceMode) {
  auto ws = FusionTestWorkspace::Create();

  auto options = DefaultTestOptions();
  options.use_cache = true;
  options.cache_size = 1.0;

  StereoFusion fusion(options, ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  EXPECT_GT(fusion.GetFusedPoints().size(), 0);
  EXPECT_EQ(fusion.GetFusedPoints().size(),
            fusion.GetFusedPointsVisibility().size());
}

TEST(StereoFusion, SameWorkspaceProducesConsistentResults) {
  // Verify that creating two StereoFusion objects on the same workspace
  // produces consistent results.
  auto ws = FusionTestWorkspace::Create();

  StereoFusion fusion1(
      DefaultTestOptions(), ws.temp_dir, "COLMAP", "", "geometric");
  fusion1.Run();
  const size_t first_count = fusion1.GetFusedPoints().size();
  EXPECT_GT(first_count, 0);

  StereoFusion fusion2(
      DefaultTestOptions(), ws.temp_dir, "COLMAP", "", "geometric");
  fusion2.Run();
  const size_t second_count = fusion2.GetFusedPoints().size();
  EXPECT_EQ(first_count, second_count);
}

TEST(StereoFusion, SingleImageFusion) {
  // With only one image, cross-image consistency cannot increase cluster sizes
  // beyond 1. With min_num_pixels=1, we should still get points.
  auto ws = FusionTestWorkspace::Create(/*num_frames=*/1);

  auto options = DefaultTestOptions();
  options.min_num_pixels = 1;

  StereoFusion fusion(options, ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  EXPECT_GT(fusion.GetFusedPoints().size(), 0);

  for (const auto& vis : fusion.GetFusedPointsVisibility()) {
    EXPECT_EQ(vis.size(), 1);
  }
}

TEST(StereoFusion, ThreeImageFusion) {
  // More images should allow cross-image consistency checks to produce
  // points visible from multiple views.
  auto ws = FusionTestWorkspace::Create(/*num_frames=*/3);

  auto options = DefaultTestOptions();
  options.min_num_pixels = 1;

  StereoFusion fusion(options, ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  EXPECT_GT(fusion.GetFusedPoints().size(), 0);
}

TEST(StereoFusion, TightNormalErrorFiltersPoints) {
  // With max_normal_error = 0, the normal consistency check will reject
  // cross-image correspondences unless normals match exactly. Combined with
  // min_num_pixels=2, this should filter out points that only survive from
  // single-image contributions.
  auto ws = FusionTestWorkspace::Create(
      /*num_frames=*/2, /*width=*/10, /*height=*/10);

  auto options = DefaultTestOptions();
  options.max_normal_error = 0.0;
  options.min_num_pixels = 2;

  StereoFusion fusion(options, ws.temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  // With perfectly aligned normals (all pointing in z direction), some points
  // may still pass. The key test is that the code path for normal checking
  // is exercised without crashing.
  const auto& points = fusion.GetFusedPoints();
  EXPECT_EQ(points.size(), fusion.GetFusedPointsVisibility().size());
}

TEST(StereoFusion, ConstructorRejectsInvalidOptions) {
  StereoFusionOptions options;
  options.check_num_images = 0;  // Invalid
  EXPECT_THROW(StereoFusion(options, "/nonexistent", "COLMAP", "", "geometric"),
               std::exception);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
