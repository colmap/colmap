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

#include "colmap/mvs/texture_mapping.h"

#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

// Create a 2-triangle quad mesh (4 vertices, 2 faces).
// Quad in XY plane at Z=0, from (0,0,0) to (1,1,0).
PlyMesh MakeTriangleMesh() {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 1.0f, 0.0f),
      PlyMeshVertex(0.0f, 1.0f, 0.0f),
  };
  mesh.faces = {
      PlyMeshFace(0, 1, 2),
      PlyMeshFace(0, 2, 3),
  };
  return mesh;
}

// Create a camera looking down -Z at the XY plane.
// Camera at (0.5, 0.5, 5), looking toward origin along -Z in world.
// In COLMAP convention: P_cam = R * P_world + T, camera looks along +Z_cam.
// To look along -Z_world, use 180° rotation around X: R maps -Z_world to
// +Z_cam.
Image MakeTestImage(int width, int height, const BitmapColor<uint8_t>& color) {
  // K: focal length = width, principal point at center.
  float K[9] = {0};
  K[0] = static_cast<float>(width);          // fx
  K[4] = static_cast<float>(height);         // fy
  K[2] = static_cast<float>(width) / 2.0f;   // cx
  K[5] = static_cast<float>(height) / 2.0f;  // cy
  K[8] = 1.0f;

  // R: 180° rotation around X-axis. Row-major: [1,0,0, 0,-1,0, 0,0,-1].
  // This makes the camera look along -Z in world, with +Y up.
  float R[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};

  // C = (0.5, 0.5, 5), T = -R*C.
  // T = -[1,0,0;0,-1,0;0,0,-1] * [0.5,0.5,5] = [-0.5, 0.5, 5]
  float T[3] = {-0.5f, 0.5f, 5.0f};

  Image image("test_image.png",
              static_cast<size_t>(width),
              static_cast<size_t>(height),
              K,
              R,
              T);

  Bitmap bitmap(width, height, /*as_rgb=*/true);
  bitmap.Fill(color);
  image.SetBitmap(bitmap);

  return image;
}

// Create a camera where the mesh at Z=0 is behind the camera.
// Camera at (0.5, 0.5, -5), looking along -Z in world (away from mesh).
Image MakeBehindCameraImage(int width, int height) {
  float K[9] = {0};
  K[0] = static_cast<float>(width);
  K[4] = static_cast<float>(height);
  K[2] = static_cast<float>(width) / 2.0f;
  K[5] = static_cast<float>(height) / 2.0f;
  K[8] = 1.0f;

  // R: 180° around X, looks along -Z in world.
  float R[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};

  // C = (0.5, 0.5, -5), T = -R*C = [-0.5, 0.5, -5]
  float T[3] = {-0.5f, 0.5f, -5.0f};

  Image image("behind.png",
              static_cast<size_t>(width),
              static_cast<size_t>(height),
              K,
              R,
              T);

  Bitmap bitmap(width, height, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(0));
  image.SetBitmap(bitmap);

  return image;
}

// Create a camera that views the mesh at a very grazing angle.
// Camera nearly in-plane with XY mesh, at (0.5, 0.5, 0.01), looking along -Z.
Image MakeGrazingAngleImage(int width, int height) {
  float K[9] = {0};
  K[0] = static_cast<float>(width);
  K[4] = static_cast<float>(height);
  K[2] = static_cast<float>(width) / 2.0f;
  K[5] = static_cast<float>(height) / 2.0f;
  K[8] = 1.0f;

  // R: 180° around X.
  float R[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};

  // C = (0.5, 0.5, 0.01), T = -R*C = [-0.5, 0.5, 0.01]
  float T[3] = {-0.5f, 0.5f, 0.01f};

  Image image("grazing.png",
              static_cast<size_t>(width),
              static_cast<size_t>(height),
              K,
              R,
              T);

  Bitmap bitmap(width, height, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(128));
  image.SetBitmap(bitmap);

  return image;
}

// Create a 4-face pyramid mesh.
PlyMesh MakePyramidMesh() {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 1.0f, 0.0f),
      PlyMeshVertex(0.0f, 1.0f, 0.0f),
      PlyMeshVertex(0.5f, 0.5f, 1.0f),
  };
  mesh.faces = {
      PlyMeshFace(0, 1, 4),
      PlyMeshFace(1, 2, 4),
      PlyMeshFace(2, 3, 4),
      PlyMeshFace(3, 0, 4),
  };
  return mesh;
}

// Create a 12-face cube mesh (2 triangles per face).
PlyMesh MakeCubeMesh() {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0, 0, 0),
      PlyMeshVertex(1, 0, 0),
      PlyMeshVertex(1, 1, 0),
      PlyMeshVertex(0, 1, 0),
      PlyMeshVertex(0, 0, 1),
      PlyMeshVertex(1, 0, 1),
      PlyMeshVertex(1, 1, 1),
      PlyMeshVertex(0, 1, 1),
  };
  mesh.faces = {
      // Front (Z=0): 0,1,2 and 0,2,3
      PlyMeshFace(0, 1, 2),
      PlyMeshFace(0, 2, 3),
      // Back (Z=1): 4,6,5 and 4,7,6
      PlyMeshFace(4, 6, 5),
      PlyMeshFace(4, 7, 6),
      // Left (X=0): 0,3,7 and 0,7,4
      PlyMeshFace(0, 3, 7),
      PlyMeshFace(0, 7, 4),
      // Right (X=1): 1,5,6 and 1,6,2
      PlyMeshFace(1, 5, 6),
      PlyMeshFace(1, 6, 2),
      // Bottom (Y=0): 0,4,5 and 0,5,1
      PlyMeshFace(0, 4, 5),
      PlyMeshFace(0, 5, 1),
      // Top (Y=1): 3,2,6 and 3,6,7
      PlyMeshFace(3, 2, 6),
      PlyMeshFace(3, 6, 7),
  };
  return mesh;
}

TEST(MeshTextureMapping, EndToEnd) {
  const PlyMesh mesh = MakeTriangleMesh();
  const BitmapColor<uint8_t> red(200, 50, 50);
  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, red));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  const auto result = MeshTextureMapping(mesh, images, options);

  // Check basic properties.
  EXPECT_GT(result.atlas_width, 0);
  EXPECT_GT(result.atlas_height, 0);
  EXPECT_EQ(result.face_uvs.size(), mesh.faces.size() * 6);
  EXPECT_EQ(result.face_view_ids.size(), mesh.faces.size());

  // Both faces should be assigned to view 0.
  EXPECT_EQ(result.face_view_ids[0], 0);
  EXPECT_EQ(result.face_view_ids[1], 0);

  // All UVs should be in [0, 1].
  for (size_t i = 0; i < result.face_uvs.size(); ++i) {
    EXPECT_GE(result.face_uvs[i], 0.0f);
    EXPECT_LE(result.face_uvs[i], 1.0f);
  }

  // Atlas should contain approximately the red color.
  // Check a pixel that should be baked.
  bool found_colored_pixel = false;
  for (int y = 0; y < result.atlas_height && !found_colored_pixel; ++y) {
    for (int x = 0; x < result.atlas_width && !found_colored_pixel; ++x) {
      const auto color = result.texture_atlas.GetPixel(x, y);
      if (color && color->r > 100 && color->g < 150 && color->b < 150) {
        found_colored_pixel = true;
      }
    }
  }
  EXPECT_TRUE(found_colored_pixel);
}

TEST(MeshTextureMapping, EmptyMesh) {
  PlyMesh mesh;
  std::vector<Image> images;
  images.push_back(MakeTestImage(64, 64, BitmapColor<uint8_t>(128)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;

  const auto result = MeshTextureMapping(mesh, images, options);

  EXPECT_EQ(result.atlas_width, 0);
  EXPECT_EQ(result.atlas_height, 0);
  EXPECT_TRUE(result.face_uvs.empty());
  EXPECT_TRUE(result.face_view_ids.empty());
}

TEST(MeshTextureMapping, NoVisibleFaces) {
  const PlyMesh mesh = MakeTriangleMesh();
  std::vector<Image> images;
  images.push_back(MakeBehindCameraImage(256, 256));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;

  const auto result = MeshTextureMapping(mesh, images, options);

  // All faces should be unassigned.
  for (size_t i = 0; i < result.face_view_ids.size(); ++i) {
    EXPECT_EQ(result.face_view_ids[i], -1);
  }
}

TEST(MeshTextureMapping, SingleFaceSingleView) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, BitmapColor<uint8_t>(100, 200, 50)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  const auto result = MeshTextureMapping(mesh, images, options);
  EXPECT_EQ(result.face_view_ids[0], 0);
}

TEST(MeshTextureMapping, SingleFaceTwoViews) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  // Image 0: closer camera -> larger projected area.
  auto img_close = MakeTestImage(256, 256, BitmapColor<uint8_t>(255, 0, 0));

  // Image 1: farther camera -> smaller projected area.
  float K[9] = {0};
  K[0] = 256.0f;
  K[4] = 256.0f;
  K[2] = 128.0f;
  K[5] = 128.0f;
  K[8] = 1.0f;
  // R: 180° around X, same orientation as MakeTestImage.
  float R[9] = {1, 0, 0, 0, -1, 0, 0, 0, -1};
  // C = (0.5, 0.5, 20), T = -R*C = [-0.5, 0.5, 20]
  float T[3] = {-0.5f, 0.5f, 20.0f};
  Image img_far("far.png", 256, 256, K, R, T);
  Bitmap bitmap(256, 256, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(0, 255, 0));
  img_far.SetBitmap(bitmap);

  std::vector<Image> images = {img_close, img_far};

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  const auto result = MeshTextureMapping(mesh, images, options);

  // Should pick the closer camera (view 0) as it has larger projected area.
  EXPECT_EQ(result.face_view_ids[0], 0);
}

TEST(MeshTextureMapping, FaceBehindCamera) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  std::vector<Image> images;
  images.push_back(MakeBehindCameraImage(256, 256));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;

  const auto result = MeshTextureMapping(mesh, images, options);
  EXPECT_EQ(result.face_view_ids[0], -1);
}

TEST(MeshTextureMapping, GrazingAngleRejected) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  std::vector<Image> images;
  images.push_back(MakeGrazingAngleImage(256, 256));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  // Set high threshold to reject grazing angles.
  options.min_cos_normal_angle = 0.9;

  const auto result = MeshTextureMapping(mesh, images, options);
  EXPECT_EQ(result.face_view_ids[0], -1);
}

TEST(MeshTextureMapping, NeighborSmoothing) {
  // Create a strip mesh centered around (0.5, 0.5) so all faces are visible.
  // Smaller extent so all vertices project within image bounds.
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.0f, 1.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
      PlyMeshVertex(1.0f, 1.0f, 0.0f),
  };
  mesh.faces = {
      PlyMeshFace(0, 1, 4),
      PlyMeshFace(0, 4, 3),
      PlyMeshFace(1, 2, 5),
      PlyMeshFace(1, 5, 4),
  };

  // Both views see all faces.
  auto img0 = MakeTestImage(512, 512, BitmapColor<uint8_t>(255, 0, 0));
  auto img1 = MakeTestImage(512, 512, BitmapColor<uint8_t>(0, 255, 0));

  std::vector<Image> images = {img0, img1};

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 3;
  options.inpaint_radius = 0;

  const auto result = MeshTextureMapping(mesh, images, options);

  // After smoothing, all faces should be assigned since they're all visible.
  int assigned = 0;
  for (size_t i = 0; i < result.face_view_ids.size(); ++i) {
    if (result.face_view_ids[i] >= 0) ++assigned;
  }
  EXPECT_EQ(assigned, static_cast<int>(mesh.faces.size()));

  // All assigned faces should have the same view (smoothing promotes
  // uniformity).
  int first_view = result.face_view_ids[0];
  for (size_t i = 1; i < result.face_view_ids.size(); ++i) {
    EXPECT_EQ(result.face_view_ids[i], first_view);
  }
}

TEST(MeshTextureMapping, CubeAdjacency) {
  // Verify the cube mesh produces faces with proper adjacency
  // by running the full pipeline.
  const PlyMesh mesh = MakeCubeMesh();
  std::vector<Image> images;

  // Camera looking at front face.
  images.push_back(MakeTestImage(512, 512, BitmapColor<uint8_t>(200, 100, 50)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;
  options.min_visible_vertices = 1;

  const auto result = MeshTextureMapping(mesh, images, options);

  // At least the front-facing triangles should be assigned.
  EXPECT_EQ(result.face_view_ids.size(), 12u);
  int assigned = 0;
  for (int v : result.face_view_ids) {
    if (v >= 0) ++assigned;
  }
  EXPECT_GT(assigned, 0);
}

TEST(MeshTextureMapping, UVRange) {
  const PlyMesh mesh = MakeTriangleMesh();
  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, BitmapColor<uint8_t>(128)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  const auto result = MeshTextureMapping(mesh, images, options);

  for (size_t i = 0; i < result.face_uvs.size(); ++i) {
    EXPECT_GE(result.face_uvs[i], 0.0f) << "UV index " << i;
    EXPECT_LE(result.face_uvs[i], 1.0f) << "UV index " << i;
  }
}

TEST(MeshTextureMapping, BakeSolidColor) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  const BitmapColor<uint8_t> src_color(180, 90, 45);
  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, src_color));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  const auto result = MeshTextureMapping(mesh, images, options);

  // Find baked pixels and verify they match the source color.
  int baked_count = 0;
  for (int y = 0; y < result.atlas_height; ++y) {
    for (int x = 0; x < result.atlas_width; ++x) {
      const auto color = result.texture_atlas.GetPixel(x, y);
      ASSERT_TRUE(color.has_value());
      if (color->r != 0 || color->g != 0 || color->b != 0) {
        // Due to bilinear interpolation, allow some tolerance.
        EXPECT_NEAR(color->r, src_color.r, 5);
        EXPECT_NEAR(color->g, src_color.g, 5);
        EXPECT_NEAR(color->b, src_color.b, 5);
        ++baked_count;
      }
    }
  }
  EXPECT_GT(baked_count, 0);
}

TEST(MeshTextureMapping, InpaintFillsNearbyPixels) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, BitmapColor<uint8_t>(200, 100, 50)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 3;

  const auto result_inpaint = MeshTextureMapping(mesh, images, options);

  // Without inpainting for comparison.
  options.inpaint_radius = 0;
  const auto result_no_inpaint = MeshTextureMapping(mesh, images, options);

  // Count non-black pixels. Inpainted result should have more.
  int count_inpaint = 0, count_no_inpaint = 0;
  for (int y = 0; y < result_inpaint.atlas_height; ++y) {
    for (int x = 0; x < result_inpaint.atlas_width; ++x) {
      const auto color = result_inpaint.texture_atlas.GetPixel(x, y);
      if (color && (color->r != 0 || color->g != 0 || color->b != 0)) {
        ++count_inpaint;
      }
    }
  }
  for (int y = 0; y < result_no_inpaint.atlas_height; ++y) {
    for (int x = 0; x < result_no_inpaint.atlas_width; ++x) {
      const auto color = result_no_inpaint.texture_atlas.GetPixel(x, y);
      if (color && (color->r != 0 || color->g != 0 || color->b != 0)) {
        ++count_no_inpaint;
      }
    }
  }
  EXPECT_GT(count_inpaint, count_no_inpaint);
}

TEST(MeshTextureMapping, InpaintDoesNotOverwriteBaked) {
  PlyMesh mesh;
  mesh.vertices = {
      PlyMeshVertex(0.0f, 0.0f, 0.0f),
      PlyMeshVertex(1.0f, 0.0f, 0.0f),
      PlyMeshVertex(0.5f, 1.0f, 0.0f),
  };
  mesh.faces = {PlyMeshFace(0, 1, 2)};

  const BitmapColor<uint8_t> src_color(200, 100, 50);
  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, src_color));

  // Run without inpainting.
  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;
  const auto result_base = MeshTextureMapping(mesh, images, options);

  // Run with inpainting.
  options.inpaint_radius = 5;
  const auto result_inpaint = MeshTextureMapping(mesh, images, options);

  // Check that baked pixels are unchanged.
  for (int y = 0; y < result_base.atlas_height; ++y) {
    for (int x = 0; x < result_base.atlas_width; ++x) {
      const auto base_color = result_base.texture_atlas.GetPixel(x, y);
      const auto inpaint_color = result_inpaint.texture_atlas.GetPixel(x, y);
      ASSERT_TRUE(base_color.has_value());
      ASSERT_TRUE(inpaint_color.has_value());
      if (base_color->r != 0 || base_color->g != 0 || base_color->b != 0) {
        EXPECT_EQ(base_color->r, inpaint_color->r);
        EXPECT_EQ(base_color->g, inpaint_color->g);
        EXPECT_EQ(base_color->b, inpaint_color->b);
      }
    }
  }
}

TEST(MeshTextureMapping, GlobalColorCorrectionNoSeams) {
  // Single view, single region -> no seams -> no correction applied.
  PlyMesh mesh = MakeTriangleMesh();
  std::vector<Image> images;
  images.push_back(MakeTestImage(256, 256, BitmapColor<uint8_t>(150)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = true;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;

  const auto result_with_cc = MeshTextureMapping(mesh, images, options);

  options.apply_color_correction = false;
  const auto result_without_cc = MeshTextureMapping(mesh, images, options);

  // Results should be identical since there are no seams.
  EXPECT_EQ(result_with_cc.atlas_width, result_without_cc.atlas_width);
  EXPECT_EQ(result_with_cc.atlas_height, result_without_cc.atlas_height);

  for (int y = 0; y < result_with_cc.atlas_height; ++y) {
    for (int x = 0; x < result_with_cc.atlas_width; ++x) {
      const auto c1 = result_with_cc.texture_atlas.GetPixel(x, y);
      const auto c2 = result_without_cc.texture_atlas.GetPixel(x, y);
      ASSERT_TRUE(c1.has_value());
      ASSERT_TRUE(c2.has_value());
      EXPECT_EQ(c1->r, c2->r);
      EXPECT_EQ(c1->g, c2->g);
      EXPECT_EQ(c1->b, c2->b);
    }
  }
}

TEST(MeshTextureMapping, PyramidMultiFace) {
  const PlyMesh mesh = MakePyramidMesh();
  std::vector<Image> images;
  images.push_back(
      MakeTestImage(512, 512, BitmapColor<uint8_t>(100, 150, 200)));

  MeshTextureMappingOptions options;
  options.apply_color_correction = false;
  options.view_selection_smoothing_iterations = 0;
  options.inpaint_radius = 0;
  options.min_visible_vertices = 1;

  const auto result = MeshTextureMapping(mesh, images, options);
  EXPECT_EQ(result.face_view_ids.size(), 4u);
  EXPECT_EQ(result.face_uvs.size(), 24u);

  // At least some faces should be assigned.
  int assigned = 0;
  for (int v : result.face_view_ids) {
    if (v >= 0) ++assigned;
  }
  EXPECT_GT(assigned, 0);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
