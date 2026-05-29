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

#pragma once

#include "colmap/mvs/image.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/ply.h"

#include <vector>

namespace colmap {
namespace mvs {

struct MeshTextureMappingOptions {
  // Minimum cosine of angle between face normal and view direction.
  // Faces viewed at more grazing angles are rejected.
  double min_cos_normal_angle = 0.1;

  // Minimum number of face vertices that must project inside the image
  // for a view to be considered a candidate (1, 2, or 3).
  int min_visible_vertices = 3;

  // Number of neighbor-smoothing iterations for view selection.
  // Reduces fragmentation by swapping face labels to match neighbors.
  int view_selection_smoothing_iterations = 3;

  // Padding in pixels between atlas patches.
  int atlas_patch_padding = 2;

  // Number of pixels to dilate/inpaint around baked regions.
  int inpaint_radius = 5;

  // Whether to apply global color correction (Waechter et al. 2014).
  bool apply_color_correction = true;

  // Regularization weight for the color correction system.
  double color_correction_regularization = 0.1;

  // Number of threads (default: all available).
  int num_threads = -1;

  // Scale factor for the texture atlas resolution.
  // 1.0 = native source-image resolution (default).
  // < 1.0 = lower resolution (e.g. 0.5 = half).
  // > 1.0 = higher resolution (e.g. 2.0 = double).
  double texture_scale_factor = 1.0;

  bool Check() const;
  void Print() const;
};

struct MeshTextureMappingResult {
  // The texture atlas image (RGB).
  Bitmap texture_atlas;

  // Per-face UV coordinates: 3 UVs per face (wedge/per-corner UVs).
  // face_uvs[face_idx * 6 + vertex_in_face * 2 + 0] = u
  // face_uvs[face_idx * 6 + vertex_in_face * 2 + 1] = v
  // UV range: [0, 1], (0,0) = bottom-left, (1,1) = top-right.
  std::vector<float> face_uvs;

  // Per-face view assignment: index into the images vector.
  // -1 means no view was assigned (face not textured).
  std::vector<int> face_view_ids;

  // Atlas dimensions.
  int atlas_width = 0;
  int atlas_height = 0;
};

// Produce a texture atlas with UV coordinates for a triangle mesh
// given calibrated multi-view images.
//
// Based on: Waechter, M., Moehrle, N., and Goesele, M.,
// "Let there be color! Large-scale texturing of 3D reconstructions,"
// European Conference on Computer Vision (ECCV), 2014.
MeshTextureMappingResult MeshTextureMapping(
    const PlyMesh& mesh,
    const std::vector<Image>& images,
    const MeshTextureMappingOptions& options);

}  // namespace mvs
}  // namespace colmap
