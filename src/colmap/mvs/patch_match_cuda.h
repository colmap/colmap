// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/mvs/cuda_texture.h"
#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/gpu_mat.h"
#include "colmap/mvs/gpu_mat_prng.h"
#include "colmap/mvs/gpu_mat_ref_image.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match.h"

#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

namespace colmap {
namespace mvs {

class PatchMatchCuda {
 public:
  PatchMatchCuda(const PatchMatchOptions& options,
                 const PatchMatch::Problem& problem);

  void Run();

  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  Mat<float> GetSelProbMap() const;
  std::vector<int> GetConsistentImageIdxs() const;

 private:
  template <int kWindowSize, int kWindowStep>
  void RunWithWindowSizeAndStep();

  void ComputeCudaConfig();

  void BindRefImageTexture();

  void InitRefImage();
  void InitSourceImages();
  void InitTransforms();
  void InitWorkspaceMemory();

  // Rotate reference image by 90 degrees in counter-clockwise direction.
  void Rotate();

  const PatchMatchOptions options_;
  const PatchMatch::Problem problem_;

  // Dimensions for sweeping from top to bottom, i.e. one thread per column.
  dim3 sweep_block_size_;
  dim3 sweep_grid_size_;
  // Dimensions for element-wise operations, i.e. one thread per pixel.
  dim3 elem_wise_block_size_;
  dim3 elem_wise_grid_size_;

  // Original (not rotated) dimension of reference image.
  size_t ref_width_;
  size_t ref_height_;

  // Rotation of reference image in pi/2. This is equivalent to the number of
  // calls to `rotate` mod 4.
  int rotation_in_half_pi_;

  // Reference and source image input data.
  std::unique_ptr<CudaArrayLayeredTexture<uint8_t>> ref_image_texture_;
  std::unique_ptr<CudaArrayLayeredTexture<uint8_t>> src_images_texture_;
  std::unique_ptr<CudaArrayLayeredTexture<float>> src_depth_maps_texture_;

  // Relative poses from rotated versions of reference image to source images
  // corresponding to _rotationInHalfPi:
  //
  //    [S(1), S(2), S(3), ..., S(n)]
  //
  // where n is the number of source images and:
  //
  //    S(i) = [K_i(0, 0), K_i(0, 2), K_i(1, 1), K_i(1, 2), R_i(:), T_i(:)
  //            C_i(:), P(:), P^-1(:)]
  //
  // where i denotes the index of the source image and K is its calibration.
  // R, T, C, P, P^-1 denote the relative rotation, translation, camera
  // center, projection, and inverse projection from there reference to the
  // i-th source image.
  std::unique_ptr<CudaArrayLayeredTexture<float>> poses_texture_[4];

  // Calibration matrix for rotated versions of reference image
  // as {K[0, 0], K[0, 2], K[1, 1], K[1, 2]} corresponding to _rotationInHalfPi.
  float ref_K_host_[4][4];
  float ref_inv_K_host_[4][4];

  // Data for reference image.
  std::unique_ptr<GpuMatRefImage> ref_image_;
  std::unique_ptr<GpuMat<float>> depth_map_;
  std::unique_ptr<GpuMat<float>> normal_map_;
  std::unique_ptr<GpuMat<float>> sel_prob_map_;
  std::unique_ptr<GpuMat<float>> prev_sel_prob_map_;
  std::unique_ptr<GpuMat<float>> cost_map_;
  std::unique_ptr<GpuMatPRNG> rand_state_map_;
  std::unique_ptr<GpuMat<uint8_t>> consistency_mask_;

  // Shared memory is too small to hold local state for each thread,
  // so this is workspace memory in global memory.
  std::unique_ptr<GpuMat<float>> global_workspace_;
};

}  // namespace mvs
}  // namespace colmap
