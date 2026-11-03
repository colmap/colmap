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

#include "colmap/feature/extractor.h"

#include <memory>

namespace colmap {

// Create a cusift feature extractor: a CUDA SIFT extractor that replicates
// COLMAP's own VLFeat CPU backend rather than SiftGPU, so it can serve as a
// commercially-usable replacement for SiftGPU's non-commercial license.
//
// It is configured entirely by SiftExtractionOptions (options.sift) and emits
// descriptors tagged FeatureExtractorType::SIFT, because the features it
// produces are SIFT features in COLMAP's usual UBC layout and are meant to be
// consumed by the existing SIFT matchers and databases unchanged. The options
// it cannot serve are rejected loudly rather than silently ignored - see
// src/thirdparty/cusift/README.md and the option checks in cusift.cc.
//
// Requires CUDA and a GPU; returns nullptr if COLMAP was built without CUDA.
std::unique_ptr<FeatureExtractor> CreateCusiftFeatureExtractor(
    const FeatureExtractionOptions& options);

// Number of octaves cusift builds for a width x height image. It follows
// SiftGPU as COLMAP runs it, not SiftExtractionOptions::num_octaves: COLMAP
// passes no octave count to SiftGPU, so SiftGPU builds every octave down to an
// 8-16 pixel image, floor(log2(min(width, height))) - 3 counted on the image
// after first_octave's upsampling (PyramidCU::ResizePyramid), and at least 1.
// Matching it keeps the large-scale features that SiftGPU relies on in blurred
// images, where fine-scale features vanish. COLMAP's CPU (VLFeat) SIFT does use
// num_octaves, so at its default of 4 it stops at a smaller maximum scale.
int CusiftNumOctaves(int width, int height, int first_octave);

// Result of describing externally supplied keypoints; see
// ExtractCusiftDescriptorsForKeypoints below.
struct CusiftDescriptorOnlyResult {
  // One row per ACCEPTED keypoint, normalized and quantized exactly as the
  // extractor would have done, and tagged FeatureExtractorType::SIFT.
  FeatureDescriptors descriptors;
  // Index into the input keypoints for each row of `descriptors`, so callers
  // can align against their own reference rows. Ascending.
  std::vector<size_t> keypoint_indices;
  // Keypoints whose sigma places them in VLFeat octave -1. The underlying
  // isolation path is non-upsampled only and would silently clamp them to
  // octave 0, describing them from the wrong pyramid level, so they are
  // rejected instead. At COLMAP's default first_octave = -1 this is a large
  // fraction of a real image's features.
  size_t num_rejected_octave_minus1 = 0;
  // Keypoints whose descriptor window fell outside the image, detected as a
  // zero-norm descriptor rather than guessed from a support radius. Rejected
  // so they cannot zero-fill into a caller's statistics.
  size_t num_rejected_degenerate = 0;
};

// Computes cusift descriptors for keypoints supplied by the caller, skipping
// detection and orientation entirely. This isolates descriptor-kernel fidelity:
// feed one extractor's (x, y, scale, orientation) and compare the descriptors
// it produces against another extractor's for the same features, with detection
// differences removed as a variable.
//
// Keypoint x/y are interpreted in COLMAP's stored convention (pixel centre at
// 0.5) and converted to the kernel's VLFeat frame internally, and orientation
// is taken as VLFeat-convention radians via
// FeatureKeypoint::ComputeOrientation. To describe SiftGPU-sourced keypoints,
// negate their stored orientation before calling, because COLMAP stores SiftGPU
// theta mirrored (sift.cc:716-719).
//
// Note that the pyramid level each descriptor samples from is re-derived from
// sigma, which is cusift's own decision and may differ from the source
// extractor's at scale boundaries. That is part of what is being measured, but
// residual disagreement should be checked against level-boundary proximity
// before being attributed to the descriptor kernel.
CusiftDescriptorOnlyResult ExtractCusiftDescriptorsForKeypoints(
    const Bitmap& bitmap,
    const FeatureKeypoints& keypoints,
    const FeatureExtractionOptions& options);

}  // namespace colmap
