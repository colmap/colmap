// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/extractor.h"

#include <memory>
#include <string>

namespace colmap {

struct CusiftExtractionOptions {
  // Number of octaves to build. -1 (the default) picks it as COLMAP's SiftGPU
  // backend does (see CusiftNumOctaves); a positive value builds exactly that
  // many. Note that this is separate from
  // SiftExtractionOptions::num_octaves, which only the CPU (VLFeat) backend
  // uses and whose default of 4 stops at a much smaller maximum scale.
  int num_octaves = -1;

  bool Check() const;
};

// Whether cusift can serve these SIFT options. cusift implements none of
// affine shape estimation, domain-size pooling, upright features, darkness
// adaptivity or the covariant extractor; it supports first_octave 0 or -1 and
// a fixed octave_resolution of 3. Logs the reason and returns false otherwise.
// FeatureExtractionOptions::Check() calls it for FeatureExtractorType::CUSIFT,
// so bad options are reported before any extraction thread starts.
bool CheckCusiftSiftOptions(const SiftExtractionOptions& sift);

// Create a cusift feature extractor: a CUDA SIFT extractor that replicates
// COLMAP's VLFeat CPU backend. Unlike SiftGPU it is MIT-licensed (see
// src/thirdparty/cusift/LICENSE).
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

// Binds the calling thread to the single GPU named by `gpu_index` (-1 picks
// the best device, as colmap::SetBestCudaDevice does) and initialises cusift's
// memory pool on it. cudaSetDevice is per-thread, so every extraction and
// matching thread calls this for itself. Throws if COLMAP was built without
// CUDA, no device is available, or `gpu_index` names more than one GPU.
void BindCusiftDevice(const std::string& gpu_index);

// Number of octaves cusift builds for a width x height image. It follows
// SiftGPU as COLMAP runs it, not SiftExtractionOptions::num_octaves: COLMAP
// passes no octave count to SiftGPU, so SiftGPU builds every octave down to an
// 8-16 pixel image, floor(log2(min(width, height))) - 3 counted on the image
// after first_octave's upsampling (PyramidCU::ResizePyramid), and at least 1.
// Matching it keeps the large-scale features that SiftGPU relies on in blurred
// images, where fine-scale features vanish. COLMAP's CPU (VLFeat) SIFT does use
// num_octaves, so at its default of 4 it stops at a smaller maximum scale.
int CusiftNumOctaves(int width, int height, int first_octave);

// The octave count cusift uses: options.num_octaves if positive, else
// CusiftNumOctaves(width, height, first_octave).
int CusiftEffectiveNumOctaves(const CusiftExtractionOptions& options,
                              int width,
                              int height,
                              int first_octave);

}  // namespace colmap
