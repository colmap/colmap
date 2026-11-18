// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/extractor.h"

#include <memory>
#include <string>

namespace colmap {

struct CusiftExtractionOptions {
  // Number of octaves. A positive value builds that many, or fewer if the
  // pyramid would halve below 1 px; the default is the CPU backend's
  // (SiftExtractionOptions::num_octaves), so cusift builds VLFeat's pyramid.
  // -1 picks it as COLMAP's SiftGPU backend does (see CusiftNumOctaves).
  // SiftExtractionOptions::num_octaves, which the CPU backend uses, must stay
  // at its default under cusift.
  int num_octaves = 4;

  bool Check() const;
};

// Whether cusift supports these SIFT options: no affine shape, domain-size
// pooling, upright features, darkness adaptivity or covariant extractor;
// first_octave 0 or -1; octave_resolution 3; num_octaves at its default
// (CusiftExtractionOptions::num_octaves sets cusift's). Logs the reason if not.
bool CheckCusiftSiftOptions(const SiftExtractionOptions& sift);

// The first SIFT option cusift does not support, as a message that names it
// (for example "sift.upright: ..."); empty if CheckCusiftSiftOptions passes.
std::string CusiftUnsupportedSiftOption(const SiftExtractionOptions& sift);

// Resets every SIFT option cusift does not support (see
// CheckCusiftSiftOptions) to its SIFT default, so that CheckCusiftSiftOptions
// passes. Leaves the supported options, first_octave 0 among them, unchanged.
void ClearUnsupportedCusiftSiftOptions(SiftExtractionOptions* sift);

// CUDA SIFT extractor replicating COLMAP's VLFeat CPU backend, configured by
// options.sift. Its descriptors are tagged FeatureExtractorType::SIFT, so the
// SIFT matchers and databases use them unchanged. Returns nullptr if COLMAP was
// built without cusift (CUSIFT_ENABLED, which needs CUDA >= 11.2).
//
// The extractor is bound to the GPU named by options.gpu_index, which Create
// makes the calling thread's current device (BindCusiftDevice); Extract binds
// its calling thread to that GPU again. Use one extractor from one thread at a
// time; it may be destroyed on any thread.
std::unique_ptr<FeatureExtractor> CreateCusiftFeatureExtractor(
    const FeatureExtractionOptions& options);

// Binds the calling thread to the single GPU named by `gpu_index` (-1: the
// best device, as SetBestCudaDevice) and initialises cusift's memory pool on
// it. Throws without cusift, without a device, or if `gpu_index` names more
// than one GPU.
void BindCusiftDevice(const std::string& gpu_index);

// Octave count as SiftGPU builds it under COLMAP, which passes it no count:
// floor(log2(min(width, height))) - 3 after first_octave's upsampling, at
// least 1 (PyramidCU::ResizePyramid).
int CusiftNumOctaves(int width, int height, int first_octave);

// options.num_octaves if positive, else CusiftNumOctaves(...); either clamped
// so that the smallest octave is at least one pixel wide and high.
int CusiftEffectiveNumOctaves(const CusiftExtractionOptions& options,
                              int width,
                              int height,
                              int first_octave);

}  // namespace colmap
