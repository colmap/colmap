// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/extractor.h"

#include <memory>
#include <string>

namespace colmap {

struct CusiftExtractionOptions {
  // Number of octaves. -1 picks it as COLMAP's SiftGPU backend does (see
  // CusiftNumOctaves); a positive value builds exactly that many. Separate
  // from SiftExtractionOptions::num_octaves, which only the CPU backend uses.
  int num_octaves = -1;

  bool Check() const;
};

// Whether cusift supports these SIFT options: no affine shape, domain-size
// pooling, upright features, darkness adaptivity or covariant extractor;
// first_octave 0 or -1; octave_resolution 3. Logs the reason if not.
bool CheckCusiftSiftOptions(const SiftExtractionOptions& sift);

// CUDA SIFT extractor replicating COLMAP's VLFeat CPU backend, configured by
// options.sift. Its descriptors are tagged FeatureExtractorType::SIFT, so the
// SIFT matchers and databases use them unchanged. Returns nullptr if COLMAP was
// built without cusift (CUSIFT_ENABLED, which needs CUDA >= 11.2).
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

// options.num_octaves if positive, else CusiftNumOctaves(...).
int CusiftEffectiveNumOctaves(const CusiftExtractionOptions& options,
                              int width,
                              int height,
                              int first_octave);

}  // namespace colmap
