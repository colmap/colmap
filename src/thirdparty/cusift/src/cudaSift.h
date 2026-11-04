// SPDX-License-Identifier: MIT
// Copyright (c) 2017 Mårten Björkman (CudaSift; see ../LICENSE.CudaSift)
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudaImage.h"
#include <array>
#include <vector>


// Packed kernel I/O for the VLFeat pipeline: only the fields the kernels read
// and write.
//   orientation: reads x, y, sigma, level      -> writes VLFeatOri
//   descriptor:  reads x, y, sigma, ori, level -> writes 128 floats, rescales x/y/sigma
struct VLFeatKp {
  float x, y, sigma, ori;  // ori in RADIANS (the kernel's own convention)
  int level;               // gradient-plane index, = is - s_min
};

struct VLFeatOri {         // orientation-kernel output
  float angles[4];         // radians, VLFeat convention
  int count;               // 0..4
};

// A detected keypoint: octave o, scale index is, integer position (ix, iy),
// refined position (x, y), absolute sigma, and refined scale s.
struct DetectedKeypoint {
  int o, is, ix, iy;
  float x, y, sigma, s;
};

// Full VLFeat-parity pipeline: detection (extrema scan, Newton refinement, edge
// test), orientation(s) and descriptors, all on the GPU, with each keypoint
// expanded to up to maxNumOrientations orientations.
// firstOctave = VLFeat o_min: 0, or -1 for COLMAP's default 2x-upsampled
// extra fine octave (roughly doubles the keypoint count).
void ExtractSiftFullVLFeat(CudaImage &img, int numOctaves, float peakThresh, float edgeThresh,
                           int maxNumOrientations, int firstOctave, int maxNumFeatures,
                           std::vector<float> &outX, std::vector<float> &outY,
                           std::vector<float> &outSigma, std::vector<float> &outOrientation,
                           std::vector<std::array<float, 128>> &outDescriptors);

#endif
