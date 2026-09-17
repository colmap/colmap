// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/types.h"

#include <array>
#include <cstdint>
#include <vector>

namespace colmap {

class Bitmap;

// Convert feature keypoints to vector of points.
std::vector<Eigen::Vector2d> FeatureKeypointsToPointsVector(
    const FeatureKeypoints& keypoints);

// L2-normalize feature descriptor, where each row represents one feature.
void L2NormalizeFeatureDescriptors(FeatureDescriptorsFloatData* descriptors);

// L1-Root-normalize feature descriptors, where each row represents one feature.
// See "Three things everyone should know to improve object retrieval",
// Relja Arandjelovic and Andrew Zisserman, CVPR 2012.
void L1RootNormalizeFeatureDescriptors(
    FeatureDescriptorsFloatData* descriptors);

// Convert normalized floating point feature descriptor to unsigned byte
// representation by linear scaling from range [0, 0.5] to [0, 255]. Truncation
// to a maximum value of 0.5 is used to avoid precision loss and follows the
// common practice of representing SIFT vectors.
FeatureDescriptorsData FeatureDescriptorsToUnsignedByte(
    const Eigen::Ref<const FeatureDescriptorsFloatData>& descriptors);

// Extract the descriptors corresponding to the largest-scale features.
void ExtractTopScaleFeatures(FeatureKeypoints* keypoints,
                             FeatureDescriptors* descriptors,
                             size_t num_features);

// Convert an HWC uint8 image buffer to a row-major CHW float tensor,
// normalized to [0, 1]. The pitch is the scan-line size in bytes.
std::vector<float> HWCToCHW(const uint8_t* data,
                            int width,
                            int height,
                            int pitch);

// Convert an RGB bitmap to a row-major CHW float tensor, normalized to [0, 1].
std::vector<float> BitmapToCHW(const Bitmap& bitmap);

// Cache for per-image data (e.g. extracted features or descriptor indexes),
// keyed by image id. Retains the two most recently used entries; inserting
// a third image evicts the least recently used entry, so matching consecutive
// image pairs (A, B) then (B, C) only creates features for C. Returned
// references stay valid until the entry is evicted. Entries with invalid
// image ids are never reused. Images must expose an image_id member.
template <typename T>
class ImageFeatureCache {
 public:
  template <typename Image, typename CreateFn>
  T& GetOrCreate(const Image& image, CreateFn&& create) {
    for (int i = 0; i < kCapacity; ++i) {
      if (image.image_id != kInvalidImageId &&
          entries_[i].image_id == image.image_id) {
        most_recent_ = i;
        return entries_[i].value;
      }
    }
    most_recent_ = kCapacity - 1 - most_recent_;
    Entry& entry = entries_[most_recent_];
    entry.value = create(image);
    entry.image_id = image.image_id;
    return entry.value;
  }

 private:
  static constexpr int kCapacity = 2;
  struct Entry {
    image_t image_id = kInvalidImageId;
    T value;
  };
  std::array<Entry, kCapacity> entries_;
  int most_recent_ = 0;
};

}  // namespace colmap
