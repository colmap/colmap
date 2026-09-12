// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/types.h"

namespace colmap {

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

}  // namespace colmap
