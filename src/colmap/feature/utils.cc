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

#include "colmap/feature/utils.h"

#include "colmap/math/math.h"

namespace colmap {

std::vector<Eigen::Vector2d> FeatureKeypointsToPointsVector(
    const FeatureKeypoints& keypoints) {
  std::vector<Eigen::Vector2d> points(keypoints.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    points[i] = Eigen::Vector2d(keypoints[i].x, keypoints[i].y);
  }
  return points;
}

void L2NormalizeFeatureDescriptors(FeatureDescriptorsFloat* descriptors) {
  descriptors->rowwise().normalize();
}

void L1RootNormalizeFeatureDescriptors(FeatureDescriptorsFloat* descriptors) {
  for (Eigen::MatrixXf::Index r = 0; r < descriptors->rows(); ++r) {
    descriptors->row(r) *= 1 / descriptors->row(r).lpNorm<1>();
    descriptors->row(r) = descriptors->row(r).array().sqrt();
  }
}

FeatureDescriptors FeatureDescriptorsToUnsignedByte(
    const Eigen::Ref<const FeatureDescriptorsFloat>& descriptors) {
  FeatureDescriptors descriptors_unsigned_byte(descriptors.rows(),
                                               descriptors.cols());
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    for (Eigen::MatrixXf::Index c = 0; c < descriptors.cols(); ++c) {
      const float scaled_value = std::round(512.0f * descriptors(r, c));
      descriptors_unsigned_byte(r, c) =
          TruncateCast<float, uint8_t>(scaled_value);
    }
  }
  return descriptors_unsigned_byte;
}

void ExtractTopScaleFeatures(FeatureKeypoints* keypoints,
                             FeatureDescriptors* descriptors,
                             const size_t num_features) {
  THROW_CHECK_EQ(keypoints->size(), descriptors->rows());
  THROW_CHECK_GT(num_features, 0);

  if (static_cast<size_t>(descriptors->rows()) <= num_features) {
    return;
  }

  std::vector<std::pair<size_t, float>> scales;
  scales.reserve(keypoints->size());
  for (size_t i = 0; i < keypoints->size(); ++i) {
    scales.emplace_back(i, (*keypoints)[i].ComputeScale());
  }

  std::partial_sort(scales.begin(),
                    scales.begin() + num_features,
                    scales.end(),
                    [](const std::pair<size_t, float>& scale1,
                       const std::pair<size_t, float>& scale2) {
                      return scale1.second > scale2.second;
                    });

  FeatureKeypoints top_scale_keypoints(num_features);
  FeatureDescriptors top_scale_descriptors(num_features, descriptors->cols());
  for (size_t i = 0; i < num_features; ++i) {
    top_scale_keypoints[i] = (*keypoints)[scales[i].first];
    top_scale_descriptors.row(i) = descriptors->row(scales[i].first);
  }

  *keypoints = std::move(top_scale_keypoints);
  *descriptors = std::move(top_scale_descriptors);
}

}  // namespace colmap
