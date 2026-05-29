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

#include "colmap/feature/types.h"

#include "colmap/util/logging.h"

#include <cstring>

namespace colmap {

FeatureKeypoint::FeatureKeypoint() : FeatureKeypoint(0, 0) {}

FeatureKeypoint::FeatureKeypoint(const float x, const float y)
    : FeatureKeypoint(x, y, 1, 0, 0, 1) {}

FeatureKeypoint::FeatureKeypoint(const float x_,
                                 const float y_,
                                 const float scale,
                                 const float orientation)
    : x(x_), y(y_) {
  THROW_CHECK_GE(scale, 0.0);
  const float scale_cos_orientation = scale * std::cos(orientation);
  const float scale_sin_orientation = scale * std::sin(orientation);
  a11 = scale_cos_orientation;
  a12 = -scale_sin_orientation;
  a21 = scale_sin_orientation;
  a22 = scale_cos_orientation;
}

FeatureKeypoint::FeatureKeypoint(const float x_,
                                 const float y_,
                                 const float a11_,
                                 const float a12_,
                                 const float a21_,
                                 const float a22_)
    : x(x_), y(y_), a11(a11_), a12(a12_), a21(a21_), a22(a22_) {}

FeatureKeypoint FeatureKeypoint::FromShapeParameters(const float x,
                                                     const float y,
                                                     const float scale_x,
                                                     const float scale_y,
                                                     const float orientation,
                                                     const float shear) {
  THROW_CHECK_GE(scale_x, 0.0);
  THROW_CHECK_GE(scale_y, 0.0);
  return FeatureKeypoint(x,
                         y,
                         scale_x * std::cos(orientation),
                         -scale_y * std::sin(orientation + shear),
                         scale_x * std::sin(orientation),
                         scale_y * std::cos(orientation + shear));
}

void FeatureKeypoint::Rescale(const float scale) { Rescale(scale, scale); }

void FeatureKeypoint::Rescale(const float scale_x, const float scale_y) {
  THROW_CHECK_GT(scale_x, 0);
  THROW_CHECK_GT(scale_y, 0);
  x *= scale_x;
  y *= scale_y;
  a11 *= scale_x;
  a12 *= scale_y;
  a21 *= scale_x;
  a22 *= scale_y;
}

void FeatureKeypoint::Rot90(int k, int width, int height) {
  k = k % 4;
  if (k < 0) {
    k += 4;
  }
  if (k == 0) {
    return;
  }
  float new_x = x, new_y = y;
  float new_a11 = a11, new_a12 = a12, new_a21 = a21, new_a22 = a22;
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);

  if (k == 1) {  // 90 CCW
    new_x = y;
    new_y = w - x;
    new_a11 = a21;
    new_a12 = a22;
    new_a21 = -a11;
    new_a22 = -a12;
  } else if (k == 2) {  // 180 CCW
    new_x = w - x;
    new_y = h - y;
    new_a11 = -a11;
    new_a12 = -a12;
    new_a21 = -a21;
    new_a22 = -a22;
  } else if (k == 3) {  // 270 CCW
    new_x = h - y;
    new_y = x;
    new_a11 = -a21;
    new_a12 = -a22;
    new_a21 = a11;
    new_a22 = a12;
  }
  x = new_x;
  y = new_y;
  a11 = new_a11;
  a12 = new_a12;
  a21 = new_a21;
  a22 = new_a22;
}

float FeatureKeypoint::ComputeScale() const {
  return (ComputeScaleX() + ComputeScaleY()) / 2.0f;
}

float FeatureKeypoint::ComputeScaleX() const {
  return std::sqrt(a11 * a11 + a21 * a21);
}

float FeatureKeypoint::ComputeScaleY() const {
  return std::sqrt(a12 * a12 + a22 * a22);
}

float FeatureKeypoint::ComputeOrientation() const {
  return std::atan2(a21, a11);
}

float FeatureKeypoint::ComputeShear() const {
  return std::atan2(-a12, a22) - ComputeOrientation();
}

FeatureDescriptors FeatureDescriptors::FromFloat(
    const FeatureDescriptorsFloat& float_desc) {
  FeatureDescriptors result;
  result.type = float_desc.type;
  const Eigen::Index rows = float_desc.data.rows();
  const Eigen::Index float_cols = float_desc.data.cols();

  switch (float_desc.type) {
    case FeatureExtractorType::SIFT:
      // cast each float value to uint8
      result.data = float_desc.data.cast<uint8_t>();
      break;
    case FeatureExtractorType::ALIKED_N16ROT:
    case FeatureExtractorType::ALIKED_N32: {
      // reinterpret float32 data as uint8 bytes
      const Eigen::Index uint8_cols = float_cols * sizeof(float);
      result.data.resize(rows, uint8_cols);
      std::memcpy(result.data.data(),
                  float_desc.data.data(),
                  rows * float_cols * sizeof(float));
      break;
    }
    default:
      LOG(FATAL_THROW) << "Unsupported feature type: "
                       << FeatureExtractorTypeToString(float_desc.type);
  }
  return result;
}

FeatureDescriptorsFloat FeatureDescriptors::ToFloat() const {
  return FeatureDescriptorsFloat::FromBytes(*this);
}

FeatureDescriptorsFloat FeatureDescriptorsFloat::FromBytes(
    const FeatureDescriptors& byte_desc) {
  FeatureDescriptorsFloat result;
  result.type = byte_desc.type;
  const Eigen::Index rows = byte_desc.data.rows();
  const Eigen::Index uint8_cols = byte_desc.data.cols();

  switch (byte_desc.type) {
    case FeatureExtractorType::SIFT:
      // cast each uint8 value to float
      result.data = byte_desc.data.cast<float>();
      break;
    case FeatureExtractorType::ALIKED_N16ROT:
    case FeatureExtractorType::ALIKED_N32: {
      // reinterpret uint8 bytes as float32 data
      THROW_CHECK_EQ(uint8_cols % sizeof(float), 0);
      const Eigen::Index float_cols = uint8_cols / sizeof(float);
      result.data.resize(rows, float_cols);
      std::memcpy(result.data.data(), byte_desc.data.data(), rows * uint8_cols);
      break;
    }
    default:
      LOG(FATAL_THROW) << "Unsupported feature type: "
                       << FeatureExtractorTypeToString(byte_desc.type);
  }
  return result;
}

FeatureDescriptors FeatureDescriptorsFloat::ToBytes() const {
  return FeatureDescriptors::FromFloat(*this);
}

FeatureKeypointsMatrix KeypointsToMatrix(
    const FeatureKeypoints& feature_keypoints) {
  const size_t num_features = feature_keypoints.size();
  FeatureKeypointsMatrix keypoints(num_features, kKeypointMatrixCols);
  for (size_t i = 0; i < num_features; ++i) {
    keypoints(i, 0) = feature_keypoints[i].x;
    keypoints(i, 1) = feature_keypoints[i].y;
    keypoints(i, 2) = feature_keypoints[i].ComputeScale();
    keypoints(i, 3) = feature_keypoints[i].ComputeOrientation();
  }
  return keypoints;
}

FeatureKeypoints KeypointsFromMatrix(
    const Eigen::Ref<const FeatureKeypointsMatrix>& keypoints) {
  FeatureKeypoints feature_keypoints(keypoints.rows());
  for (Eigen::Index i = 0; i < keypoints.rows(); ++i) {
    feature_keypoints[i] = FeatureKeypoint(
        keypoints(i, 0), keypoints(i, 1), keypoints(i, 2), keypoints(i, 3));
  }
  return feature_keypoints;
}

FeatureMatchesMatrix MatchesToMatrix(const FeatureMatches& feature_matches) {
  const size_t num_matches = feature_matches.size();
  FeatureMatchesMatrix matches(num_matches, 2);
  for (size_t i = 0; i < num_matches; ++i) {
    matches(i, 0) = feature_matches[i].point2D_idx1;
    matches(i, 1) = feature_matches[i].point2D_idx2;
  }
  return matches;
}

FeatureMatches MatchesFromMatrix(
    const Eigen::Ref<const FeatureMatchesMatrix>& matches) {
  FeatureMatches feature_matches(matches.rows());
  for (Eigen::Index i = 0; i < matches.rows(); ++i) {
    feature_matches[i] = FeatureMatch(matches(i, 0), matches(i, 1));
  }
  return feature_matches;
}

}  // namespace colmap
