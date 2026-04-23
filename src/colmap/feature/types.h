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

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/enum_utils.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

MAKE_ENUM_CLASS_OVERLOAD_STREAM(
    FeatureExtractorType, -1, UNDEFINED, SIFT, ALIKED_N16ROT, ALIKED_N32);
MAKE_ENUM_CLASS_OVERLOAD_STREAM(FeatureMatcherType,
                                -1,
                                UNDEFINED,
                                SIFT_BRUTEFORCE,
                                SIFT_LIGHTGLUE,
                                ALIKED_BRUTEFORCE,
                                ALIKED_LIGHTGLUE);

struct FeatureKeypoint {
  FeatureKeypoint();
  FeatureKeypoint(float x, float y);
  FeatureKeypoint(float x, float y, float scale, float orientation);
  FeatureKeypoint(float x, float y, float a11, float a12, float a21, float a22);

  static FeatureKeypoint FromShapeParameters(float x,
                                             float y,
                                             float scale_x,
                                             float scale_y,
                                             float orientation,
                                             float shear);

  // Rescale the feature location and shape size by the given scale factor.
  void Rescale(float scale);
  void Rescale(float scale_x, float scale_y);

  // Rotate the feature location and shape by k * 90 degrees counter-clockwise
  // around the image center. The width and height are the dimensions of the
  // image the keypoint is currently defined on.
  void Rot90(int k, int width, int height);

  // Compute shape parameters from affine shape.
  float ComputeScale() const;
  float ComputeScaleX() const;
  float ComputeScaleY() const;
  float ComputeOrientation() const;
  float ComputeShear() const;

  // Location of the feature, with the origin at the upper left image corner,
  // i.e. the upper left pixel has the coordinate (0.5, 0.5).
  float x;
  float y;

  // Affine shape of the feature.
  float a11;
  float a12;
  float a21;
  float a22;

  inline bool operator==(const FeatureKeypoint& other) const {
    return x == other.x && y == other.y && a11 == other.a11 &&
           a12 == other.a12 && a21 == other.a21 && a22 == other.a22;
  }

  inline bool operator!=(const FeatureKeypoint& other) const {
    return !(*this == other);
  }
};

using FeatureDescriptor =
    Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor>;
using FeatureKeypoints = std::vector<FeatureKeypoint>;

// Matrix types for descriptor data.
using FeatureDescriptorsData =
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FeatureDescriptorsFloatData =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Forward declaration for conversion methods.
struct FeatureDescriptorsFloat;

// Feature descriptors with associated extractor type metadata.
struct FeatureDescriptors {
  FeatureDescriptors() = default;
  FeatureDescriptors(FeatureExtractorType type, FeatureDescriptorsData data)
      : type(type), data(std::move(data)) {}

  // Create from float descriptors by reinterpreting as uint8 bytes.
  static FeatureDescriptors FromFloat(
      const FeatureDescriptorsFloat& float_desc);

  // Convert to float descriptors by reinterpreting uint8 data as float32.
  FeatureDescriptorsFloat ToFloat() const;

  FeatureExtractorType type = FeatureExtractorType::UNDEFINED;
  FeatureDescriptorsData data;
};

struct FeatureDescriptorsFloat {
  FeatureDescriptorsFloat() = default;
  FeatureDescriptorsFloat(FeatureExtractorType type,
                          FeatureDescriptorsFloatData data)
      : type(type), data(std::move(data)) {}

  // Create from byte descriptors by reinterpreting uint8 data as float32.
  static FeatureDescriptorsFloat FromBytes(const FeatureDescriptors& byte_desc);

  // Convert to byte descriptors by reinterpreting float32 data as uint8.
  FeatureDescriptors ToBytes() const;

  FeatureExtractorType type = FeatureExtractorType::UNDEFINED;
  FeatureDescriptorsFloatData data;
};

struct FeatureMatch {
  FeatureMatch()
      : point2D_idx1(kInvalidPoint2DIdx), point2D_idx2(kInvalidPoint2DIdx) {}
  FeatureMatch(const point2D_t point2D_idx1, const point2D_t point2D_idx2)
      : point2D_idx1(point2D_idx1), point2D_idx2(point2D_idx2) {}

  // Feature index in first image.
  point2D_t point2D_idx1 = kInvalidPoint2DIdx;

  // Feature index in second image.
  point2D_t point2D_idx2 = kInvalidPoint2DIdx;

  inline bool operator==(const FeatureMatch& other) const {
    return point2D_idx1 == other.point2D_idx1 &&
           point2D_idx2 == other.point2D_idx2;
  }

  inline bool operator!=(const FeatureMatch& other) const {
    return !(*this == other);
  }
};

using FeatureMatches = std::vector<FeatureMatch>;

inline constexpr int kKeypointMatrixCols = 4;

using FeatureKeypointsMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, kKeypointMatrixCols, Eigen::RowMajor>;
using FeatureMatchesMatrix =
    Eigen::Matrix<uint32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;

// Convert FeatureKeypoints to an Nx4 matrix [x, y, scale, orientation].
FeatureKeypointsMatrix KeypointsToMatrix(
    const FeatureKeypoints& feature_keypoints);

// Convert an Nx4 matrix [x, y, scale, orientation] to FeatureKeypoints.
FeatureKeypoints KeypointsFromMatrix(
    const Eigen::Ref<const FeatureKeypointsMatrix>& keypoints);

// Convert FeatureMatches to an Nx2 matrix of point2D indices.
FeatureMatchesMatrix MatchesToMatrix(const FeatureMatches& feature_matches);

// Convert an Nx2 matrix of point2D indices to FeatureMatches.
FeatureMatches MatchesFromMatrix(
    const Eigen::Ref<const FeatureMatchesMatrix>& matches);

}  // namespace colmap
