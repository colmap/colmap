// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

class Reconstruction;

// 3D similarity transform with 7 degrees of freedom.
// Transforms point x from a to b as: x_in_b = scale * R * x_in_a + t.
class SimilarityTransform3 {
 public:
  // Default construct identity transform.
  SimilarityTransform3();

  // Construct from existing transform.
  explicit SimilarityTransform3(const Eigen::Matrix3x4d& matrix);
  SimilarityTransform3(double scale,
                       const Eigen::Vector4d& qvec,
                       const Eigen::Vector3d& tvec);

  SimilarityTransform3 Inverse() const;

  // Matrix that transforms points as x_in_b = matrix * x_in_a.homogeneous().
  const Eigen::Matrix3x4d& Matrix() const;

  // Transformation parameters.
  double Scale() const;
  Eigen::Vector4d Rotation() const;
  Eigen::Vector3d Translation() const;

  // Estimate tgtFromSrc transform. Return true if successful.
  bool Estimate(const std::vector<Eigen::Vector3d>& src,
                const std::vector<Eigen::Vector3d>& tgt);

  // Apply transform to point.
  inline Eigen::Vector3d operator*(const Eigen::Vector3d& x) const;

  // Transform world for camFromWorld pose.
  // TODO(jsch): Rename and refactor with future RigidTransform class.
  void TransformPose(Eigen::Vector4d* qvec, Eigen::Vector3d* tvec) const;

  // Read from or write to text file without loss of precision.
  void ToFile(const std::string& path) const;
  static SimilarityTransform3 FromFile(const std::string& path);

 private:
  Eigen::Matrix3x4d matrix_;
};

// Robustly compute alignment between reconstructions by finding images that
// are registered in both reconstructions. The alignment is then estimated
// robustly inside RANSAC from corresponding projection centers. An alignment
// is verified by reprojecting common 3D point observations.
// The min_inlier_observations threshold determines how many observations
// in a common image must reproject within the given threshold.
bool ComputeAlignmentBetweenReconstructions(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    double min_inlier_observations,
    double max_reproj_error,
    SimilarityTransform3* tgtFromSrc);

// Robustly compute alignment between reconstructions by finding images that
// are registered in both reconstructions. The alignment is then estimated
// robustly inside RANSAC from corresponding projection centers and by
// minimizing the Euclidean distance between them in world space.
bool ComputeAlignmentBetweenReconstructions(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    double max_proj_center_error,
    SimilarityTransform3* tgtFromSrc);

// Compute image alignment errors in the target coordinate frame.
struct ImageAlignmentError {
  image_t image_id = kInvalidImageId;
  double rotation_error_deg = -1;
  double proj_center_error = -1;
};

std::vector<ImageAlignmentError> ComputeImageAlignmentError(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const SimilarityTransform3& tgtFromSrc);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

Eigen::Vector3d SimilarityTransform3::operator*(
    const Eigen::Vector3d& x) const {
  return matrix_ * x.homogeneous();
}

}  // namespace colmap
