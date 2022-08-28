// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

#ifndef COLMAP_SRC_BASE_SIMILARITY_TRANSFORM_H_
#define COLMAP_SRC_BASE_SIMILARITY_TRANSFORM_H_

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "estimators/similarity_transform.h"
#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

struct RANSACOptions;
class Reconstruction;

// 3D similarity transformation with 7 degrees of freedom.
class SimilarityTransform3 {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SimilarityTransform3();

  explicit SimilarityTransform3(const Eigen::Matrix3x4d& matrix);

  explicit SimilarityTransform3(
      const Eigen::Transform<double, 3, Eigen::Affine>& transform);

  SimilarityTransform3(const double scale, const Eigen::Vector4d& qvec,
                       const Eigen::Vector3d& tvec);

  void Write(const std::string& path);

  template <bool kEstimateScale = true>
  bool Estimate(const std::vector<Eigen::Vector3d>& src,
                const std::vector<Eigen::Vector3d>& dst);

  SimilarityTransform3 Inverse() const;

  void TransformPoint(Eigen::Vector3d* xyz) const;
  void TransformPose(Eigen::Vector4d* qvec, Eigen::Vector3d* tvec) const;

  Eigen::Matrix4d Matrix() const;
  double Scale() const;
  Eigen::Vector4d Rotation() const;
  Eigen::Vector3d Translation() const;

  static SimilarityTransform3 FromFile(const std::string& path);

 private:
  Eigen::Transform<double, 3, Eigen::Affine> transform_;
};

// Robustly compute alignment between reconstructions by finding images that
// are registered in both reconstructions. The alignment is then estimated
// robustly inside RANSAC from corresponding projection centers. An alignment
// is verified by reprojecting common 3D point observations.
// The min_inlier_observations threshold determines how many observations
// in a common image must reproject within the given threshold.
bool ComputeAlignmentBetweenReconstructions(
    const Reconstruction& src_reconstruction,
    const Reconstruction& ref_reconstruction,
    const double min_inlier_observations, const double max_reproj_error,
    Eigen::Matrix3x4d* alignment);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <bool kEstimateScale>
bool SimilarityTransform3::Estimate(const std::vector<Eigen::Vector3d>& src,
                                    const std::vector<Eigen::Vector3d>& dst) {
  const auto results =
      SimilarityTransformEstimator<3, kEstimateScale>().Estimate(src, dst);
  if (results.empty()) {
    return false;
  }

  CHECK_EQ(results.size(), 1);
  transform_.matrix().topLeftCorner<3, 4>() = results[0];

  return true;
}

}  // namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(colmap::SimilarityTransform3)

#endif  // COLMAP_SRC_BASE_SIMILARITY_TRANSFORM_H_
