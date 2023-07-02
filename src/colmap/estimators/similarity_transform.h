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

#ifndef COLMAP_SRC_ESTIMATORS_SIMILARITY_TRANSFORM_H_
#define COLMAP_SRC_ESTIMATORS_SIMILARITY_TRANSFORM_H_

#include "colmap/base/projection.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// N-D similarity transform estimator from corresponding point pairs in the
// source and destination coordinate systems.
//
// This algorithm is based on the following paper:
//
//      S. Umeyama. Least-Squares Estimation of Transformation Parameters
//      Between Two Point Patterns. IEEE Transactions on Pattern Analysis and
//      Machine Intelligence, Volume 13 Issue 4, Page 376-380, 1991.
//      http://www.stanford.edu/class/cs273/refs/umeyama.pdf
//
// and uses the Eigen implementation.
template <int kDim, bool kEstimateScale = true>
class SimilarityTransformEstimator {
 public:
  typedef Eigen::Matrix<double, kDim, 1> X_t;
  typedef Eigen::Matrix<double, kDim, 1> Y_t;
  typedef Eigen::Matrix<double, kDim, kDim + 1> M_t;

  // The minimum number of samples needed to estimate a model. Note that
  // this only returns the true minimal sample in the two-dimensional case.
  // For higher dimensions, the system will alway be over-determined.
  static const int kMinNumSamples = kDim;

  // Estimate the similarity transform.
  //
  // @param src      Set of corresponding source points.
  // @param dst      Set of corresponding destination points.
  //
  // @return         4x4 homogeneous transformation matrix.
  static std::vector<M_t> Estimate(const std::vector<X_t>& src,
                                   const std::vector<Y_t>& dst);

  // Calculate the transformation error for each corresponding point pair.
  //
  // Residuals are defined as the squared transformation error when
  // transforming the source to the destination coordinates.
  //
  // @param src        Set of corresponding points in the source coordinate
  //                   system as a Nx3 matrix.
  // @param dst        Set of corresponding points in the destination
  //                   coordinate system as a Nx3 matrix.
  // @param matrix     4x4 homogeneous transformation matrix.
  // @param residuals  Output vector of residuals for each point pair.
  static void Residuals(const std::vector<X_t>& src,
                        const std::vector<Y_t>& dst,
                        const M_t& matrix,
                        std::vector<double>* residuals);
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <int kDim, bool kEstimateScale>
std::vector<typename SimilarityTransformEstimator<kDim, kEstimateScale>::M_t>
SimilarityTransformEstimator<kDim, kEstimateScale>::Estimate(
    const std::vector<X_t>& src, const std::vector<Y_t>& dst) {
  CHECK_EQ(src.size(), dst.size());

  Eigen::Matrix<double, kDim, Eigen::Dynamic> src_mat(kDim, src.size());
  Eigen::Matrix<double, kDim, Eigen::Dynamic> dst_mat(kDim, dst.size());
  for (size_t i = 0; i < src.size(); ++i) {
    src_mat.col(i) = src[i];
    dst_mat.col(i) = dst[i];
  }

  const M_t model = Eigen::umeyama(src_mat, dst_mat, kEstimateScale)
                        .topLeftCorner(kDim, kDim + 1);

  if (model.array().isNaN().any()) {
    return std::vector<M_t>{};
  }

  return {model};
}

template <int kDim, bool kEstimateScale>
void SimilarityTransformEstimator<kDim, kEstimateScale>::Residuals(
    const std::vector<X_t>& src,
    const std::vector<Y_t>& dst,
    const M_t& matrix,
    std::vector<double>* residuals) {
  CHECK_EQ(src.size(), dst.size());

  residuals->resize(src.size());

  for (size_t i = 0; i < src.size(); ++i) {
    const Y_t dst_transformed = matrix * src[i].homogeneous();
    (*residuals)[i] = (dst[i] - dst_transformed).squaredNorm();
  }
}

}  // namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_SIMILARITY_TRANSFORM_H_
