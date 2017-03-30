// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_ESTIMATORS_SIMILARITY_TRANSFORM_H_
#define COLMAP_SRC_ESTIMATORS_SIMILARITY_TRANSFORM_H_

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/projection.h"
#include "util/alignment.h"
#include "util/logging.h"
#include "util/types.h"

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
                        const std::vector<Y_t>& dst, const M_t& matrix,
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

  std::vector<M_t> models(1);
  models[0] = Eigen::umeyama(src_mat, dst_mat, kEstimateScale)
                  .topLeftCorner(kDim, kDim + 1);

  return models;
}

template <int kDim, bool kEstimateScale>
void SimilarityTransformEstimator<kDim, kEstimateScale>::Residuals(
    const std::vector<X_t>& src, const std::vector<Y_t>& dst, const M_t& matrix,
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
