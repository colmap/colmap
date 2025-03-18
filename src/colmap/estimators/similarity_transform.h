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

#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/sim3.h"
#include "colmap/optim/loransac.h"
#include "colmap/optim/ransac.h"
#include "colmap/util/eigen_alignment.h"
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
  // @param tgt      Set of corresponding destination points.
  //
  // @return         4x4 homogeneous transformation matrix.
  static void Estimate(const std::vector<X_t>& src,
                       const std::vector<Y_t>& tgt,
                       std::vector<M_t>* tgt_from_src);

  // Calculate the transformation error for each corresponding point pair.
  //
  // Residuals are defined as the squared transformation error when
  // transforming the source to the destination coordinates.
  //
  // @param src           Set of corresponding points in the source coordinate
  //                      system as a Nx3 matrix.
  // @param tgt           Set of corresponding points in the destination
  //                      coordinate system as a Nx3 matrix.
  // @param tgt_from_src  4x4 homogeneous transformation matrix.
  // @param residuals     Output vector of residuals for each point pair.
  static void Residuals(const std::vector<X_t>& src,
                        const std::vector<Y_t>& tgt,
                        const M_t& tgt_from_src,
                        std::vector<double>* residuals);
};

bool EstimateRigid3d(const std::vector<Eigen::Vector3d>& src,
                     const std::vector<Eigen::Vector3d>& tgt,
                     Rigid3d& tgt_from_src);

typename RANSAC<SimilarityTransformEstimator<3, false>>::Report
EstimateRigid3dRobust(const std::vector<Eigen::Vector3d>& src,
                      const std::vector<Eigen::Vector3d>& tgt,
                      const RANSACOptions& options,
                      Rigid3d& tgt_from_src);

bool EstimateSim3d(const std::vector<Eigen::Vector3d>& src,
                   const std::vector<Eigen::Vector3d>& tgt,
                   Sim3d& tgt_from_src);

typename RANSAC<SimilarityTransformEstimator<3, true>>::Report
EstimateSim3dRobust(const std::vector<Eigen::Vector3d>& src,
                    const std::vector<Eigen::Vector3d>& tgt,
                    const RANSACOptions& options,
                    Sim3d& tgt_from_src);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <int kDim, bool kEstimateScale>
void SimilarityTransformEstimator<kDim, kEstimateScale>::Estimate(
    const std::vector<X_t>& src,
    const std::vector<Y_t>& tgt,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(src.size(), tgt.size());
  THROW_CHECK_GE(src.size(), kMinNumSamples);
  THROW_CHECK(models != nullptr);

  models->clear();

  using MatrixType = Eigen::Matrix<double, kDim, Eigen::Dynamic>;
  const Eigen::Map<const MatrixType> src_mat(
      reinterpret_cast<const double*>(src.data()), kDim, src.size());
  const Eigen::Map<const MatrixType> tgt_mat(
      reinterpret_cast<const double*>(tgt.data()), kDim, tgt.size());

  if (Eigen::FullPivLU<MatrixType>(src_mat).rank() < kMinNumSamples ||
      Eigen::FullPivLU<MatrixType>(tgt_mat).rank() < kMinNumSamples) {
    return;
  }

  const M_t sol = Eigen::umeyama(src_mat, tgt_mat, kEstimateScale)
                      .template topLeftCorner<kDim, kDim + 1>();

  if (sol.hasNaN()) {
    return;
  }

  models->resize(1);
  (*models)[0] = sol;
}

template <int kDim, bool kEstimateScale>
void SimilarityTransformEstimator<kDim, kEstimateScale>::Residuals(
    const std::vector<X_t>& src,
    const std::vector<Y_t>& tgt,
    const M_t& tgt_from_src,
    std::vector<double>* residuals) {
  const size_t num_points = src.size();
  THROW_CHECK_EQ(num_points, tgt.size());
  residuals->resize(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    (*residuals)[i] =
        (tgt[i] - tgt_from_src * src[i].homogeneous()).squaredNorm();
  }
}

}  // namespace colmap
