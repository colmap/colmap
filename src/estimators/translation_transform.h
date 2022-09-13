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

#include "util/alignment.h"
#include "util/logging.h"
#include "util/types.h"

namespace colmap {

// Estimate a N-D translation transformation between point pairs.
template <int kDim>
class TranslationTransformEstimator {
 public:
  typedef Eigen::Matrix<double, kDim, 1> X_t;
  typedef Eigen::Matrix<double, kDim, 1> Y_t;
  typedef Eigen::Matrix<double, kDim, 1> M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 1;

  // Estimate the 2D translation transform.
  //
  // @param points1      Set of corresponding source 2D points.
  // @param points2      Set of corresponding destination 2D points.
  //
  // @return             Translation vector.
  static std::vector<M_t> Estimate(const std::vector<X_t>& points1,
                                   const std::vector<Y_t>& points2);

  // Calculate the squared translation error.
  //
  // @param points1      Set of corresponding source 2D points.
  // @param points2      Set of corresponding destination 2D points.
  // @param translation  Translation vector.
  // @param residuals    Output vector of residuals for each point pair.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2, const M_t& translation,
                        std::vector<double>* residuals);
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <int kDim>
std::vector<typename TranslationTransformEstimator<kDim>::M_t>
TranslationTransformEstimator<kDim>::Estimate(const std::vector<X_t>& points1,
                                              const std::vector<Y_t>& points2) {
  CHECK_EQ(points1.size(), points2.size());

  X_t mean_src = X_t::Zero();
  Y_t mean_dst = Y_t::Zero();

  for (size_t i = 0; i < points1.size(); ++i) {
    mean_src += points1[i];
    mean_dst += points2[i];
  }

  mean_src /= points1.size();
  mean_dst /= points2.size();

  std::vector<M_t> models(1);
  models[0] = mean_dst - mean_src;

  return models;
}

template <int kDim>
void TranslationTransformEstimator<kDim>::Residuals(
    const std::vector<X_t>& points1, const std::vector<Y_t>& points2,
    const M_t& translation, std::vector<double>* residuals) {
  CHECK_EQ(points1.size(), points2.size());

  residuals->resize(points1.size());

  for (size_t i = 0; i < points1.size(); ++i) {
    const M_t diff = points2[i] - points1[i] - translation;
    (*residuals)[i] = diff.squaredNorm();
  }
}

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_SIMILARITY_TRANSFORM_H_
