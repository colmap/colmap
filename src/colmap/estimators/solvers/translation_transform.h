// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// Estimate a N-D translation transformation between point pairs.
template <int kDim>
class TranslationTransformEstimator {
 public:
  using X_t = Eigen::Matrix<double, kDim, 1>;
  using Y_t = Eigen::Matrix<double, kDim, 1>;
  using M_t = Eigen::Matrix<double, kDim, 1>;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 1;

  // Estimate the N-D translation transform.
  //
  // @param points1      Set of corresponding source points.
  // @param points2      Set of corresponding destination points.
  //
  // @return             Translation vector.
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  // Calculate the squared translation error.
  //
  // @param points1      Set of corresponding source points.
  // @param points2      Set of corresponding destination points.
  // @param translation  Translation vector.
  // @param residuals    Output vector of residuals for each point pair.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& translation,
                        std::vector<double>* residuals);
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <int kDim>
void TranslationTransformEstimator<kDim>::Estimate(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  THROW_CHECK(models != nullptr);

  models->clear();

  X_t mean_src = X_t::Zero();
  Y_t mean_dst = Y_t::Zero();

  for (size_t i = 0; i < points1.size(); ++i) {
    mean_src += points1[i];
    mean_dst += points2[i];
  }

  mean_src /= points1.size();
  mean_dst /= points2.size();

  models->resize(1);
  (*models)[0] = mean_dst - mean_src;
}

template <int kDim>
void TranslationTransformEstimator<kDim>::Residuals(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    const M_t& translation,
    std::vector<double>* residuals) {
  THROW_CHECK_EQ(points1.size(), points2.size());

  residuals->resize(points1.size());

  for (size_t i = 0; i < points1.size(); ++i) {
    const M_t diff = points2[i] - points1[i] - translation;
    (*residuals)[i] = diff.squaredNorm();
  }
}

}  // namespace colmap
