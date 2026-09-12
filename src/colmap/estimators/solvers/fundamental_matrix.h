// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/eigen_alignment.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Fundamental matrix estimator from corresponding point pairs.
//
// This algorithm solves the 7-Point problem and is based on the following
// paper:
//
//    Zhengyou Zhang and T. Kanade, Determining the Epipolar Geometry and its
//    Uncertainty: A Review, International Journal of Computer Vision, 1998.
//    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.33.4540
class FundamentalMatrixSevenPointEstimator {
 public:
  using X_t = Eigen::Vector2d;
  using Y_t = Eigen::Vector2d;
  using M_t = Eigen::Matrix3d;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 7;

  // Estimate either 1 or 3 possible fundamental matrix solutions from a set of
  // corresponding points.
  //
  // The number of corresponding points must be exactly 7.
  //
  // @param points1  First set of corresponding points.
  // @param points2  Second set of corresponding points
  //
  // @return         Up to 4 solutions as a vector of 3x3 fundamental matrices.
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  // Calculate the residuals of a set of corresponding points and a given
  // fundamental matrix.
  //
  // Residuals are defined as the squared Sampson error.
  //
  // @param points1    First set of corresponding points as Nx2 matrix.
  // @param points2    Second set of corresponding points as Nx2 matrix.
  // @param F          3x3 fundamental matrix.
  // @param residuals  Output vector of residuals.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& F,
                        std::vector<double>* residuals);
};

// Fundamental matrix estimator from corresponding point pairs.
//
// This algorithm solves the 8-Point problem based on the following paper:
//
//    Hartley and Zisserman, Multiple View Geometry, algorithm 11.1, page 282.
class FundamentalMatrixEightPointEstimator {
 public:
  using X_t = Eigen::Vector2d;
  using Y_t = Eigen::Vector2d;
  using M_t = Eigen::Matrix3d;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 8;

  // Estimate fundamental matrix solutions from a set of corresponding points.
  //
  // The number of corresponding points must be at least 8.
  //
  // @param points1  First set of corresponding points.
  // @param points2  Second set of corresponding points
  //
  // @return         Single solution as a vector of 3x3 fundamental matrices.
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  // Calculate the residuals of a set of corresponding points and a given
  // fundamental matrix.
  //
  // Residuals are defined as the squared Sampson error.
  //
  // @param points1    First set of corresponding points as Nx2 matrix.
  // @param points2    Second set of corresponding points as Nx2 matrix.
  // @param F          3x3 fundamental matrix.
  // @param residuals  Output vector of residuals.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& F,
                        std::vector<double>* residuals);
};

// Refine a fundamental matrix in place by minimizing the Sampson error over the
// given correspondences, starting from *F.
//
// Optimizes the SVD factorization of Bartoli and Sturm, "Non-Linear Estimation
// of the Fundamental Matrix With Minimal Parameters", PAMI 2004, which keeps
// rank 2 and the scale gauge exact at every iterate, unlike the 8-point
// algorithm, which truncates the smallest singular value after the fact.
//
// Points are centered and normalized internally, with a scale shared by both
// views to keep the cost in pixel space. The fit is a plain least squares, so
// the points are expected to be an inlier set; robustness comes from the
// surrounding RANSAC.
//
// Returns false and leaves *F unchanged if it cannot be factorized (zero or
// numerically rank 1) or if the solve leaves non-finite parameters.
bool RefineFundamentalMatrixSampson(const std::vector<Eigen::Vector2d>& points1,
                                    const std::vector<Eigen::Vector2d>& points2,
                                    Eigen::Matrix3d* F);

// Fundamental matrix refiner for use as the local estimator of LO-RANSAC.
//
// Provides Refine() rather than Estimate(), so LO-RANSAC passes the current
// best model as the initial value, which a non-minimal solver cannot use.
class FundamentalMatrixSampsonEstimator {
 public:
  using X_t = Eigen::Vector2d;
  using Y_t = Eigen::Vector2d;
  using M_t = Eigen::Matrix3d;

  // The minimum number of samples needed to refine a model. Refining an
  // already determined model is only meaningful on an over-determined set.
  static const int kMinNumSamples = 8;

  // Refine *F in place, see RefineFundamentalMatrixSampson.
  static bool Refine(const std::vector<X_t>& points1,
                     const std::vector<Y_t>& points2,
                     M_t* F);

  // Squared Sampson error residuals, matching the estimators above.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& F,
                        std::vector<double>* residuals);
};

}  // namespace colmap
