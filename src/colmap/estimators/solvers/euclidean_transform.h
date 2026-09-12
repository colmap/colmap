// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/solvers/similarity_transform.h"

namespace colmap {

// N-D Euclidean transform estimator from corresponding point pairs in the
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
template <int kDim>
using EuclideanTransformEstimator = SimilarityTransformEstimator<kDim, false>;

}  // namespace colmap
