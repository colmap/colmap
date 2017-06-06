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

#ifndef COLMAP_SRC_ESTIMATORS_AFFINE_TRANSFORM_H_
#define COLMAP_SRC_ESTIMATORS_AFFINE_TRANSFORM_H_

#include <vector>

#include <Eigen/Core>

#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

class AffineTransformEstimator {
 public:
  typedef Eigen::Vector2d X_t;
  typedef Eigen::Vector2d Y_t;
  typedef Eigen::Matrix<double, 2, 3> M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  // Estimate the affine transformation from at least 3 correspondences.
  static std::vector<M_t> Estimate(const std::vector<X_t>& points1,
                                   const std::vector<Y_t>& points2);

  // Compute the squared transformation error.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2, const M_t& E,
                        std::vector<double>* residuals);
};

}  // namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_AFFINE_TRANSFORM_H_
