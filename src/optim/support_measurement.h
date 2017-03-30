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

#ifndef COLMAP_SRC_OPTIM_SUPPORT_MEASUREMENT_H_
#define COLMAP_SRC_OPTIM_SUPPORT_MEASUREMENT_H_

#include <cstddef>
#include <limits>
#include <vector>

namespace colmap {

// Measure the support of a model by counting the number of inliers and
// summing all inlier residuals. The support is better if it has more inliers
// and a smaller residual sum.
struct InlierSupportMeasurer {
  struct Support {
    // The number of inliers.
    size_t num_inliers = 0;

    // The sum of all inlier residuals.
    double residual_sum = std::numeric_limits<double>::max();
  };

  // Compute the support of the residuals.
  Support Evaluate(const std::vector<double>& residuals,
                   const double max_residual);

  // Compare the two supports and return the better support.
  bool Compare(const Support& support1, const Support& support2);
};

// Measure the support of a model by its fitness to the data as used in MSAC.
// A support is better if it has a smaller MSAC score.
struct MEstimatorSupportMeasurer {
  struct Support {
    // The number of inliers.
    size_t num_inliers = 0;

    // The MSAC score, defined as the truncated sum of residuals.
    double score = std::numeric_limits<double>::max();
  };

  // Compute the support of the residuals.
  Support Evaluate(const std::vector<double>& residuals,
                   const double max_residual);

  // Compare the two supports and return the better support.
  bool Compare(const Support& support1, const Support& support2);
};

}  // namespace colmap

#endif  // COLMAP_SRC_OPTIM_SUPPORT_MEASUREMENT_H_
