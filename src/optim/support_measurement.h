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
