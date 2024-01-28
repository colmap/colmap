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

#include "colmap/optim/support_measurement.h"

#include "colmap/util/logging.h"

#include <unordered_set>

namespace colmap {

InlierSupportMeasurer::Support InlierSupportMeasurer::Evaluate(
    const std::vector<double>& residuals, const double max_residual) {
  Support support;
  support.num_inliers = 0;
  support.residual_sum = 0;

  for (const auto residual : residuals) {
    if (residual <= max_residual) {
      support.num_inliers += 1;
      support.residual_sum += residual;
    }
  }

  return support;
}

bool InlierSupportMeasurer::Compare(const Support& support1,
                                    const Support& support2) {
  if (support1.num_inliers > support2.num_inliers) {
    return true;
  } else {
    return support1.num_inliers == support2.num_inliers &&
           support1.residual_sum < support2.residual_sum;
  }
}

UniqueInlierSupportMeasurer::Support UniqueInlierSupportMeasurer::Evaluate(
    const std::vector<double>& residuals, const double max_residual) {
  THROW_CHECK_EQ(residuals.size(), unique_sample_ids_.size());
  Support support;
  support.num_inliers = 0;
  support.num_unique_inliers = 0;
  support.residual_sum = 0;

  std::unordered_set<size_t> inlier_point_ids;
  for (size_t idx = 0; idx < residuals.size(); ++idx) {
    if (residuals[idx] <= max_residual) {
      support.num_inliers += 1;
      inlier_point_ids.insert(unique_sample_ids_[idx]);
      support.residual_sum += residuals[idx];
    }
  }
  support.num_unique_inliers = inlier_point_ids.size();
  return support;
}

bool UniqueInlierSupportMeasurer::Compare(const Support& support1,
                                          const Support& support2) {
  if (support1.num_unique_inliers > support2.num_unique_inliers) {
    return true;
  } else if (support1.num_unique_inliers == support2.num_unique_inliers) {
    if (support1.num_inliers > support2.num_inliers) {
      return true;
    } else {
      return support1.num_inliers == support2.num_inliers &&
             support1.residual_sum < support2.residual_sum;
    }
  } else {
    return false;
  }
}

MEstimatorSupportMeasurer::Support MEstimatorSupportMeasurer::Evaluate(
    const std::vector<double>& residuals, const double max_residual) {
  Support support;
  support.num_inliers = 0;
  support.score = 0;

  for (const auto residual : residuals) {
    if (residual <= max_residual) {
      support.num_inliers += 1;
      support.score += residual;
    } else {
      support.score += max_residual;
    }
  }

  return support;
}

bool MEstimatorSupportMeasurer::Compare(const Support& support1,
                                        const Support& support2) {
  return support1.score < support2.score;
}

}  // namespace colmap
