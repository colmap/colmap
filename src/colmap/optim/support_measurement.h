// SPDX-License-Identifier: BSD-3-Clause

#pragma once

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
  Support Evaluate(const std::vector<double>& residuals, double max_residual);

  // Compare the two supports.
  bool IsLeftBetter(const Support& left, const Support& right);
};

// Measure the support of a model by counting the number of unique inliers
// (e.g., visible 3D points), number of inliers, and summing all inlier
// residuals. Each sample should have an associated id. Samples with the same id
// are only counted once in num_unique_inliers. The support is better if it has
// more unique inliers, more inliers, and a smaller residual sum.
struct UniqueInlierSupportMeasurer {
  struct Support {
    // The number of unique inliers;
    size_t num_unique_inliers = 0;

    // The number of inliers.
    // This is still needed for determining the dynamic number of iterations.
    size_t num_inliers = 0;

    // The sum of all inlier residuals.
    double residual_sum = std::numeric_limits<double>::max();
  };

  explicit UniqueInlierSupportMeasurer(std::vector<size_t> unique_sample_ids);

  // Compute the support of the residuals.
  Support Evaluate(const std::vector<double>& residuals, double max_residual);

  // Compare the two supports.
  bool IsLeftBetter(const Support& left, const Support& right);

 private:
  std::vector<size_t> unique_sample_ids_;
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
  Support Evaluate(const std::vector<double>& residuals, double max_residual);

  // Compare the two supports.
  bool IsLeftBetter(const Support& left, const Support& right);
};

}  // namespace colmap
