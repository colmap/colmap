// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/optim/support_measurement.h"

#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"

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

bool InlierSupportMeasurer::IsLeftBetter(const Support& left,
                                         const Support& right) {
  if (left.num_inliers > right.num_inliers) {
    return true;
  } else {
    return left.num_inliers == right.num_inliers &&
           left.residual_sum < right.residual_sum;
  }
}

UniqueInlierSupportMeasurer::UniqueInlierSupportMeasurer(
    std::vector<size_t> unique_sample_ids)
    : unique_sample_ids_(std::move(unique_sample_ids)) {}

UniqueInlierSupportMeasurer::Support UniqueInlierSupportMeasurer::Evaluate(
    const std::vector<double>& residuals, const double max_residual) {
  THROW_CHECK_EQ(residuals.size(), unique_sample_ids_.size());
  Support support;
  support.num_inliers = 0;
  support.num_unique_inliers = 0;
  support.residual_sum = 0;

  FlatHashSet<size_t> inlier_point_ids;
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

bool UniqueInlierSupportMeasurer::IsLeftBetter(const Support& left,
                                               const Support& right) {
  if (left.num_unique_inliers > right.num_unique_inliers) {
    return true;
  } else if (left.num_unique_inliers == right.num_unique_inliers) {
    if (left.num_inliers > right.num_inliers) {
      return true;
    } else {
      return left.num_inliers == right.num_inliers &&
             left.residual_sum < right.residual_sum;
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

bool MEstimatorSupportMeasurer::IsLeftBetter(const Support& left,
                                             const Support& right) {
  return left.score < right.score;
}

}  // namespace colmap
