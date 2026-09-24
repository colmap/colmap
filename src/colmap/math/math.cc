// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/math/math.h"

namespace colmap {

double StudentTTwoDofThreshold(const double chi2_threshold, const double dof) {
  THROW_CHECK_GT(chi2_threshold, 0);
  THROW_CHECK_GT(dof, 0);
  if (std::isinf(dof)) {
    return chi2_threshold;
  }
  // The squared Mahalanobis distance d^2 of bivariate Student-t noise has the
  // survival function (1 + d^2 / dof)^(-dof / 2), and exp(-d^2 / 2) in the
  // Gaussian limit.
  const auto quantile = [dof](const double tail_probability) {
    return dof * (std::pow(tail_probability, -2.0 / dof) - 1.0);
  };
  const double gaussian_median = 2.0 * std::log(2.0);
  return quantile(std::exp(-0.5 * chi2_threshold)) * gaussian_median /
         quantile(0.5);
}

// Implementation based on: https://blog.plover.com/math/choose.html
uint64_t NChooseK(uint64_t n, uint64_t k) {
  if (n == 0 || n < k) {
    return 0;
  }

  uint64_t r = 1;
  for (uint64_t d = 1; d <= k; ++d) {
    r *= n--;
    r /= d;
  }
  return r;
}

}  // namespace colmap
