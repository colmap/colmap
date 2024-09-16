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

#include "colmap/geometry/covariance.h"

#include "colmap/util/logging.h"

namespace colmap {

bool InsideUncertaintyInterval(const Eigen::Vector2d& mean,
                               const Eigen::Matrix2d& cov,
                               const Eigen::Vector2d& x,
                               double sigma_factor) {
  // Closed-form computation of Eigen values.
  const double a = cov(0, 0);
  const double b = cov(1, 0);
  const double c = cov(1, 1);
  const double term1 = 0.5 * (a + c);
  const double a_minus_c = a - c;
  const double term2 = std::sqrt(0.25 * a_minus_c * a_minus_c + b * b);
  const double lambda1 = term1 + term2;
  const double lambda2 = term1 - term2;
  // Compute orientation of uncertainty ellipse.
  const double theta = std::atan2(lambda1 - a, b);
  const double cos_theta = std::cos(theta);
  const double sin_theta = std::sin(theta);

  const Eigen::Vector2d x_centered = x - mean;
  Eigen::Vector2d x_aligned;
  x_aligned(0) = cos_theta * x_centered(0) + sin_theta * x_centered(1);
  x_aligned(1) = sin_theta * x_centered(0) - cos_theta * x_centered(1);

  // Major/minor axes of uncertainty ellipse are sqrt(lambda).
  const double val = x_aligned(0) * x_aligned(0) / lambda1 +
                     x_aligned(1) * x_aligned(1) / lambda2;

  return val <= sigma_factor * sigma_factor;
}

}  // namespace colmap
