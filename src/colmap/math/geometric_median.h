// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#pragma once

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Weiszfeld's algorithm for the L1 geometric median of a point set: the
// point minimizing the sum of Euclidean distances to every input point.
// Unlike the arithmetic mean, it is robust to outliers, so it is used
// (instead of an arbitrary point such as the first row) to pick a
// deterministic, outlier-conditioned reference point, e.g. an ENU tangent-
// plane origin. Undefined for an empty input.
inline Eigen::Vector3d GeometricMedian(
    const std::vector<Eigen::Vector3d>& points) {
  Eigen::Vector3d median = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& p : points) {
    median += p;
  }
  median /= static_cast<double>(points.size());

  constexpr int kMaxIters = 100;
  constexpr double kConvergenceTol = 1e-9;
  constexpr double kDegenerateDistTol = 1e-9;
  for (int iter = 0; iter < kMaxIters; ++iter) {
    Eigen::Vector3d numerator = Eigen::Vector3d::Zero();
    double weight_sum = 0.0;
    for (const Eigen::Vector3d& p : points) {
      const double dist = (p - median).norm();
      if (dist < kDegenerateDistTol) {
        // The current estimate coincides with a sample; skip it to avoid a
        // division by (near-)zero rather than perturbing the estimate.
        continue;
      }
      const double weight = 1.0 / dist;
      numerator += weight * p;
      weight_sum += weight;
    }
    if (weight_sum < kDegenerateDistTol) {
      break;
    }
    const Eigen::Vector3d next = numerator / weight_sum;
    const bool converged = (next - median).norm() < kConvergenceTol;
    median = next;
    if (converged) {
      break;
    }
  }
  return median;
}

}  // namespace colmap
