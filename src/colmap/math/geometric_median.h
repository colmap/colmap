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

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// L1 geometric median: the point minimizing the sum of Euclidean distances to
// every input point. Unlike the arithmetic mean it is not dragged by a small
// number of gross outliers, which is why it selects the ENU tangent origin: a
// single kilometre-scale GPS row must not move the origin, because every
// downstream transform is expressed relative to it.
//
// Uses the Vardi-Zhang modification of Weiszfeld's algorithm. The plain
// iteration is undefined when an iterate lands exactly on a sample, because
// that sample's weight 1/|p - m| diverges. Dropping the coinciding sample is
// the common shortcut and is wrong: it silently optimizes a different
// objective, and when a sample is repeated many times -- which is exactly what
// a stationary capture produces -- the dropped mass is the dominant term. The
// modification instead treats the coinciding sample as a subgradient term and
// takes the correctly damped step, so a repeated sample is a genuine fixed
// point rather than a discontinuity.
//
// Deterministic for a fixed input order: the accumulation order never varies,
// so repeated runs on the same vector are bit-identical.
inline Eigen::Vector3d GeometricMedian(
    const std::vector<Eigen::Vector3d>& points) {
  THROW_CHECK(!points.empty()) << "geometric median of an empty point set";
  for (const Eigen::Vector3d& p : points) {
    THROW_CHECK(p.allFinite()) << "geometric median of a non-finite point";
  }
  if (points.size() == 1) {
    return points[0];
  }

  // Distance below which an iterate counts as coinciding with a sample. ECEF
  // inputs are ~6.4e6 m, so this is ~1e-16 relative -- at the limit of double
  // precision rather than an arbitrary tolerance.
  constexpr double kCoincidentDist = 1e-9;
  constexpr double kConvergenceTol = 1e-9;
  constexpr int kMaxIters = 256;

  Eigen::Vector3d median = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& p : points) {
    median += p;
  }
  median /= static_cast<double>(points.size());

  for (int iter = 0; iter < kMaxIters; ++iter) {
    Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();
    double weight_sum = 0.0;
    // Number of samples coinciding with the current iterate, and the direction
    // the non-coinciding samples pull it. Both feed the Vardi-Zhang step.
    int num_coincident = 0;
    Eigen::Vector3d pull = Eigen::Vector3d::Zero();

    for (const Eigen::Vector3d& p : points) {
      const Eigen::Vector3d delta = p - median;
      const double dist = delta.norm();
      if (dist < kCoincidentDist) {
        ++num_coincident;
        continue;
      }
      const double weight = 1.0 / dist;
      weighted_sum += weight * p;
      weight_sum += weight;
      pull += delta * weight;
    }

    if (weight_sum <= 0.0) {
      // Every sample coincides with the iterate: it is exactly the median.
      break;
    }

    const Eigen::Vector3d weiszfeld = weighted_sum / weight_sum;

    Eigen::Vector3d next;
    if (num_coincident == 0) {
      next = weiszfeld;
    } else {
      // Vardi-Zhang: the coinciding samples contribute a subgradient of
      // magnitude num_coincident. When the remaining samples pull harder than
      // that, step toward them by the shortfall; otherwise the current point
      // is optimal and the iteration stops there.
      const double pull_norm = pull.norm();
      if (pull_norm <= static_cast<double>(num_coincident)) {
        break;
      }
      const double scale =
          1.0 - static_cast<double>(num_coincident) / pull_norm;
      next = median + scale * (weiszfeld - median);
    }

    THROW_CHECK(next.allFinite()) << "geometric median diverged";
    const bool converged = (next - median).norm() < kConvergenceTol;
    median = next;
    if (converged) {
      break;
    }
  }

  THROW_CHECK(median.allFinite()) << "geometric median is not finite";
  return median;
}

}  // namespace colmap
