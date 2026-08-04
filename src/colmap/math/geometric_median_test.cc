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

#include "colmap/math/geometric_median.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Objective the geometric median minimizes: sum of distances to all samples.
double CostAt(const std::vector<Eigen::Vector3d>& points,
              const Eigen::Vector3d& x) {
  double cost = 0.0;
  for (const Eigen::Vector3d& p : points) {
    cost += (p - x).norm();
  }
  return cost;
}

// Asserts no small perturbation lowers the objective, i.e. the returned point
// really is a minimizer and not merely wherever the iteration stopped.
void ExpectLocallyOptimal(const std::vector<Eigen::Vector3d>& points,
                          const Eigen::Vector3d& median,
                          const double step) {
  const double cost = CostAt(points, median);
  for (int axis = 0; axis < 3; ++axis) {
    for (const double sign : {-1.0, 1.0}) {
      Eigen::Vector3d probe = median;
      probe(axis) += sign * step;
      EXPECT_GE(CostAt(points, probe), cost - 1e-6)
          << "a perturbation along axis " << axis << " lowered the objective";
    }
  }
}

TEST(GeometricMedian, EmptyInputThrows) {
  EXPECT_ANY_THROW(GeometricMedian({}));
}

TEST(GeometricMedian, NonFiniteInputThrows) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_ANY_THROW(
      GeometricMedian({Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(nan, 0, 0)}));
  EXPECT_ANY_THROW(
      GeometricMedian({Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, inf, 0)}));
}

TEST(GeometricMedian, SinglePointIsItself) {
  const Eigen::Vector3d p(1.0, -2.0, 3.0);
  EXPECT_LT((GeometricMedian({p}) - p).norm(), 1e-12);
}

TEST(GeometricMedian, IdenticalPointsReturnThatPoint) {
  const Eigen::Vector3d p(4.0, 5.0, 6.0);
  const std::vector<Eigen::Vector3d> points(7, p);
  EXPECT_LT((GeometricMedian(points) - p).norm(), 1e-9);
}

TEST(GeometricMedian, TwoPointsLieOnTheSegment) {
  const Eigen::Vector3d a(0, 0, 0);
  const Eigen::Vector3d b(10, 0, 0);
  const Eigen::Vector3d m = GeometricMedian({a, b});
  // Every point on the segment is optimal; require it to be on the segment.
  EXPECT_LT(std::abs(m.y()), 1e-6);
  EXPECT_LT(std::abs(m.z()), 1e-6);
  EXPECT_GE(m.x(), -1e-6);
  EXPECT_LE(m.x(), 10.0 + 1e-6);
}

TEST(GeometricMedian, SymmetricSetReturnsCentre) {
  const std::vector<Eigen::Vector3d> points = {Eigen::Vector3d(-1, 0, 0),
                                               Eigen::Vector3d(1, 0, 0),
                                               Eigen::Vector3d(0, -1, 0),
                                               Eigen::Vector3d(0, 1, 0)};
  EXPECT_LT(GeometricMedian(points).norm(), 1e-6);
}

TEST(GeometricMedian, CoincidentDominantSampleIsAFixedPoint) {
  // The failure mode the Vardi-Zhang modification exists for: a sample repeated
  // enough times that it *is* the median. Dropping coinciding samples, as plain
  // Weiszfeld shortcuts do, would let the two stragglers pull the result away.
  std::vector<Eigen::Vector3d> points(10, Eigen::Vector3d(2.0, 3.0, 4.0));
  points.emplace_back(50.0, 3.0, 4.0);
  points.emplace_back(2.0, 90.0, 4.0);

  const Eigen::Vector3d median = GeometricMedian(points);
  EXPECT_LT((median - Eigen::Vector3d(2.0, 3.0, 4.0)).norm(), 1e-6);
  ExpectLocallyOptimal(points, median, 1e-3);
}

TEST(GeometricMedian, CoincidentButOutvotedSampleMoves) {
  // Complement of the previous case: too few duplicates to hold the point, so
  // the iteration must leave it rather than reporting a false fixed point.
  std::vector<Eigen::Vector3d> points(2, Eigen::Vector3d(0, 0, 0));
  for (int i = 0; i < 9; ++i) {
    points.emplace_back(10.0, 0.0, 0.0);
  }
  const Eigen::Vector3d median = GeometricMedian(points);
  EXPECT_GT(median.x(), 1.0);
  ExpectLocallyOptimal(points, median, 1e-3);
}

TEST(GeometricMedian, CollinearData) {
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 9; ++i) {
    points.emplace_back(static_cast<double>(i), 0.0, 0.0);
  }
  const Eigen::Vector3d median = GeometricMedian(points);
  EXPECT_NEAR(median.x(), 4.0, 1e-3);
  EXPECT_LT(std::abs(median.y()), 1e-6);
}

TEST(GeometricMedian, GrossOutlierDoesNotDragTheResult) {
  // The property the ENU origin depends on. A kilometre-scale GPS row is a
  // valid archive row (it is rejected later by robust fitting, not at import),
  // so the origin must not follow it.
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 20; ++i) {
    points.emplace_back(
        static_cast<double>(i % 5), static_cast<double>(i % 3), 0.0);
  }
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& p : points) mean += p;

  points.emplace_back(100000.0, 100000.0, 100000.0);
  mean += points.back();
  mean /= static_cast<double>(points.size());

  const Eigen::Vector3d median = GeometricMedian(points);
  EXPECT_LT(median.norm(), 10.0) << "outlier moved the median";
  EXPECT_GT(mean.norm(), 1000.0) << "the mean should be dragged; if not, the "
                                    "test no longer demonstrates robustness";
}

TEST(GeometricMedian, EcefScaleCoordinates) {
  // Real inputs are ~6.4e6 m. Convergence tolerances must still behave there.
  const Eigen::Vector3d base(1369512.85, -4014717.23, 4747304.12);
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 12; ++i) {
    points.push_back(base + Eigen::Vector3d(i * 0.5, -i * 0.25, i * 0.1));
  }
  const Eigen::Vector3d median = GeometricMedian(points);
  EXPECT_TRUE(median.allFinite());
  EXPECT_LT((median - base).norm(), 50.0);
  ExpectLocallyOptimal(points, median, 1e-2);
}

TEST(GeometricMedian, DeterministicForAFixedInputOrder) {
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 25; ++i) {
    points.emplace_back(std::sin(i), std::cos(i * 1.7), std::sin(i * 0.3));
  }
  const Eigen::Vector3d first = GeometricMedian(points);
  for (int repeat = 0; repeat < 5; ++repeat) {
    const Eigen::Vector3d again = GeometricMedian(points);
    EXPECT_EQ(first.x(), again.x());
    EXPECT_EQ(first.y(), again.y());
    EXPECT_EQ(first.z(), again.z());
  }
}

TEST(GeometricMedian, PermutationChangesResultOnlyNegligibly) {
  // Floating-point accumulation order differs under permutation, so the result
  // is not required to be bit-identical -- only to agree far below any
  // tolerance that matters for a tangent-plane origin.
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 30; ++i) {
    points.emplace_back(std::sin(i * 2.1), std::cos(i), std::sin(i * 0.7));
  }
  const Eigen::Vector3d reference = GeometricMedian(points);

  std::vector<int> order(points.size());
  std::iota(order.begin(), order.end(), 0);
  for (int shift = 1; shift < 4; ++shift) {
    std::rotate(order.begin(), order.begin() + shift, order.end());
    std::vector<Eigen::Vector3d> permuted;
    permuted.reserve(points.size());
    for (const int index : order) permuted.push_back(points[index]);
    EXPECT_LT((GeometricMedian(permuted) - reference).norm(), 1e-6);
  }
}

TEST(GeometricMedian, BeatsTheMeanOnTheObjective) {
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 15; ++i) {
    points.emplace_back(static_cast<double>(i), 0.0, 0.0);
  }
  points.emplace_back(5000.0, 0.0, 0.0);

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& p : points) mean += p;
  mean /= static_cast<double>(points.size());

  EXPECT_LT(CostAt(points, GeometricMedian(points)), CostAt(points, mean));
}

}  // namespace
}  // namespace colmap
