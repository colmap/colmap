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

#include "colmap/estimators/fundamental_matrix_degensac.h"

#include "colmap/estimators/solvers/homography_matrix.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/homography_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/optim/loransac.h"
#include "colmap/util/logging.h"

#include <array>
#include <optional>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace colmap {
namespace {

// All C(7,3) = 35 triplets of the seven minimal-sample indices. Enumerated once
// so the sample degeneracy test does not recompute them per hypothesis.
constexpr int kNumSampleTriplets = 35;
constexpr std::array<std::array<int, 3>, kNumSampleTriplets> kSampleTriplets = {
    {
        {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 1, 5}, {0, 1, 6}, {0, 2, 3},
        {0, 2, 4}, {0, 2, 5}, {0, 2, 6}, {0, 3, 4}, {0, 3, 5}, {0, 3, 6},
        {0, 4, 5}, {0, 4, 6}, {0, 5, 6}, {1, 2, 3}, {1, 2, 4}, {1, 2, 5},
        {1, 2, 6}, {1, 3, 4}, {1, 3, 5}, {1, 3, 6}, {1, 4, 5}, {1, 4, 6},
        {1, 5, 6}, {2, 3, 4}, {2, 3, 5}, {2, 3, 6}, {2, 4, 5}, {2, 4, 6},
        {2, 5, 6}, {3, 4, 5}, {3, 4, 6}, {3, 5, 6}, {4, 5, 6},
    }};

}  // namespace

Eigen::Vector3d EpipoleFromFundamentalMatrix(const Eigen::Matrix3d& F) {
  // The left epipole e2 satisfies F^T e2 = 0, i.e. it is the left null vector
  // of F and thus the last left singular vector (last column of U).
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU);
  return svd.matrixU().col(2).normalized();
}

std::optional<Eigen::Matrix3d> HomographyFromFundamentalAndPoints(
    const Eigen::Matrix3d& F,
    const Eigen::Vector3d& epipole2,
    const std::array<Eigen::Vector2d, 3>& points1,
    const std::array<Eigen::Vector2d, 3>& points2) {
  // A = [e2]_x F is a particular homography compatible with F (H&Z
  // Result 13.6).
  const Eigen::Matrix3d A = CrossProductMatrix(epipole2) * F;

  Eigen::Matrix3d M;
  Eigen::Vector3d b;
  for (int i = 0; i < 3; ++i) {
    const Eigen::Vector3d x1 = points1[i].homogeneous();
    const Eigen::Vector3d x2 = points2[i].homogeneous();
    const Eigen::Vector3d x2_cross_e2 = x2.cross(epipole2);
    const double denom = x2_cross_e2.squaredNorm();
    if (denom < 1e-12) {
      // The point lies (near) the epipole, so the transfer is undefined.
      return std::nullopt;
    }
    b(i) = x2.cross(A * x1).dot(x2_cross_e2) / denom;
    M.row(i) = x1.transpose();
  }

  // Reject collinear first-image points, for which M is singular.
  Eigen::FullPivLU<Eigen::Matrix3d> lu(M);
  if (!lu.isInvertible()) {
    return std::nullopt;
  }

  const Eigen::Vector3d Minv_b = lu.solve(b);
  return A - epipole2 * Minv_b.transpose();
}

std::optional<Eigen::Matrix3d> DetectSampleHDegeneracy(
    const Eigen::Matrix3d& F,
    const std::vector<Eigen::Vector2d>& sample_points1,
    const std::vector<Eigen::Vector2d>& sample_points2,
    const double h_max_residual,
    const int min_sample_h_inliers) {
  THROW_CHECK_EQ(sample_points1.size(), 7);
  THROW_CHECK_EQ(sample_points2.size(), 7);

  const Eigen::Vector3d epipole2 = EpipoleFromFundamentalMatrix(F);

  int best_num_consistent = min_sample_h_inliers - 1;
  std::optional<Eigen::Matrix3d> best_H;
  for (const auto& triplet : kSampleTriplets) {
    const std::array<Eigen::Vector2d, 3> tri_points1 = {
        sample_points1[triplet[0]],
        sample_points1[triplet[1]],
        sample_points1[triplet[2]]};
    const std::array<Eigen::Vector2d, 3> tri_points2 = {
        sample_points2[triplet[0]],
        sample_points2[triplet[1]],
        sample_points2[triplet[2]]};

    const std::optional<Eigen::Matrix3d> H = HomographyFromFundamentalAndPoints(
        F, epipole2, tri_points1, tri_points2);
    if (!H.has_value()) {
      continue;
    }

    int num_consistent = 0;
    for (size_t i = 0; i < sample_points1.size(); ++i) {
      if (ComputeSquaredHomographyError(
              sample_points1[i], sample_points2[i], *H) <= h_max_residual) {
        ++num_consistent;
      }
    }
    if (num_consistent > best_num_consistent) {
      best_num_consistent = num_consistent;
      best_H = H;
    }
  }

  return best_H;
}

bool IsSampleHDegenerate(const Eigen::Matrix3d& F,
                         const std::vector<Eigen::Vector2d>& sample_points1,
                         const std::vector<Eigen::Vector2d>& sample_points2,
                         const double h_max_residual,
                         const int min_sample_h_inliers) {
  return DetectSampleHDegeneracy(F,
                                 sample_points1,
                                 sample_points2,
                                 h_max_residual,
                                 min_sample_h_inliers)
      .has_value();
}

std::optional<Eigen::Matrix3d> FundamentalFromPlaneAndParallax(
    const Eigen::Matrix3d& seed_H,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const double sampson_max_residual,
    const double h_max_residual,
    const int max_trials) {
  THROW_CHECK_EQ(points1.size(), points2.size());

  // The seed homography is built from only three sample correspondences (via
  // the plane-corrupted sample fundamental matrix), so it is only approximate.
  // Refit it on all of its consistent correspondences to obtain an accurate
  // dominant-plane homography; otherwise the off-plane classification below is
  // polluted with plane points and the epipole recovery becomes unreliable.
  Eigen::Matrix3d H = seed_H;
  HomographyMatrixEstimator homography_estimator;
  std::vector<Eigen::Vector2d> plane_points1;
  std::vector<Eigen::Vector2d> plane_points2;
  std::vector<Eigen::Matrix3d> homographies;
  constexpr int kNumRefitIters = 2;
  for (int iter = 0; iter < kNumRefitIters; ++iter) {
    plane_points1.clear();
    plane_points2.clear();
    for (size_t i = 0; i < points1.size(); ++i) {
      if (ComputeSquaredHomographyError(points1[i], points2[i], H) <=
          h_max_residual) {
        plane_points1.push_back(points1[i]);
        plane_points2.push_back(points2[i]);
      }
    }
    if (plane_points1.size() <
        static_cast<size_t>(HomographyMatrixEstimator::kMinNumSamples)) {
      break;
    }
    homographies.clear();
    homography_estimator.Estimate(plane_points1, plane_points2, &homographies);
    if (homographies.empty()) {
      break;
    }
    H = homographies[0];
  }

  // Off-plane correspondences carry the parallax that constrains the epipole.
  std::vector<size_t> off_plane_idxs;
  off_plane_idxs.reserve(points1.size());
  for (size_t i = 0; i < points1.size(); ++i) {
    if (ComputeSquaredHomographyError(points1[i], points2[i], H) >
        h_max_residual) {
      off_plane_idxs.push_back(i);
    }
  }
  if (off_plane_idxs.size() < 2) {
    return std::nullopt;
  }

  // Precompute the transferred plane points H * x1 for the off-plane
  // candidates.
  std::vector<Eigen::Vector3d> lines(off_plane_idxs.size());
  for (size_t i = 0; i < off_plane_idxs.size(); ++i) {
    const size_t idx = off_plane_idxs[i];
    lines[i] = points2[idx].homogeneous().cross(H * points1[idx].homogeneous());
  }

  InlierSupportMeasurer support_measurer;
  InlierSupportMeasurer::Support best_support;
  std::optional<Eigen::Matrix3d> best_F;
  std::vector<double> residuals;

  const size_t num_off_plane = off_plane_idxs.size();
  const size_t num_pairs = num_off_plane * (num_off_plane - 1) / 2;
  const size_t num_trials =
      std::min<size_t>(static_cast<size_t>(std::max(max_trials, 1)), num_pairs);
  for (size_t trial = 0; trial < num_trials; ++trial) {
    size_t a = RandomUniformInteger<size_t>(0, num_off_plane - 1);
    size_t b = RandomUniformInteger<size_t>(0, num_off_plane - 1);
    if (a == b) {
      b = (b + 1) % num_off_plane;
    }

    // The epipole is the intersection of the two parallax lines.
    const Eigen::Vector3d epipole2 = lines[a].cross(lines[b]);
    if (epipole2.norm() < 1e-9) {
      continue;
    }
    const Eigen::Matrix3d F = CrossProductMatrix(epipole2.normalized()) * H;

    ComputeSquaredSampsonError(points1, points2, F, &residuals);
    const auto support =
        support_measurer.Evaluate(residuals, sampson_max_residual);
    if (support_measurer.IsLeftBetter(support, best_support)) {
      best_support = support;
      best_F = F;
    }
  }

  return best_F;
}

FundamentalMatrixDegensacEstimator::FundamentalMatrixDegensacEstimator(
    const std::vector<Eigen::Vector2d>* points1,
    const std::vector<Eigen::Vector2d>* points2,
    const double sampson_max_residual,
    const double h_max_residual,
    const int min_sample_h_inliers,
    const int max_plane_parallax_trials)
    : points1_(THROW_CHECK_NOTNULL(points1)),
      points2_(THROW_CHECK_NOTNULL(points2)),
      sampson_max_residual_(sampson_max_residual),
      h_max_residual_(h_max_residual),
      min_sample_h_inliers_(min_sample_h_inliers),
      max_plane_parallax_trials_(max_plane_parallax_trials) {}

void FundamentalMatrixDegensacEstimator::Estimate(
    const std::vector<X_t>& sample_points1,
    const std::vector<Y_t>& sample_points2,
    std::vector<M_t>* models) const {
  THROW_CHECK(models != nullptr);

  std::vector<M_t> sample_models;
  FundamentalMatrixSevenPointEstimator::Estimate(
      sample_points1, sample_points2, &sample_models);

  models->clear();
  models->reserve(sample_models.size());
  for (const auto& sample_model : sample_models) {
    const std::optional<Eigen::Matrix3d> plane_H =
        DetectSampleHDegeneracy(sample_model,
                                sample_points1,
                                sample_points2,
                                h_max_residual_,
                                min_sample_h_inliers_);
    if (!plane_H.has_value()) {
      // Non-degenerate sample: keep the minimal-solver hypothesis as-is.
      models->push_back(sample_model);
      continue;
    }
    // H-degenerate sample: replace the plane-corrupted hypothesis by a
    // plane-and-parallax completion, or drop it if there is no usable parallax.
    const std::optional<Eigen::Matrix3d> completed_model =
        FundamentalFromPlaneAndParallax(*plane_H,
                                        *points1_,
                                        *points2_,
                                        sampson_max_residual_,
                                        h_max_residual_,
                                        max_plane_parallax_trials_);
    if (completed_model.has_value()) {
      models->push_back(*completed_model);
    }
  }
}

void FundamentalMatrixDegensacEstimator::Residuals(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    const M_t& F,
    std::vector<double>* residuals) const {
  ComputeSquaredSampsonError(points1, points2, F, residuals);
}

FundamentalMatrixDegensac::FundamentalMatrixDegensac(Options options)
    : options_(options) {
  options_.ransac.Check();
}

FundamentalMatrixDegensac::Report FundamentalMatrixDegensac::Estimate(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2) {
  const double sampson_max_residual =
      options_.ransac.max_error * options_.ransac.max_error;
  const double h_max_error = options_.h_consistency_max_error > 0
                                 ? options_.h_consistency_max_error
                                 : options_.ransac.max_error;
  const double h_max_residual = h_max_error * h_max_error;

  // Inject the DEGENSAC minimal solver into LO-RANSAC, using the 8-point solver
  // as the local optimizer, mirroring the plain fundamental matrix estimation.
  FundamentalMatrixDegensacEstimator estimator(
      &points1,
      &points2,
      sampson_max_residual,
      h_max_residual,
      options_.min_sample_h_inliers,
      options_.max_plane_parallax_trials);
  LORANSAC<FundamentalMatrixDegensacEstimator,
           FundamentalMatrixEightPointEstimator>
      ransac(options_.ransac, estimator);
  const auto ransac_report = ransac.Estimate(points1, points2);

  // Copy into the stable, estimator-agnostic report type.
  Report report;
  report.success = ransac_report.success;
  report.num_trials = ransac_report.num_trials;
  report.support = ransac_report.support;
  report.inlier_mask = ransac_report.inlier_mask;
  report.model = ransac_report.model;
  return report;
}

}  // namespace colmap
