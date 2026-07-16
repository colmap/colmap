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

#include "colmap/estimators/solvers/fundamental_matrix.h"
#include "colmap/estimators/solvers/homography_matrix.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/homography_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/optim/loransac.h"
#include "colmap/util/logging.h"

#include <array>
#include <cmath>
#include <optional>
#include <utility>

#include <Eigen/Geometry>
#include <Eigen/LU>

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

// Number of triplets randomly sampled from a non-minimal sample when searching
// for the dominant plane (a minimal sample enumerates all 35 exhaustively).
constexpr int kNumSampledTriplets = 50;

}  // namespace

Eigen::Vector3d EpipoleFromFundamentalMatrix(const Eigen::Matrix3d& F) {
  // The epipole e2 is the left null vector (F^T e2 = 0), i.e. it is orthogonal
  // to the column space of the rank-2 matrix F and thus parallel to the cross
  // product of two of its columns. Using the column pair with the largest cross
  // product is numerically stable and avoids a full SVD.
  const Eigen::Vector3d e01 = F.col(0).cross(F.col(1));
  const Eigen::Vector3d e02 = F.col(0).cross(F.col(2));
  const Eigen::Vector3d e12 = F.col(1).cross(F.col(2));
  const double n01 = e01.squaredNorm();
  const double n02 = e02.squaredNorm();
  const double n12 = e12.squaredNorm();
  if (n01 >= n02 && n01 >= n12) {
    return e01.normalized();
  }
  return n02 >= n12 ? e02.normalized() : e12.normalized();
}

namespace {

// Plane-induced homography compatible with the epipolar geometry from three
// correspondences (H&Z Result 13.6), given the precomputed A = [e2]_x F and the
// epipole e2. Splitting this out lets the degeneracy test reuse A across all
// triplets of a sample instead of recomputing it each time.
std::optional<Eigen::Matrix3d> HomographyFromCompatible(
    const Eigen::Matrix3d& A,
    const Eigen::Vector3d& epipole2,
    const std::array<Eigen::Vector2d, 3>& points1,
    const std::array<Eigen::Vector2d, 3>& points2) {
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

  // Reject collinear first-image points (scale-invariant singularity check),
  // then solve with the closed-form 3x3 inverse.
  const double det = M.determinant();
  const double scale = M.row(0).norm() * M.row(1).norm() * M.row(2).norm();
  if (scale < 1e-12 || std::abs(det) < 1e-9 * scale) {
    return std::nullopt;
  }
  const Eigen::Vector3d Minv_b = M.inverse() * b;
  return A - epipole2 * Minv_b.transpose();
}

}  // namespace

std::optional<Eigen::Matrix3d> HomographyFromFundamentalAndPoints(
    const Eigen::Matrix3d& F,
    const Eigen::Vector3d& epipole2,
    const std::array<Eigen::Vector2d, 3>& points1,
    const std::array<Eigen::Vector2d, 3>& points2) {
  // A = [e2]_x F is a particular homography compatible with F (H&Z
  // Result 13.6).
  return HomographyFromCompatible(
      CrossProductMatrix(epipole2) * F, epipole2, points1, points2);
}

std::optional<Eigen::Matrix3d> DetectSampleHDegeneracy(
    const Eigen::Matrix3d& F,
    const std::vector<Eigen::Vector2d>& sample_points1,
    const std::vector<Eigen::Vector2d>& sample_points2,
    const double h_max_residual,
    const double min_sample_h_inlier_ratio) {
  const size_t num_samples = sample_points1.size();
  THROW_CHECK_EQ(sample_points1.size(), sample_points2.size());
  THROW_CHECK_GE(num_samples, 3);

  const Eigen::Vector3d epipole2 = EpipoleFromFundamentalMatrix(F);
  // A = [e2]_x F is shared by all triplets; compute it once.
  const Eigen::Matrix3d A = CrossProductMatrix(epipole2) * F;
  const int min_consistent = std::max<int>(
      3,
      static_cast<int>(std::llround(min_sample_h_inlier_ratio * num_samples)));

  int best_num_consistent = min_consistent - 1;
  std::optional<Eigen::Matrix3d> best_H;

  // Returns true if all sample points are consistent (nothing left to improve).
  const auto evaluate_triplet = [&](int i, int j, int k) {
    const std::array<Eigen::Vector2d, 3> tri_points1 = {
        sample_points1[i], sample_points1[j], sample_points1[k]};
    const std::array<Eigen::Vector2d, 3> tri_points2 = {
        sample_points2[i], sample_points2[j], sample_points2[k]};
    const std::optional<Eigen::Matrix3d> H =
        HomographyFromCompatible(A, epipole2, tri_points1, tri_points2);
    if (!H.has_value()) {
      return false;
    }
    int num_consistent = 0;
    for (size_t p = 0; p < num_samples; ++p) {
      if (ComputeSquaredHomographyError(
              sample_points1[p], sample_points2[p], *H) <= h_max_residual) {
        ++num_consistent;
      }
    }
    if (num_consistent > best_num_consistent) {
      best_num_consistent = num_consistent;
      best_H = H;
    }
    return best_num_consistent >= static_cast<int>(num_samples);
  };

  if (num_samples ==
      static_cast<size_t>(FundamentalMatrixDegensacEstimator::kMinNumSamples)) {
    // Minimal sample: exhaustively enumerate all triplets.
    for (const auto& triplet : kSampleTriplets) {
      if (evaluate_triplet(triplet[0], triplet[1], triplet[2])) {
        break;
      }
    }
  } else {
    // Non-minimal sample (e.g. a local-optimization inlier set): sample
    // triplets. On a plane-dominated sample most triplets lie on the plane, so
    // a modest number reliably discovers it.
    const int last = static_cast<int>(num_samples) - 1;
    for (int t = 0; t < kNumSampledTriplets; ++t) {
      const int i = RandomUniformInteger<int>(0, last);
      const int j = RandomUniformInteger<int>(0, last);
      const int k = RandomUniformInteger<int>(0, last);
      if (i == j || j == k || i == k) {
        continue;
      }
      if (evaluate_triplet(i, j, k)) {
        break;
      }
    }
  }

  return best_H;
}

bool IsSampleHDegenerate(const Eigen::Matrix3d& F,
                         const std::vector<Eigen::Vector2d>& sample_points1,
                         const std::vector<Eigen::Vector2d>& sample_points2,
                         const double h_max_residual,
                         const double min_sample_h_inlier_ratio) {
  return DetectSampleHDegeneracy(F,
                                 sample_points1,
                                 sample_points2,
                                 h_max_residual,
                                 min_sample_h_inlier_ratio)
      .has_value();
}

namespace {

// Squared forward transfer error of a correspondence under a homography.
double TransferError(const Eigen::Vector2d& point1,
                     const Eigen::Vector2d& point2,
                     const Eigen::Matrix3d& H) {
  return ComputeSquaredHomographyError(point1, point2, H);
}

// Draws `num` distinct entries of `src` into the front of `scratch` (a working
// copy of `src`) via a partial Fisher-Yates shuffle.
void SampleDistinct(size_t num, std::vector<size_t>* scratch) {
  const size_t n = scratch->size();
  for (size_t i = 0; i < num; ++i) {
    const size_t j = i + RandomUniformInteger<size_t>(0, n - 1 - i);
    std::swap((*scratch)[i], (*scratch)[j]);
  }
}

}  // namespace

std::optional<Eigen::Matrix3d> FundamentalFromPlaneAndParallax(
    const Eigen::Matrix3d& seed_H,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const double sampson_max_residual,
    const double plane_max_residual,
    const double off_plane_min_residual,
    const int max_trials) {
  THROW_CHECK_EQ(points1.size(), points2.size());

  // The seed homography is built from only three sample correspondences (via
  // the plane-corrupted sample fundamental matrix), so it is only approximate.
  // Refit it on all of its plane inliers to obtain an accurate dominant-plane
  // homography; otherwise the off-plane classification below is polluted with
  // plane points and the epipole recovery becomes unreliable.
  Eigen::Matrix3d H = seed_H;
  HomographyMatrixEstimator homography_estimator;
  std::vector<size_t> plane_idxs;
  std::vector<Eigen::Vector2d> plane_points1;
  std::vector<Eigen::Vector2d> plane_points2;
  std::vector<Eigen::Matrix3d> homographies;
  constexpr int kNumRefitIters = 2;
  // A homography is well determined by a modest, spatially spread subset, so
  // cap the number of correspondences used for the DLT to avoid an O(N) SVD
  // when the dominant plane has many inliers.
  constexpr size_t kMaxHomographyFitPoints = 64;
  for (int iter = 0; iter < kNumRefitIters; ++iter) {
    plane_idxs.clear();
    for (size_t i = 0; i < points1.size(); ++i) {
      if (TransferError(points1[i], points2[i], H) <= plane_max_residual) {
        plane_idxs.push_back(i);
      }
    }
    if (plane_idxs.size() <
        static_cast<size_t>(HomographyMatrixEstimator::kMinNumSamples)) {
      break;
    }
    const size_t num_fit = std::min(kMaxHomographyFitPoints, plane_idxs.size());
    if (plane_idxs.size() > num_fit) {
      SampleDistinct(num_fit, &plane_idxs);
    }
    plane_points1.clear();
    plane_points2.clear();
    for (size_t i = 0; i < num_fit; ++i) {
      plane_points1.push_back(points1[plane_idxs[i]]);
      plane_points2.push_back(points2[plane_idxs[i]]);
    }
    homographies.clear();
    homography_estimator.Estimate(plane_points1, plane_points2, &homographies);
    if (homographies.empty()) {
      break;
    }
    H = homographies[0];
  }

  // The dominant-plane homography is now fixed, so the transfer error of every
  // correspondence against it is reused below (off-plane classification here
  // and the plane/off-plane split for the mixed-sample refit) instead of
  // recomputed.
  std::vector<double> plane_transfer_errors(points1.size());
  for (size_t i = 0; i < points1.size(); ++i) {
    plane_transfer_errors[i] = TransferError(points1[i], points2[i], H);
  }

  // Only correspondences well off the plane carry reliable parallax: near-plane
  // points have tiny, noise-dominated parallax that destabilizes the epipole.
  std::vector<size_t> off_plane_idxs;
  off_plane_idxs.reserve(points1.size());
  for (size_t i = 0; i < points1.size(); ++i) {
    if (plane_transfer_errors[i] > off_plane_min_residual) {
      off_plane_idxs.push_back(i);
    }
  }
  if (off_plane_idxs.size() < 2) {
    return std::nullopt;
  }

  // Precompute the transferred plane points H * x1 and the off-plane points, so
  // the epipole search below scores over only the off-plane subset.
  const size_t num_off_plane = off_plane_idxs.size();
  std::vector<Eigen::Vector3d> lines(num_off_plane);
  std::vector<Eigen::Vector2d> off_points1(num_off_plane);
  std::vector<Eigen::Vector2d> off_points2(num_off_plane);
  for (size_t i = 0; i < num_off_plane; ++i) {
    const size_t idx = off_plane_idxs[i];
    lines[i] = points2[idx].homogeneous().cross(H * points1[idx].homogeneous());
    off_points1[i] = points1[idx];
    off_points2[i] = points2[idx];
  }

  InlierSupportMeasurer support_measurer;
  std::vector<double> residuals;

  // Plane correspondences are consistent with F = [e2]_x H for ANY epipole e2
  // (x2^T [e2]_x H x1 ~ x2^T [e2]_x x2 = 0 when H x1 ~ x2), so among the
  // F = [e2]_x H candidates only the off-plane correspondences discriminate.
  // Scoring epipole candidates over just the off-plane subset ranks them
  // correctly at a fraction of the cost of scoring all correspondences.
  InlierSupportMeasurer::Support best_off_support;
  std::optional<Eigen::Matrix3d> best_F;
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

    ComputeSquaredSampsonError(off_points1, off_points2, F, &residuals);
    const auto support =
        support_measurer.Evaluate(residuals, sampson_max_residual);
    if (support_measurer.IsLeftBetter(support, best_off_support)) {
      best_off_support = support;
      best_F = F;
    }
  }

  if (!best_F.has_value()) {
    return std::nullopt;
  }

  // Seed the full-data support of the best epipole candidate, against which the
  // mixed-sample refinement below is compared.
  ComputeSquaredSampsonError(points1, points2, *best_F, &residuals);
  InlierSupportMeasurer::Support best_support =
      support_measurer.Evaluate(residuals, sampson_max_residual);

  // Refine the recovered fundamental matrix by fitting the 8-point algorithm to
  // mixed samples of plane and off-plane inliers, so the epipole is constrained
  // by several off-plane points rather than just the two used above. Fitting on
  // a balanced sample avoids the plane bias that a full-inlier fit would incur.
  std::vector<size_t> plane_inliers;
  std::vector<size_t> off_plane_inliers;
  for (size_t i = 0; i < points1.size(); ++i) {
    if (plane_transfer_errors[i] <= plane_max_residual) {
      plane_inliers.push_back(i);
    }
  }
  ComputeSquaredSampsonError(points1, points2, *best_F, &residuals);
  for (size_t i = 0; i < points1.size(); ++i) {
    if (residuals[i] <= sampson_max_residual &&
        plane_transfer_errors[i] > off_plane_min_residual) {
      off_plane_inliers.push_back(i);
    }
  }

  constexpr size_t kNumPlaneSample = 6;
  constexpr size_t kMaxNumOffPlaneSample = 4;
  if (plane_inliers.size() >= kNumPlaneSample &&
      off_plane_inliers.size() >= 2) {
    const size_t num_off_sample =
        std::min(kMaxNumOffPlaneSample, off_plane_inliers.size());
    FundamentalMatrixEightPointEstimator eight_point;
    std::vector<Eigen::Vector2d> sample_points1;
    std::vector<Eigen::Vector2d> sample_points2;
    std::vector<Eigen::Matrix3d> refined_models;
    constexpr int kNumRefineTrials = 15;
    for (int trial = 0; trial < kNumRefineTrials; ++trial) {
      SampleDistinct(kNumPlaneSample, &plane_inliers);
      SampleDistinct(num_off_sample, &off_plane_inliers);
      sample_points1.clear();
      sample_points2.clear();
      for (size_t i = 0; i < kNumPlaneSample; ++i) {
        sample_points1.push_back(points1[plane_inliers[i]]);
        sample_points2.push_back(points2[plane_inliers[i]]);
      }
      for (size_t i = 0; i < num_off_sample; ++i) {
        sample_points1.push_back(points1[off_plane_inliers[i]]);
        sample_points2.push_back(points2[off_plane_inliers[i]]);
      }

      refined_models.clear();
      eight_point.Estimate(sample_points1, sample_points2, &refined_models);
      if (refined_models.empty()) {
        continue;
      }
      ComputeSquaredSampsonError(
          points1, points2, refined_models[0], &residuals);
      const auto support =
          support_measurer.Evaluate(residuals, sampson_max_residual);
      if (support_measurer.IsLeftBetter(support, best_support)) {
        best_support = support;
        best_F = refined_models[0];
      }
    }
  }

  return best_F;
}

FundamentalMatrixDegensacEstimator::FundamentalMatrixDegensacEstimator(
    const std::vector<Eigen::Vector2d>* points1,
    const std::vector<Eigen::Vector2d>* points2,
    const double sampson_max_residual,
    const double plane_max_residual,
    const double off_plane_min_residual,
    const double min_sample_h_inlier_ratio,
    const int max_plane_parallax_trials)
    : points1_(THROW_CHECK_NOTNULL(points1)),
      points2_(THROW_CHECK_NOTNULL(points2)),
      sampson_max_residual_(sampson_max_residual),
      plane_max_residual_(plane_max_residual),
      off_plane_min_residual_(off_plane_min_residual),
      min_sample_h_inlier_ratio_(min_sample_h_inlier_ratio),
      max_plane_parallax_trials_(max_plane_parallax_trials) {}

void FundamentalMatrixDegensacEstimator::Estimate(
    const std::vector<X_t>& sample_points1,
    const std::vector<Y_t>& sample_points2,
    std::vector<M_t>* models) const {
  THROW_CHECK(models != nullptr);
  THROW_CHECK_GE(sample_points1.size(), kMinNumSamples);

  // Fit the fundamental matrix from the sample: the 7-point solver for a
  // minimal sample (yielding up to three roots), the 8-point solver otherwise
  // (e.g. the local-optimization inlier set).
  std::vector<M_t> sample_models;
  if (sample_points1.size() == static_cast<size_t>(kMinNumSamples)) {
    FundamentalMatrixSevenPointEstimator::Estimate(
        sample_points1, sample_points2, &sample_models);
  } else {
    FundamentalMatrixEightPointEstimator::Estimate(
        sample_points1, sample_points2, &sample_models);
  }

  models->clear();
  models->reserve(sample_models.size());
  for (const auto& sample_model : sample_models) {
    const std::optional<Eigen::Matrix3d> plane_H =
        DetectSampleHDegeneracy(sample_model,
                                sample_points1,
                                sample_points2,
                                plane_max_residual_,
                                min_sample_h_inlier_ratio_);
    if (!plane_H.has_value()) {
      // Non-degenerate sample: keep the fitted hypothesis as-is.
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
                                        plane_max_residual_,
                                        off_plane_min_residual_,
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

RANSAC<FundamentalMatrixDegensacEstimator>::Report
EstimateFundamentalMatrixDegensac(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const FundamentalMatrixDegensacOptions& options) {
  const double sampson_max_residual =
      options.ransac.max_error * options.ransac.max_error;
  // A correspondence counts as on the dominant plane within a looser margin
  // than the inlier threshold; only points well off the plane serve as
  // parallax.
  const double plane_max_error =
      options.plane_max_error > 0 ? options.plane_max_error
                                  : std::sqrt(3.0) * options.ransac.max_error;
  const double off_plane_min_error = options.off_plane_min_error > 0
                                         ? options.off_plane_min_error
                                         : 10.0 * options.ransac.max_error;
  const double plane_max_residual = plane_max_error * plane_max_error;
  const double off_plane_min_residual =
      off_plane_min_error * off_plane_min_error;

  // The DEGENSAC estimator is used as BOTH the hypothesis and the
  // local-optimization solver, so the local optimization also applies the
  // degeneracy handling instead of re-fitting a plane-corrupted model.
  FundamentalMatrixDegensacEstimator estimator(
      &points1,
      &points2,
      sampson_max_residual,
      plane_max_residual,
      off_plane_min_residual,
      options.min_sample_h_inlier_ratio,
      options.max_plane_parallax_trials);
  LORANSAC<FundamentalMatrixDegensacEstimator,
           FundamentalMatrixDegensacEstimator>
      ransac(options.ransac, estimator, estimator);
  return ransac.Estimate(points1, points2);
}

}  // namespace colmap
