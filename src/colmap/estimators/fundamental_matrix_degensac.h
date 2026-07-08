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

#include "colmap/estimators/solvers/fundamental_matrix.h"
#include "colmap/optim/random_sampler.h"
#include "colmap/optim/ransac.h"
#include "colmap/optim/support_measurement.h"
#include "colmap/util/eigen_alignment.h"

#include <array>
#include <optional>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// DEGENSAC-based fundamental matrix estimation, robust to a dominant scene
// plane. Based on the following paper:
//
//    Ondrej Chum, Tomas Werner, Jiri Matas, "Two-View Geometry Estimation
//    Unaffected by a Dominant Plane", CVPR 2005.
//    https://cmp.felk.cvut.cz/~werner/papers/chum-degen-cvpr05.pdf
//
// When the scene contains a dominant plane, a random 7-point minimal sample
// frequently contains five or more coplanar correspondences. The fundamental
// matrix computed from such an H-degenerate sample is compatible with the plane
// homography but its epipole is essentially unconstrained, yet it can still
// accrue high inlier support (all plane points fit it). The recovered epipolar
// geometry is then wrong even though RANSAC reports a confident result.
//
// This detects H-degenerate minimal samples during RANSAC and, instead of
// accepting the plane-corrupted model, completes it using plane-and-parallax:
// the dominant plane homography is refit on the H-consistent data and the
// epipole (hence the fundamental matrix) is recovered from the off-plane
// parallax. The completed model explains both the plane and the off-plane
// points, so it wins on support. If no usable off-plane parallax exists (i.e.
// the configuration is genuinely degenerate), the sample is rejected.
//
// The DEGENSAC logic is implemented as a minimal solver that is injected into
// the existing (LO-)RANSAC machinery. Because the minimal solver receives the
// 7-point sample but the plane-and-parallax completion also needs the full
// correspondence set, the estimator holds pointers to the full data passed to
// RANSAC. It is a drop-in replacement for FundamentalMatrixSevenPointEstimator
// as the hypothesis solver in LORANSAC.
class FundamentalMatrixDegensacEstimator {
 public:
  using X_t = Eigen::Vector2d;
  using Y_t = Eigen::Vector2d;
  using M_t = Eigen::Matrix3d;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 7;

  // @param points1                  Full first-image correspondences (must
  //                                 outlive the estimator and be the same data
  //                                 passed to RANSAC).
  // @param points2                  Full second-image correspondences.
  // @param sampson_max_residual     Squared max Sampson error used when scoring
  //                                 completion candidates (typically the RANSAC
  //                                 max_error squared).
  // @param h_max_residual           Squared max transfer error for
  //                                 H-consistency.
  // @param min_sample_h_inliers     Coplanar-count threshold for degeneracy.
  // @param max_plane_parallax_trials Off-plane pairs sampled during completion.
  FundamentalMatrixDegensacEstimator(
      const std::vector<Eigen::Vector2d>* points1,
      const std::vector<Eigen::Vector2d>* points2,
      double sampson_max_residual,
      double h_max_residual,
      int min_sample_h_inliers,
      int max_plane_parallax_trials);

  // Estimate fundamental matrix hypotheses from a 7-point minimal sample. For
  // each hypothesis whose sample is H-degenerate, the plane-corrupted model is
  // replaced by a plane-and-parallax completion, or dropped if no usable
  // off-plane parallax exists.
  void Estimate(const std::vector<X_t>& sample_points1,
                const std::vector<Y_t>& sample_points2,
                std::vector<M_t>* models) const;

  // Squared Sampson error residuals over the given correspondences.
  void Residuals(const std::vector<X_t>& points1,
                 const std::vector<Y_t>& points2,
                 const M_t& F,
                 std::vector<double>* residuals) const;

 private:
  const std::vector<Eigen::Vector2d>* points1_;
  const std::vector<Eigen::Vector2d>* points2_;
  double sampson_max_residual_;
  double h_max_residual_;
  int min_sample_h_inliers_;
  int max_plane_parallax_trials_;
};

// Convenience driver that runs the DEGENSAC estimator inside LO-RANSAC (with
// the 8-point solver as the local optimizer), mirroring how the plain
// fundamental matrix is estimated. Provided so callers get a stable Report type
// and options.
class FundamentalMatrixDegensac {
 public:
  // Report type identical to the one produced by the plain LO-RANSAC estimator
  // for the fundamental matrix, so it is a drop-in replacement.
  using Report = RANSAC<FundamentalMatrixSevenPointEstimator,
                        InlierSupportMeasurer,
                        RandomSampler>::Report;

  struct Options {
    // RANSAC options that control sampling, scoring, and termination. As in
    // RANSAC/LO-RANSAC, `max_error` is a pixel error that is squared
    // internally, since Sampson residuals are squared errors.
    RANSACOptions ransac;

    // Maximum pixel error for a sample correspondence to be considered
    // consistent with a candidate plane homography during the sample
    // degeneracy test. Squared internally. If <= 0, falls back to
    // `ransac.max_error`.
    double h_consistency_max_error = -1;

    // Minimum number of the 7 minimal-sample correspondences that must be
    // consistent with a single plane homography for the sample to be considered
    // H-degenerate (and thus completed via plane-and-parallax).
    int min_sample_h_inliers = 5;

    // Maximum number of off-plane correspondence pairs sampled during the
    // plane-and-parallax model completion to recover the epipole. Only a couple
    // of off-plane inliers are needed, so a small budget suffices; the
    // completed model is scored and locally optimized by the surrounding RANSAC
    // anyway.
    int max_plane_parallax_trials = 25;
  };

  explicit FundamentalMatrixDegensac(Options options);

  // Robustly estimate the fundamental matrix mapping points in image 1 to
  // epipolar lines in image 2 from corresponding pixel observations.
  //
  // @param points1  First set of corresponding image points.
  // @param points2  Second set of corresponding image points.
  //
  // @return         The report with the results of the estimation.
  Report Estimate(const std::vector<Eigen::Vector2d>& points1,
                  const std::vector<Eigen::Vector2d>& points2);

 private:
  const Options options_;
};

// Compute the epipole in the second image, i.e. the left null vector e2 of the
// fundamental matrix with F^T e2 = 0. Returned in homogeneous coordinates and
// normalized to unit length.
Eigen::Vector3d EpipoleFromFundamentalMatrix(const Eigen::Matrix3d& F);

// Compute the plane-induced homography compatible with the epipolar geometry F
// from three point correspondences, following Hartley & Zisserman, "Multiple
// View Geometry", Result 13.6:
//
//    H = [e2]_x F - e2 (M^{-1} b)^T
//
// where the rows of M are the homogeneous first-image points x1_i, and
//
//    b_i = (x2_i x ([e2]_x F x1_i)) . (x2_i x e2) / ||x2_i x e2||^2.
//
// @param F         3x3 fundamental matrix (image 1 to image 2).
// @param epipole2  Left epipole e2 (see EpipoleFromFundamentalMatrix).
// @param points1   Three first-image points.
// @param points2   Three corresponding second-image points.
//
// @return          The homography, or nullopt if the configuration is
//                  degenerate (collinear first-image points, or a point at the
//                  epipole).
std::optional<Eigen::Matrix3d> HomographyFromFundamentalAndPoints(
    const Eigen::Matrix3d& F,
    const Eigen::Vector3d& epipole2,
    const std::array<Eigen::Vector2d, 3>& points1,
    const std::array<Eigen::Vector2d, 3>& points2);

// Test a 7-point minimal sample for H-degeneracy, i.e. whether five or more of
// the seven correspondences lie on a common scene plane. For each triplet of
// the sample, the plane homography compatible with F is constructed and the
// number of sample correspondences consistent with it (squared forward transfer
// error <= h_max_residual) is counted. Returns the homography of the triplet
// with the most consistent correspondences if that count reaches
// `min_sample_h_inliers`, otherwise nullopt.
//
// @param F                    3x3 fundamental matrix estimated from the sample.
// @param sample_points1       Seven first-image sample points.
// @param sample_points2       Seven second-image sample points.
// @param h_max_residual       Squared max transfer error for H-consistency.
// @param min_sample_h_inliers Threshold on the coplanar count (typically 5).
std::optional<Eigen::Matrix3d> DetectSampleHDegeneracy(
    const Eigen::Matrix3d& F,
    const std::vector<Eigen::Vector2d>& sample_points1,
    const std::vector<Eigen::Vector2d>& sample_points2,
    double h_max_residual,
    int min_sample_h_inliers);

// Convenience predicate wrapping DetectSampleHDegeneracy.
bool IsSampleHDegenerate(const Eigen::Matrix3d& F,
                         const std::vector<Eigen::Vector2d>& sample_points1,
                         const std::vector<Eigen::Vector2d>& sample_points2,
                         double h_max_residual,
                         int min_sample_h_inliers);

// Recover the fundamental matrix from a dominant plane homography and the
// off-plane parallax (plane-and-parallax model completion). Correspondences
// whose squared forward transfer error under H exceeds `h_max_residual` are
// treated as off-plane candidates; the epipole is recovered from a pair of them
// as e2 = (x2_a x H x1_a) x (x2_b x H x1_b), giving F = [e2]_x H. Pairs are
// sampled robustly and the F with the largest squared-Sampson support
// (threshold `sampson_max_residual`) over all correspondences is returned.
//
// @param H                    Dominant plane homography (image 1 to image 2).
// @param points1              First-image correspondences (all data).
// @param points2              Second-image correspondences (all data).
// @param sampson_max_residual Squared max Sampson error for scoring support.
// @param h_max_residual       Squared max transfer error to classify off-plane.
// @param max_trials           Maximum number of off-plane pairs to sample.
//
// @return                     The recovered fundamental matrix, or nullopt if
//                             there are too few off-plane correspondences.
std::optional<Eigen::Matrix3d> FundamentalFromPlaneAndParallax(
    const Eigen::Matrix3d& H,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    double sampson_max_residual,
    double h_max_residual,
    int max_trials);

}  // namespace colmap
