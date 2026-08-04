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

#include "colmap/optim/ransac.h"

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
// When the scene contains a dominant plane, a random minimal sample frequently
// contains a majority of coplanar correspondences. The fundamental matrix
// computed from such an H-degenerate sample is compatible with the plane
// homography but its epipole is essentially unconstrained, yet it can still
// accrue high inlier support (all plane points fit it). The recovered epipolar
// geometry is then wrong even though RANSAC reports a confident result.
//
// This is implemented as a solver that detects H-degeneracy of a sample and,
// instead of returning the plane-corrupted model, completes it via
// plane-and-parallax: the dominant plane homography is refit on the
// H-consistent data and the epipole (hence the fundamental matrix) is recovered
// from the off-plane parallax. If no usable off-plane parallax exists, the
// sample is rejected.
//
// The solver handles both minimal (7-point) and non-minimal (>= 8-point)
// samples, so it can be used as BOTH the hypothesis and the local-optimization
// estimator in LO-RANSAC. This is important: a plain non-minimal solver would
// re-fit a plane-corrupted model on the (plane-dominated) inlier set during
// local optimization, undoing the completion.
//
// The estimator needs the full correspondence set (not just the minimal sample)
// for the plane-and-parallax completion, so it holds pointers to the data
// passed to RANSAC.
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
  // @param plane_max_residual       Squared max transfer error for a
  //                                 correspondence to count as lying on the
  //                                 dominant plane (degeneracy detection and
  //                                 homography refit).
  // @param off_plane_min_residual   Squared min transfer error for a
  //                                 correspondence to be used as an off-plane
  //                                 parallax source when recovering the
  //                                 epipole.
  // @param min_sample_h_inlier_ratio Fraction of a sample that must be
  //                                 consistent with a single plane homography
  //                                 for the sample to be H-degenerate.
  // @param max_plane_parallax_trials Off-plane pairs sampled during completion.
  FundamentalMatrixDegensacEstimator(
      const std::vector<Eigen::Vector2d>* points1,
      const std::vector<Eigen::Vector2d>* points2,
      double sampson_max_residual,
      double plane_max_residual,
      double off_plane_min_residual,
      double min_sample_h_inlier_ratio,
      int max_plane_parallax_trials);

  // Estimate fundamental matrix hypotheses from a sample (minimal 7-point or
  // non-minimal >= 8-point). For each hypothesis whose sample is H-degenerate,
  // the plane-corrupted model is replaced by a plane-and-parallax completion,
  // or dropped if no usable off-plane parallax exists.
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
  double plane_max_residual_;
  double off_plane_min_residual_;
  double min_sample_h_inlier_ratio_;
  int max_plane_parallax_trials_;
};

struct FundamentalMatrixDegensacOptions {
  // RANSAC options that control sampling, scoring, and termination. As in
  // RANSAC/LO-RANSAC, `max_error` is a pixel error that is squared internally,
  // since Sampson residuals are squared errors.
  RANSACOptions ransac;

  // Maximum pixel error for a correspondence to count as lying on the dominant
  // plane, used for the degeneracy test and the homography refit. Squared
  // internally. Deliberately looser than the inlier threshold so noisy plane
  // points are still recognized as coplanar. If <= 0, derived as
  // sqrt(3) * ransac.max_error (i.e. 3x in squared-pixel units).
  double plane_max_error = -1;

  // Minimum pixel error for a correspondence to be used as an off-plane
  // parallax source when recovering the epipole. Squared internally.
  // Deliberately much larger than the inlier threshold so only clean,
  // high-parallax points constrain the epipole (near-plane, low-parallax points
  // are ignored). If <= 0, derived as 10 * ransac.max_error (100x squared).
  double off_plane_min_error = -1;

  // Fraction of a sample that must be consistent with a single plane homography
  // for the sample to be considered H-degenerate. The default corresponds to
  // the paper's "at least 5 of 7" criterion for a minimal sample.
  double min_sample_h_inlier_ratio = 5.0 / 7.0;

  // Maximum number of off-plane correspondence pairs sampled during the
  // plane-and-parallax model completion to recover the epipole. Only a couple
  // of off-plane inliers are needed, so a small budget suffices; the completed
  // model is scored and locally optimized by the surrounding RANSAC anyway.
  // The value 25 sits at the knee of a runtime/accuracy sweep: fewer trials
  // erode success on the hardest >=98%-plane scenes (t=5: -3pts) with no
  // runtime win (the inner pair loop is not the bottleneck), while more trials
  // (or dynamic termination) yield no accuracy gain at added cost.
  int max_plane_parallax_trials = 25;
};

// Robustly estimate the fundamental matrix from corresponding image points
// using DEGENSAC inside LO-RANSAC (the DEGENSAC estimator is used as both the
// hypothesis and the local-optimization solver).
RANSAC<FundamentalMatrixDegensacEstimator>::Report
EstimateFundamentalMatrixDegensac(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const FundamentalMatrixDegensacOptions& options);

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
// Returns nullopt on degenerate input (collinear first-image points, or a point
// at the epipole).
std::optional<Eigen::Matrix3d> HomographyFromFundamentalAndPoints(
    const Eigen::Matrix3d& F,
    const Eigen::Vector3d& epipole2,
    const std::array<Eigen::Vector2d, 3>& points1,
    const std::array<Eigen::Vector2d, 3>& points2);

// Test a sample for H-degeneracy, i.e. whether a fraction of at least
// `min_sample_h_inlier_ratio` of the correspondences lie on a common scene
// plane. Plane homographies compatible with F are constructed from triplets of
// the sample (all C(7,3) triplets for a minimal sample, otherwise a fixed
// number of randomly sampled triplets) and the number of sample correspondences
// consistent with each (squared forward transfer error <= h_max_residual) is
// counted. Returns the homography of the triplet with the most consistent
// correspondences if that count reaches the threshold, otherwise nullopt.
std::optional<Eigen::Matrix3d> DetectSampleHDegeneracy(
    const Eigen::Matrix3d& F,
    const std::vector<Eigen::Vector2d>& sample_points1,
    const std::vector<Eigen::Vector2d>& sample_points2,
    double h_max_residual,
    double min_sample_h_inlier_ratio);

// Convenience predicate wrapping DetectSampleHDegeneracy.
bool IsSampleHDegenerate(const Eigen::Matrix3d& F,
                         const std::vector<Eigen::Vector2d>& sample_points1,
                         const std::vector<Eigen::Vector2d>& sample_points2,
                         double h_max_residual,
                         double min_sample_h_inlier_ratio);

// Recover the fundamental matrix from a dominant plane homography and the
// off-plane parallax (plane-and-parallax model completion). The seed homography
// is refit on its plane inliers (squared transfer error <=
// `plane_max_residual`); then correspondences whose squared transfer error
// exceeds `off_plane_min_residual` are treated as clean off-plane parallax
// sources, and the epipole is recovered from a pair of them as
// e2 = (x2_a x H x1_a) x (x2_b x H x1_b), giving F = [e2]_x H. Pairs are
// sampled robustly and the best F is then refined by fitting the 8-point
// algorithm to mixed samples of plane and off-plane inliers (so the epipole is
// constrained by more than two off-plane points). The F with the largest
// squared-Sampson support (threshold `sampson_max_residual`) over all
// correspondences is returned.
//
// @return  The recovered fundamental matrix, or nullopt if there are too few
//          off-plane correspondences.
std::optional<Eigen::Matrix3d> FundamentalFromPlaneAndParallax(
    const Eigen::Matrix3d& seed_H,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    double sampson_max_residual,
    double plane_max_residual,
    double off_plane_min_residual,
    int max_trials);

}  // namespace colmap
