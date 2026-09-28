// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_alignment.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

struct GRNPObservation {
  Rigid3d cam_from_rig;
  // Bearing in the camera frame, bundled with its Jacobian d(ray) / d(pixel) so
  // Residuals can score in pixel units with the tangent Sampson error. Only the
  // residual uses the Jacobian; the solvers read the ray alone.
  CamRayWithJac ray_with_jac_in_cam;
};

// Minimal generalized relative pose estimator for the 5+1-point case, based on
// poselib's gen_relpose_5p1pt: five correspondences from one camera pair plus
// one from a different pair. Faster and better conditioned than the general
// 6pt solver; requires the 6th correspondence to come from a different camera
// pair so that the rig translation scale is observable.
class GR5P1PEstimator {
 public:
  using X_t = GRNPObservation;
  using Y_t = GRNPObservation;
  // The estimated rig2_from_rig1 relative pose between the generalized cameras.
  using M_t = Rigid3d;

  static const int kMinNumSamples = 6;

  // Estimate rig2_from_rig1 from six 2D-2D correspondences. Returns no models
  // if the sample does not contain five correspondences sharing one camera
  // pair plus a sixth from a different pair, or if the solver fails.
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* rigs2_from_rigs1);

  // Calculate the squared tangent Sampson error (in pixels) between
  // corresponding points.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& rig2_from_rig1,
                        std::vector<double>* residuals);
};

// Minimal generalized relative pose estimator based on poselib.
class GR6PEstimator {
 public:
  using X_t = GRNPObservation;
  using Y_t = GRNPObservation;
  // The estimated rig2_from_rig1 relative pose between the generalized cameras.
  using M_t = Rigid3d;

  // The minimum number of samples needed to estimate a model. Note that in
  // theory the minimum required number of samples is 6 but Laurent Kneip showed
  // in his paper that using 8 samples is more stable.
  static const int kMinNumSamples = 6;

  // Estimate the most probable solution of the GR6P problem from a set of
  // six 2D-2D point correspondences. Uses the faster gen_relpose_5p1pt solver
  // as a fast path whenever five correspondences share one camera pair (the
  // common rig case) and falls back to the general gen_relpose_6pt solver
  // otherwise.
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* rigs2_from_rigs1);

  // Calculate the squared tangent Sampson error (in pixels) between
  // corresponding points.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& rig2_from_rig1,
                        std::vector<double>* residuals);
};

// Solver for the Generalized Relative Pose problem using a minimal of 8 2D-2D
// correspondences. This implementation is based on:
//
//    "Efficient Computation of Relative Pose for Multi-Camera Systems",
//    Kneip and Li. CVPR 2014.
//
// Note that the solution to this problem is degenerate in the case of pure
// translation and when all correspondences are observed from the same cameras.
//
// The implementation is a modified and improved version of Kneip's original
// implementation in OpenGV licensed under the BSD license.
class GR8PEstimator {
 public:
  using X_t = GRNPObservation;
  using Y_t = GRNPObservation;
  // The estimated rig2_from_rig1 relative pose between the generalized cameras.
  using M_t = Rigid3d;

  // The minimum number of samples needed to estimate a model. Note that in
  // theory the minimum required number of samples is 6 but Laurent Kneip showed
  // in his paper that using 8 samples is more stable.
  static const int kMinNumSamples = 8;

  // Estimate the most probable solution of the GR8P problem from a set of
  // eight 2D-2D point correspondences.
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* rigs2_from_rigs1);

  // Calculate the squared tangent Sampson error (in pixels) between
  // corresponding points.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& rig2_from_rig1,
                        std::vector<double>* residuals);
};

}  // namespace colmap
