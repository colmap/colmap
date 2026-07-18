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

#include "colmap/estimators/solvers/relpose_one_sided_focal.h"

#include "colmap/estimators/cost_functions/tiny_manifold.h"
#include "colmap/estimators/cost_functions/tiny_sampson_error.h"
#include "colmap/estimators/solvers/poselib_utils.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/optim/tiny_solver.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <cmath>
#include <limits>

#include <Eigen/Geometry>
#include <PoseLib/solvers/relpose_6pt_onesided_focal.h>

namespace colmap {
namespace {

// The 6-DoF manifold of a relative pose plus one unknown focal length:
// rotation on SO(3), translation on the unit sphere, and the log-focal as a 1-D
// Euclidean parameter. The ambient parameter layout matches
// TinyOneSidedFocalEpipolarErrorCostFunctor:
// [qx, qy, qz, qw, tx, ty, tz, log_f1].
using RelativePoseOneSidedFocalManifold =
    ProductManifold<EigenQuaternionManifold,
                    SphereManifold<3>,
                    EuclideanManifold<1>>;

// Mixed epipolar matrix M = E * K1inv with K1inv = diag(1/f1, 1/f1, 1), which
// relates image points of the uncalibrated view to bearing rays of the
// calibrated one: ray2^T M point1 = 0.
Eigen::Matrix3d MixedEpipolarMatrix(const Eigen::Matrix3d& E,
                                    const double focal1) {
  const Eigen::DiagonalMatrix<double, 3> K1_inv(
      1.0 / focal1, 1.0 / focal1, 1.0);
  return E * K1_inv;
}

// Focal-calibrated, normalized camera rays (x / f, y / f, 1) from centered
// image points.
std::vector<Eigen::Vector3d> CalibratedRays(
    const std::vector<Eigen::Vector2d>& points, const double focal) {
  const double inv_f = 1.0 / focal;
  std::vector<Eigen::Vector3d> rays(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    rays[i] = Eigen::Vector3d(points[i].x() * inv_f, points[i].y() * inv_f, 1.0)
                  .normalized();
  }
  return rays;
}

}  // namespace

void RelativePoseOneSidedFocalEstimator::Estimate(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& cam_rays2,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(points1.size(), cam_rays2.size());
  THROW_CHECK_GE(points1.size(), kMinNumSamples);
  THROW_CHECK_NOTNULL(models)->clear();

  // Unlike poselib::relpose_6pt_shared_focal, this solver conditions its own
  // input: it isotropically rescales the uncalibrated points internally and
  // undoes that scaling on the recovered focal, which therefore comes back in
  // the pixel units of `points1`. So pass the centered points through as plain
  // homogeneous coordinates and do not pre-normalize them here.
  std::vector<Eigen::Vector3d> points1_homogeneous(points1.size());
  std::vector<Eigen::Vector3d> normalized_cam_rays2(cam_rays2.size());
  for (size_t i = 0; i < points1.size(); ++i) {
    points1_homogeneous[i] = points1[i].homogeneous();
    // The epipolar constraint is homogeneous in the rays, but normalizing keeps
    // the nullspace computation well conditioned.
    normalized_cam_rays2[i] = cam_rays2[i].normalized();
  }

  // Use the full elimination template (use_elim = false), which solves for the
  // focal directly as an unknown. The default compact template instead recovers
  // it from the fundamental matrix via a semi-calibrated Kruppa step, which is
  // numerically fragile and loses accuracy even on exact points, propagating
  // into E. The larger template costs only marginally more runtime, as both
  // variants end in the same eigendecomposition, which dominates.
  poselib::ImagePairVector image_pairs;
  poselib::relpose_6pt_onesided_focal(points1_homogeneous,
                                      normalized_cam_rays2,
                                      &image_pairs,
                                      /*use_elim=*/false);

  models->reserve(image_pairs.size());
  for (const poselib::ImagePair& image_pair : image_pairs) {
    // The solver may return degenerate solutions, so guard the focal
    // defensively.
    const double focal = image_pair.camera1.focal();
    if (!(focal > 0.0)) {
      continue;
    }
    const Rigid3d cam2_from_cam1 = ConvertPoseLibPoseToRigid3d(image_pair.pose);
    M_t model;
    model.E = EssentialMatrixFromPose(cam2_from_cam1);
    model.focal = focal;
    models->push_back(model);
  }
}

bool RelativePoseOneSidedFocalEstimator::Refine(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& cam_rays2,
    M_t* model) {
  THROW_CHECK_EQ(points1.size(), cam_rays2.size());
  THROW_CHECK_GE(points1.size(), kMinNumSamples);
  THROW_CHECK_NOTNULL(model);

  if (!(model->focal > 0.0)) {
    return false;
  }

  // Decompose the current essential matrix into a relative pose, resolving the
  // four-fold ambiguity by cheirality. The first view's rays follow from the
  // current focal estimate; the second view's are already calibrated.
  const std::vector<Eigen::Vector3d> cam_rays1 =
      CalibratedRays(points1, model->focal);
  Rigid3d cam2_from_cam1;
  std::vector<int> valid_indices;
  PoseFromEssentialMatrix(
      model->E, cam_rays1, cam_rays2, &cam2_from_cam1, &valid_indices);
  if (valid_indices.empty()) {
    // Degenerate configuration: leave the initial model unchanged.
    return false;
  }

  // Nonlinear refinement of the joint 6-DoF (5-DoF pose plus the unknown focal)
  // via ceres::TinySolver (fixed-size, allocation-free, autodiff), applying the
  // one-sided focal relative pose manifold. The functor minimizes the same
  // epipolar-line distance that Residuals() scores, so local optimization
  // improves exactly the quantity LO-RANSAC measures. Plain least squares: the
  // points are assumed to be the inlier set, so robustness comes from the
  // RANSAC inlier selection.
  TinyOneSidedFocalEpipolarErrorCostFunctor functor(points1, cam_rays2);
  TinyOneSidedFocalEpipolarErrorCostFunctor::AutoDiffFunction f(functor);
  using Solver = TinySolver<decltype(f), RelativePoseOneSidedFocalManifold>;
  Solver solver;
  Solver::Options options;
  options.max_num_iterations = 25;

  Eigen::Matrix<double, 8, 1> x;
  x.head<4>() = cam2_from_cam1.rotation().normalized().coeffs();
  x.segment<3>(4) = cam2_from_cam1.translation().normalized();
  x[7] = std::log(model->focal);
  solver.Solve(f, &x, options);

  // Keep the refined estimate only if the solve stayed finite and left a
  // non-degenerate baseline; otherwise fall back to the decomposed pose and
  // seed focal.
  const Eigen::Vector3d translation = x.segment<3>(4);
  if (x.allFinite() && translation.squaredNorm() > 0) {
    cam2_from_cam1 =
        Rigid3d(Eigen::Quaterniond(x.data()).normalized(), translation);
    model->E = EssentialMatrixFromPose(cam2_from_cam1);
    model->focal = std::exp(x[7]);
  }
  return true;
}

void RelativePoseOneSidedFocalEstimator::Residuals(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& cam_rays2,
    const M_t& model,
    std::vector<double>* residuals) {
  THROW_CHECK_EQ(points1.size(), cam_rays2.size());
  THROW_CHECK_NOTNULL(residuals);
  if (!(model.focal > 0.0)) {
    residuals->assign(points1.size(), std::numeric_limits<double>::max());
    return;
  }
  const Eigen::Matrix3d M = MixedEpipolarMatrix(model.E, model.focal);
  residuals->resize(points1.size());
  for (size_t i = 0; i < points1.size(); ++i) {
    const Eigen::Vector3d& cam_ray2 = cam_rays2[i];
    // Epipolar line induced in the first view by the second view's ray,
    // expressed in the first view's pixel coordinates.
    const Eigen::Vector3d line1 = M.transpose() * cam_ray2;
    const double squared_line_norm = line1.head<2>().squaredNorm();
    if (!(squared_line_norm > 0)) {
      // Degenerate line, e.g. from a zero ray left by a failed undistortion.
      (*residuals)[i] = std::numeric_limits<double>::max();
      continue;
    }
    const double num = cam_ray2.dot(M * points1[i].homogeneous());
    (*residuals)[i] = num * num / squared_line_norm;
  }
}

}  // namespace colmap
