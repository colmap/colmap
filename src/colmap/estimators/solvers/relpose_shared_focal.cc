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

#include "colmap/estimators/solvers/relpose_shared_focal.h"

#include "colmap/estimators/cost_functions/tiny_manifold.h"
#include "colmap/estimators/cost_functions/tiny_sampson_error.h"
#include "colmap/estimators/solvers/poselib_utils.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/optim/tiny_solver.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <cfloat>
#include <cmath>

#include <Eigen/Geometry>
#include <PoseLib/solvers/relpose_6pt_focal.h>

namespace colmap {
namespace {

// The 6-DoF manifold of a relative pose plus a shared focal length: rotation on
// SO(3), translation on the unit sphere, and the log-focal as a 1-D Euclidean
// parameter. The ambient parameter layout matches
// TinyFocalSampsonErrorCostFunctor: [qx, qy, qz, qw, tx, ty, tz, log_f].
using RelativePoseSharedFocalManifold = ProductManifold<EigenQuaternionManifold,
                                                        SphereManifold<3>,
                                                        EuclideanManifold<1>>;

// Pixel-space fundamental matrix F = Kinv * E * Kinv with Kinv = diag(1/f, 1/f,
// 1), from a calibrated essential matrix and the shared focal length.
Eigen::Matrix3d FundamentalFromEssentialSharedFocal(const Eigen::Matrix3d& E,
                                                    const double focal) {
  const double inv_f = 1.0 / focal;
  const Eigen::DiagonalMatrix<double, 3> K_inv(inv_f, inv_f, 1.0);
  return K_inv * E * K_inv;
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

void RelativePoseSharedFocalEstimator::Estimate(const std::vector<X_t>& points1,
                                                const std::vector<Y_t>& points2,
                                                std::vector<M_t>* models) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  THROW_CHECK_GE(points1.size(), kMinNumSamples);
  THROW_CHECK_NOTNULL(models)->clear();

  // The minimal solver is formulated for unit bearing vectors, so raw pixel
  // coordinates make it numerically unstable: after appending the homogeneous 1
  // and normalizing, that third component becomes negligible and the focal
  // length is recovered from it. Isotropically rescale the (already
  // principal-point-centered) points to unit magnitude first; the recovered
  // focal is expressed in the rescaled units and undone below. This mirrors the
  // normalization PoseLib applies before running the solver.
  double scale = 0.0;
  for (size_t i = 0; i < points1.size(); ++i) {
    scale += points1[i].norm() + points2[i].norm();
  }
  scale /= 2.0 * points1.size();
  if (!(scale > 0.0)) {
    return;
  }
  const double inv_scale = 1.0 / scale;

  std::vector<Eigen::Vector3d> rays1(points1.size());
  std::vector<Eigen::Vector3d> rays2(points2.size());
  for (size_t i = 0; i < points1.size(); ++i) {
    rays1[i] = (inv_scale * points1[i]).homogeneous().normalized();
    rays2[i] = (inv_scale * points2[i]).homogeneous().normalized();
  }

  poselib::ImagePairVector image_pairs;
  poselib::relpose_6pt_shared_focal(rays1, rays2, &image_pairs);

  models->reserve(image_pairs.size());
  for (const poselib::ImagePair& image_pair : image_pairs) {
    // Undo the isotropic rescaling on the recovered focal; the pose (and thus
    // the essential matrix) is scale-invariant. The solver may return
    // degenerate solutions, so guard the focal defensively.
    const double focal = image_pair.camera1.focal() * scale;
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

bool RelativePoseSharedFocalEstimator::Refine(const std::vector<X_t>& points1,
                                              const std::vector<Y_t>& points2,
                                              M_t* model) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  THROW_CHECK_GE(points1.size(), kMinNumSamples);
  THROW_CHECK_NOTNULL(model);

  if (!(model->focal > 0.0)) {
    return false;
  }

  // Decompose the current essential matrix into a relative pose, resolving the
  // four-fold ambiguity by cheirality over the focal-calibrated sample rays.
  const std::vector<Eigen::Vector3d> cam_rays1 =
      CalibratedRays(points1, model->focal);
  const std::vector<Eigen::Vector3d> cam_rays2 =
      CalibratedRays(points2, model->focal);
  Rigid3d cam2_from_cam1;
  std::vector<int> valid_indices;
  PoseFromEssentialMatrix(
      model->E, cam_rays1, cam_rays2, &cam2_from_cam1, &valid_indices);
  if (valid_indices.empty()) {
    // Degenerate configuration: leave the initial model unchanged.
    return false;
  }

  // Nonlinear pixel-space Sampson refinement of the joint 6-DoF (5-DoF pose
  // plus shared focal) via ceres::TinySolver (fixed-size, allocation-free,
  // autodiff), applying the shared-focal relative pose manifold. Plain least
  // squares: the points are assumed to be the inlier set, so robustness comes
  // from the RANSAC inlier selection.
  TinyFocalSampsonErrorCostFunctor functor(points1, points2);
  TinyFocalSampsonErrorCostFunctor::AutoDiffFunction f(functor);
  using Solver = TinySolver<decltype(f), RelativePoseSharedFocalManifold>;
  Solver solver;
  Solver::Options options;
  options.max_num_iterations = 25;

  Eigen::Matrix<double, 8, 1> x;
  x.head<4>() = cam2_from_cam1.rotation().normalized().coeffs();
  x.segment<3>(4) = cam2_from_cam1.translation().normalized();
  x[7] = std::log(model->focal);
  solver.Solve(f, &x, options);

  // Keep the refined estimate only if the solve stayed finite; otherwise fall
  // back to the decomposed pose and seed focal.
  if (x.allFinite()) {
    cam2_from_cam1 =
        Rigid3d(Eigen::Quaterniond(x.data()).normalized(), x.segment<3>(4));
    model->E = EssentialMatrixFromPose(cam2_from_cam1);
    model->focal = std::exp(x[7]);
  }
  return true;
}

void RelativePoseSharedFocalEstimator::Residuals(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    const M_t& model,
    std::vector<double>* residuals) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  THROW_CHECK_NOTNULL(residuals);
  if (!(model.focal > 0.0)) {
    residuals->assign(points1.size(), DBL_MAX);
    return;
  }
  const Eigen::Matrix3d F =
      FundamentalFromEssentialSharedFocal(model.E, model.focal);
  ComputeSquaredSampsonError(points1, points2, F, residuals);
}

}  // namespace colmap
