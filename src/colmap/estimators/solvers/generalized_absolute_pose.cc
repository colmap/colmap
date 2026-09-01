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

#include "colmap/estimators/solvers/generalized_absolute_pose.h"

#include "colmap/estimators/cost_functions/tiny_manifold.h"
#include "colmap/estimators/solvers/poselib_utils.h"
#include "colmap/optim/tiny_solver.h"
#include "colmap/util/logging.h"

#include <cmath>

#include <Eigen/Geometry>
#include <PoseLib/solvers/gp3p.h>
#include <PoseLib/solvers/gp4ps.h>
#include <PoseLib/solvers/p3p.h>
#include <ceres/tiny_solver_autodiff_function.h>

namespace colmap {
namespace {

// The 7-DoF manifold of a scaled rig_from_world transform: rotation on SO(3),
// with the translation and log-scale as Euclidean parameters. The ambient
// parameter layout matches TinyScaledRigReprojCostFunctor:
// [qx, qy, qz, qw, tx, ty, tz, log_s].
using ScaledRigFromWorldManifold =
    ProductManifold<EigenQuaternionManifold, EuclideanManifold<4>>;

// Normalized-plane reprojection cost functor for fixed-size
// (colmap::TinySolver) refinement of a scaled rig_from_world transform over
// all given 2D-3D correspondences.
class TinyScaledRigReprojCostFunctor {
 public:
  using Scalar = double;
  static constexpr int NUM_RESIDUALS = Eigen::Dynamic;
  static constexpr int NUM_PARAMETERS = 8;

  // ceres::TinySolver-compatible autodiff wrapper for this functor.
  using AutoDiffFunction =
      ceres::TinySolverAutoDiffFunction<TinyScaledRigReprojCostFunctor,
                                        NUM_RESIDUALS,
                                        NUM_PARAMETERS>;

  TinyScaledRigReprojCostFunctor(
      const std::vector<GP4PSEstimator::X_t>& points2D,
      const std::vector<Eigen::Vector3d>& points3D)
      : points2D_(points2D), points3D_(points3D) {}

  int NumResiduals() const { return 2 * static_cast<int>(points2D_.size()); }

  template <typename T>
  bool operator()(const T* const params, T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> rotation(params);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(params + 4);
    const T scale = ceres::exp(params[7]);
    for (size_t i = 0; i < points2D_.size(); ++i) {
      const Eigen::Matrix<T, 3, 1> point3D_in_rig =
          scale * (rotation * points3D_[i].cast<T>()) + translation;
      const Eigen::Matrix<T, 3, 1> point3D_in_cam =
          points2D_[i].cam_from_rig.cast<T>() * point3D_in_rig.homogeneous();
      // Reject the evaluation if a point does not project; a zero residual
      // would otherwise make an invalid pose appear as a perfect fit. The
      // solver treats this as a failed trial step and shrinks the trust
      // region, or reports failure if the initial model is invalid.
      if (point3D_in_cam.z() <= T(std::numeric_limits<double>::epsilon())) {
        return false;
      }
      const Eigen::Matrix<T, 2, 1> diff =
          points2D_[i].ray_in_cam.hnormalized().cast<T>() -
          point3D_in_cam.hnormalized();
      residuals[2 * i] = diff.x();
      residuals[2 * i + 1] = diff.y();
    }
    return true;
  }

 private:
  const std::vector<GP4PSEstimator::X_t>& points2D_;
  const std::vector<Eigen::Vector3d>& points3D_;
};

void ComputeRaysAndOriginsInRig(const std::vector<GP3PEstimator::X_t>& points2D,
                                std::vector<Eigen::Vector3d>* rays_in_rig,
                                std::vector<Eigen::Vector3d>* origins_in_rig) {
  const size_t num_points = points2D.size();
  rays_in_rig->resize(num_points);
  origins_in_rig->resize(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    const Eigen::Matrix3d rig_from_cam_rotation =
        points2D[i].cam_from_rig.leftCols<3>().transpose();
    (*rays_in_rig)[i] =
        (rig_from_cam_rotation * points2D[i].ray_in_cam).normalized();
    (*origins_in_rig)[i] =
        rig_from_cam_rotation * -points2D[i].cam_from_rig.col(3);
  }
}

void ComputeRayResiduals(const std::vector<GP3PEstimator::X_t>& points2D,
                         const std::vector<Eigen::Vector3d>& points3D,
                         const Eigen::Matrix3x4d& rig_from_world_matrix,
                         GP3PEstimator::ResidualType residual_type,
                         std::vector<double>* residuals) {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  residuals->resize(points2D.size(), 0);

  for (size_t i = 0; i < points2D.size(); ++i) {
    const Eigen::Vector3d point3D_in_cam =
        points2D[i].cam_from_rig *
        (rig_from_world_matrix * points3D[i].homogeneous()).homogeneous();

    if (point3D_in_cam.z() > std::numeric_limits<double>::epsilon()) {
      const Eigen::Vector3d& ray = points2D[i].ray_in_cam;

      if (residual_type == GP3PEstimator::ResidualType::CosineDistance) {
        const double cosine_dist =
            1 - point3D_in_cam.normalized().dot(ray.normalized());
        (*residuals)[i] = cosine_dist * cosine_dist;
      } else if (residual_type ==
                 GP3PEstimator::ResidualType::ReprojectionError) {
        const Eigen::Vector2d diff =
            ray.hnormalized() - point3D_in_cam.hnormalized();
        (*residuals)[i] = diff.squaredNorm();
      } else {
        LOG(FATAL_THROW) << "Invalid residual type";
      }
    } else {
      (*residuals)[i] = std::numeric_limits<double>::max();
    }
  }
}

}  // namespace

GP3PEstimator::GP3PEstimator(ResidualType residual_type)
    : residual_type_(residual_type) {}

void GP3PEstimator::Estimate(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             std::vector<M_t>* rigs_from_world) {
  THROW_CHECK_EQ(points2D.size(), 3);
  THROW_CHECK_EQ(points3D.size(), 3);
  THROW_CHECK_NOTNULL(rigs_from_world);

  rigs_from_world->clear();

  std::vector<Eigen::Vector3d> rays_in_rig;
  std::vector<Eigen::Vector3d> origins_in_rig;
  ComputeRaysAndOriginsInRig(points2D, &rays_in_rig, &origins_in_rig);

  std::vector<poselib::CameraPose> poses;
  if (origins_in_rig[0].isApprox(origins_in_rig[1], 1e-6) &&
      origins_in_rig[0].isApprox(origins_in_rig[2], 1e-6)) {
    poselib::p3p(rays_in_rig, points3D, &poses);
    for (poselib::CameraPose& pose : poses) {
      pose.t += origins_in_rig[0];
    }
  } else {
    poselib::gp3p(origins_in_rig, rays_in_rig, points3D, &poses);
  }

  rigs_from_world->reserve(poses.size());
  for (const poselib::CameraPose& pose : poses) {
    rigs_from_world->emplace_back(ConvertPoseLibPoseToRigid3d(pose));
  }
}

void GP3PEstimator::Residuals(const std::vector<X_t>& points2D,
                              const std::vector<Y_t>& points3D,
                              const M_t& rig_from_world,
                              std::vector<double>* residuals) const {
  ComputeRayResiduals(
      points2D, points3D, rig_from_world.ToMatrix(), residual_type_, residuals);
}

GP4PSEstimator::GP4PSEstimator(ResidualType residual_type)
    : residual_type_(residual_type) {}

void GP4PSEstimator::Estimate(const std::vector<X_t>& points2D,
                              const std::vector<Y_t>& points3D,
                              std::vector<M_t>* rigs_from_world) {
  THROW_CHECK_EQ(points2D.size(), 4);
  THROW_CHECK_EQ(points3D.size(), 4);
  THROW_CHECK_NOTNULL(rigs_from_world);

  rigs_from_world->clear();

  std::vector<Eigen::Vector3d> rays_in_rig;
  std::vector<Eigen::Vector3d> origins_in_rig;
  ComputeRaysAndOriginsInRig(points2D, &rays_in_rig, &origins_in_rig);

  // The scale is unobservable from a single projection center. Also reject
  // panoramic samples of a non-panoramic rig, which would otherwise produce
  // spurious models with arbitrary scale.
  if (origins_in_rig[0].isApprox(origins_in_rig[1], 1e-6) &&
      origins_in_rig[0].isApprox(origins_in_rig[2], 1e-6) &&
      origins_in_rig[0].isApprox(origins_in_rig[3], 1e-6)) {
    return;
  }

  // PoseLib solves scale * p + lambda * x = R * X + t with p, x the camera
  // centers and rays in the rig frame and X in the world frame, i.e., (R, t)
  // maps world points into a rig frame whose geometry is scaled by scale.
  std::vector<poselib::CameraPose> poses;
  std::vector<double> scales;
  poselib::gp4ps(origins_in_rig,
                 rays_in_rig,
                 points3D,
                 &poses,
                 &scales,
                 /*filter_solutions=*/false);

  rigs_from_world->reserve(poses.size());
  for (size_t i = 0; i < poses.size(); ++i) {
    const double scale = scales[i];
    const Rigid3d scaled_rig_from_world = ConvertPoseLibPoseToRigid3d(poses[i]);
    if (scale < std::numeric_limits<double>::epsilon() ||
        !std::isfinite(scale) || !scaled_rig_from_world.params.allFinite()) {
      continue;
    }
    // Renormalize to the unscaled rig frame:
    //   p + (lambda / scale) * x = (R / scale) * X + t / scale.
    rigs_from_world->emplace_back(
        1 / scale,
        scaled_rig_from_world.rotation(),
        Eigen::Vector3d(scaled_rig_from_world.translation() / scale));
  }
}

bool GP4PSEstimator::Refine(const std::vector<X_t>& points2D,
                            const std::vector<Y_t>& points3D,
                            M_t* rig_from_world) {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_GE(points2D.size(), kMinNumSamples);
  THROW_CHECK_NOTNULL(rig_from_world);

  if (!(rig_from_world->scale() > 0)) {
    return false;
  }

  TinyScaledRigReprojCostFunctor functor(points2D, points3D);
  TinyScaledRigReprojCostFunctor::AutoDiffFunction f(functor);
  using Solver = TinySolver<decltype(f), ScaledRigFromWorldManifold>;
  Solver solver;
  Solver::Options options;
  options.max_num_iterations = 25;

  Eigen::Matrix<double, 8, 1> x;
  x.head<4>() = rig_from_world->rotation().normalized().coeffs();
  x.segment<3>(4) = rig_from_world->translation();
  x[7] = std::log(rig_from_world->scale());
  const auto& summary = solver.Solve(f, &x, options);

  // Reject models for which the cost cannot be evaluated, e.g., with points
  // behind the cameras at the initial estimate.
  if (summary.status == Solver::COST_FUNCTION_FAILED) {
    return false;
  }

  // Keep the refined estimate only if the solve stayed finite; otherwise fall
  // back to the initial model.
  if (x.allFinite()) {
    *rig_from_world = Sim3d(std::exp(x[7]),
                            Eigen::Quaterniond(x.data()).normalized(),
                            x.segment<3>(4));
  }
  return true;
}

void GP4PSEstimator::Residuals(const std::vector<X_t>& points2D,
                               const std::vector<Y_t>& points3D,
                               const M_t& rig_from_world,
                               std::vector<double>* residuals) const {
  ComputeRayResiduals(
      points2D, points3D, rig_from_world.ToMatrix(), residual_type_, residuals);
}

}  // namespace colmap
