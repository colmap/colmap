// SPDX-License-Identifier: BSD-3-Clause

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

// The manifold of a rig_from_world transform: rotation on SO(3), with the
// translation and (for the scaled variant) the log-scale as Euclidean
// parameters. The ambient parameter layout matches TinyRigCostFunctor:
// [qx, qy, qz, qw, tx, ty, tz] with an appended log_s if scaled.
template <bool kScaled>
using RigFromWorldManifold =
    ProductManifold<EigenQuaternionManifold,
                    EuclideanManifold<kScaled ? 4 : 3>>;

// Cost functor for fixed-size (colmap::TinySolver) refinement of a
// rig_from_world transform (rigid or scaled, selected by kScaled) over all
// given 2D-3D correspondences, minimizing either the normalized-plane
// reprojection error (two residuals per observation) or, for cosine-distance
// scoring, the sine of the angle between the observed and projected rays (as
// a 3-vector cross product per observation). The sine shares the cosine
// distance's minimizer but, unlike 1 - cos, has a non-vanishing Jacobian at
// zero, which the Levenberg-Marquardt iterations require to converge.
//
// Observations that do not project in front of their camera contribute a zero
// residual, as in the other reprojection cost functors. Cheirality is instead
// enforced by the estimator's Residuals when the refined model is scored.
template <bool kScaled>
class TinyRigCostFunctor {
 public:
  using Scalar = double;
  static constexpr int NUM_RESIDUALS = Eigen::Dynamic;
  static constexpr int NUM_PARAMETERS = kScaled ? 8 : 7;

  // ceres::TinySolver-compatible autodiff wrapper for this functor.
  using AutoDiffFunction = ceres::TinySolverAutoDiffFunction<TinyRigCostFunctor,
                                                             NUM_RESIDUALS,
                                                             NUM_PARAMETERS>;

  TinyRigCostFunctor(const std::vector<GP3PEstimator::X_t>& points2D,
                     const std::vector<Eigen::Vector3d>& points3D,
                     GP3PEstimator::ResidualType residual_type)
      : points2D_(points2D),
        points3D_(points3D),
        residual_type_(residual_type) {}

  int NumResiduals() const {
    if (residual_type_ == GP3PEstimator::ResidualType::ReprojectionError) {
      return 2 * static_cast<int>(points2D_.size());
    } else {
      return 3 * static_cast<int>(points2D_.size());
    }
  }

  template <typename T>
  bool operator()(const T* const params, T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> rotation(params);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(params + 4);
    T scale = T(1);
    if constexpr (kScaled) {
      scale = ceres::exp(params[7]);
    }
    const bool use_reprojection_error =
        residual_type_ == GP3PEstimator::ResidualType::ReprojectionError;
    for (size_t i = 0; i < points2D_.size(); ++i) {
      const Eigen::Matrix<T, 3, 1> point3D_in_rig =
          scale * (rotation * points3D_[i].template cast<T>()) + translation;
      const Eigen::Matrix<T, 3, 1> point3D_in_cam =
          points2D_[i].cam_from_rig.template cast<T>() *
          point3D_in_rig.homogeneous();
      if (use_reprojection_error) {
        if (point3D_in_cam.z() <= T(std::numeric_limits<double>::epsilon())) {
          residuals[2 * i] = T(0);
          residuals[2 * i + 1] = T(0);
          continue;
        }
        const Eigen::Matrix<T, 2, 1> diff =
            points2D_[i].ray_in_cam.hnormalized().template cast<T>() -
            point3D_in_cam.hnormalized();
        residuals[2 * i] = diff.x();
        residuals[2 * i + 1] = diff.y();
      } else {
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_vec(residuals + 3 * i);
        if (point3D_in_cam.z() <= T(std::numeric_limits<double>::epsilon())) {
          residual_vec.setZero();
          continue;
        }
        residual_vec = point3D_in_cam.normalized().cross(
            points2D_[i].ray_in_cam.normalized().template cast<T>());
      }
    }
    return true;
  }

 private:
  const std::vector<GP3PEstimator::X_t>& points2D_;
  const std::vector<Eigen::Vector3d>& points3D_;
  const GP3PEstimator::ResidualType residual_type_;
};

// Nonlinear refinement of a rig pose (rigid or scaled, selected by kScaled
// with a matching Sim3d/Rigid3d model) with TinySolver. Returns false and
// leaves *rig_from_world unchanged if the solve produces a non-finite result.
template <bool kScaled, typename Model>
bool RefineRigPoseWithTinySolver(
    const std::vector<GP3PEstimator::X_t>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    GP3PEstimator::ResidualType residual_type,
    Model* rig_from_world) {
  if (residual_type != GP3PEstimator::ResidualType::ReprojectionError &&
      residual_type != GP3PEstimator::ResidualType::CosineDistance) {
    LOG(FATAL_THROW) << "Invalid residual type";
  }

  TinyRigCostFunctor<kScaled> functor(points2D, points3D, residual_type);
  typename TinyRigCostFunctor<kScaled>::AutoDiffFunction f(functor);
  using Solver = TinySolver<decltype(f), RigFromWorldManifold<kScaled>>;
  Solver solver;
  typename Solver::Options options;
  options.max_num_iterations = 25;

  constexpr int kNumParams = kScaled ? 8 : 7;
  Eigen::Matrix<double, kNumParams, 1> x;
  x.template head<4>() = rig_from_world->rotation().normalized().coeffs();
  x.template segment<3>(4) = rig_from_world->translation();
  if constexpr (kScaled) {
    x[7] = std::log(rig_from_world->scale());
  }
  solver.Solve(f, &x, options);

  if (!x.allFinite()) {
    return false;
  }

  if constexpr (kScaled) {
    *rig_from_world = Sim3d(std::exp(x[7]),
                            Eigen::Quaterniond(x.data()).normalized(),
                            x.template segment<3>(4));
  } else {
    *rig_from_world = Rigid3d(Eigen::Quaterniond(x.data()).normalized(),
                              x.template segment<3>(4));
  }
  return true;
}

void ComputeOriginsInRig(const std::vector<GP3PEstimator::X_t>& points2D,
                         std::vector<Eigen::Vector3d>* origins_in_rig) {
  const size_t num_points = points2D.size();
  origins_in_rig->resize(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    (*origins_in_rig)[i] = points2D[i].cam_from_rig.leftCols<3>().transpose() *
                           -points2D[i].cam_from_rig.col(3);
  }
}

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

bool IsPanoramicRig(const std::vector<Eigen::Vector3d>& origins_in_rig) {
  for (size_t i = 1; i < origins_in_rig.size(); ++i) {
    if (!origins_in_rig[0].isApprox(origins_in_rig[i], 1e-6)) {
      return false;
    }
  }
  return true;
}

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
  if (IsPanoramicRig(origins_in_rig)) {
    // In case of a panoramic camera/rig, fall back to P3P.
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

bool GP3PEstimator::Refine(const std::vector<X_t>& points2D,
                           const std::vector<Y_t>& points3D,
                           M_t* rig_from_world) const {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_NOTNULL(rig_from_world);

  if (points2D.size() < static_cast<size_t>(kMinNumSamples)) {
    return false;
  }

  return RefineRigPoseWithTinySolver</*kScaled=*/false>(
      points2D, points3D, residual_type_, rig_from_world);
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
  if (IsPanoramicRig(origins_in_rig)) {
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
                            M_t* rig_from_world) const {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_NOTNULL(rig_from_world);

  // The RANSAC loop can propose non-positive scales, for which the log-space
  // parameterization below is undefined. Unlike the public refinement in
  // generalized_pose.h, which throws on such an input, this is a soft failure
  // that only skips the local optimization.
  if (!(rig_from_world->scale() > 0)) {
    return false;
  }

  if (points2D.size() < static_cast<size_t>(kMinNumSamples)) {
    return false;
  }

  // The scale of the rig geometry is unobservable from a single projection
  // center.
  std::vector<Eigen::Vector3d> origins_in_rig;
  ComputeOriginsInRig(points2D, &origins_in_rig);
  if (IsPanoramicRig(origins_in_rig)) {
    return false;
  }

  return RefineRigPoseWithTinySolver</*kScaled=*/true>(
      points2D, points3D, residual_type_, rig_from_world);
}

void GP4PSEstimator::Residuals(const std::vector<X_t>& points2D,
                               const std::vector<Y_t>& points3D,
                               const M_t& rig_from_world,
                               std::vector<double>* residuals) const {
  ComputeRayResiduals(
      points2D, points3D, rig_from_world.ToMatrix(), residual_type_, residuals);
}

}  // namespace colmap
