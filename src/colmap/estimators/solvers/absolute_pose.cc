// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/absolute_pose.h"

#include "colmap/estimators/cost_functions/tiny_manifold.h"
#include "colmap/estimators/solvers/poselib_utils.h"
#include "colmap/optim/tiny_solver.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <cmath>

#include <Eigen/Geometry>
#include <PoseLib/solvers/p3p.h>
#include <PoseLib/solvers/p4pf.h>
#include <ceres/tiny_solver_autodiff_function.h>

namespace colmap {
namespace {

// The manifold of a Rigid3d transform: rotation on SO(3), with the translation
// as Euclidean parameters. The ambient parameter layout matches
// TinyPnPCostFunctor: [qx, qy, qz, qw, tx, ty, tz].
using Rigid3dManifold =
    ProductManifold<EigenQuaternionManifold, EuclideanManifold<3>>;

// Cost functor for colmap::TinySolver refinement of a cam_from_world transform
// over all given 2D-3D correspondences, minimizing normalized-plane
// reprojection errors (two residuals per observation). Working in normalized
// coordinates keeps the functor independent of the camera model: observations
// enter as rays. For pinhole cameras this shares the minimizer of the pixel
// reprojection error scored by the estimator.
//
// Observations with degenerate rays or that project behind the camera
// contribute a zero residual, as in the other reprojection cost functors.
// Cheirality is instead enforced by the estimator's Residuals when the refined
// model is scored.
class TinyPnPCostFunctor {
 public:
  using Scalar = double;
  static constexpr int NUM_RESIDUALS = Eigen::Dynamic;
  static constexpr int NUM_PARAMETERS = 7;

  // ceres::TinySolver-compatible autodiff wrapper for this functor.
  using AutoDiffFunction = ceres::TinySolverAutoDiffFunction<TinyPnPCostFunctor,
                                                             NUM_RESIDUALS,
                                                             NUM_PARAMETERS>;

  TinyPnPCostFunctor(const std::vector<Point2DWithRay>& points2D,
                     const std::vector<Eigen::Vector3d>& points3D)
      : points2D_(points2D), points3D_(points3D) {}

  int NumResiduals() const { return 2 * static_cast<int>(points2D_.size()); }

  template <typename T>
  bool operator()(const T* const params, T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> rotation(params);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(params + 4);
    for (size_t i = 0; i < points2D_.size(); ++i) {
      Eigen::Map<Eigen::Matrix<T, 2, 1>> residual_vec(residuals + 2 * i);
      if (points2D_[i].camera_ray.z() <= 0) {
        residual_vec.setZero();
        continue;
      }
      const Eigen::Matrix<T, 3, 1> point_in_cam =
          rotation * points3D_[i].template cast<T>() + translation;
      if (point_in_cam.z() <= T(0)) {
        residual_vec.setZero();
        continue;
      }
      residual_vec = points2D_[i].camera_ray.hnormalized().template cast<T>() -
                     point_in_cam.hnormalized();
    }
    return true;
  }

 private:
  const std::vector<Point2DWithRay>& points2D_;
  const std::vector<Eigen::Vector3d>& points3D_;
};

// The manifold of a P4PF model: rotation on SO(3), with the translation and
// log focal length(s) as Euclidean parameters. The ambient parameter layout
// matches TinyPnPFCostFunctor: [qx, qy, qz, qw, tx, ty, tz, logf] with an
// extra logf entry for separate focal lengths.
template <bool kSharedFocal>
using PnPFManifold =
    ProductManifold<Rigid3dManifold, EuclideanManifold<kSharedFocal ? 1 : 2>>;

// Cost functor for colmap::TinySolver refinement of a P4PF model (pose plus
// shared or separate focal lengths, selected by kSharedFocal) over all given
// 2D-3D correspondences, minimizing pixel reprojection errors (two residuals
// per observation) — the same error scored by P4PFEstimator::Residuals. The
// focal length(s) are optimized in log-space so that they stay positive.
// Observations that project behind the camera contribute a zero residual.
template <bool kSharedFocal>
class TinyPnPFCostFunctor {
 public:
  using Scalar = double;
  static constexpr int NUM_RESIDUALS = Eigen::Dynamic;
  static constexpr int NUM_PARAMETERS = kSharedFocal ? 8 : 9;

  // ceres::TinySolver-compatible autodiff wrapper for this functor.
  using AutoDiffFunction =
      ceres::TinySolverAutoDiffFunction<TinyPnPFCostFunctor,
                                        NUM_RESIDUALS,
                                        NUM_PARAMETERS>;

  TinyPnPFCostFunctor(const std::vector<Eigen::Vector2d>& points2D,
                      const std::vector<Eigen::Vector3d>& points3D)
      : points2D_(points2D), points3D_(points3D) {}

  int NumResiduals() const { return 2 * static_cast<int>(points2D_.size()); }

  template <typename T>
  bool operator()(const T* const params, T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> rotation(params);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(params + 4);
    const T focal_x = ceres::exp(params[7]);
    const T focal_y = kSharedFocal ? focal_x : ceres::exp(params[8]);
    for (size_t i = 0; i < points2D_.size(); ++i) {
      Eigen::Map<Eigen::Matrix<T, 2, 1>> residual_vec(residuals + 2 * i);
      const Eigen::Matrix<T, 3, 1> point_in_cam =
          rotation * points3D_[i].template cast<T>() + translation;
      if (point_in_cam.z() <= T(0)) {
        residual_vec.setZero();
        continue;
      }
      residual_vec.x() =
          focal_x * point_in_cam.x() / point_in_cam.z() - T(points2D_[i].x());
      residual_vec.y() =
          focal_y * point_in_cam.y() / point_in_cam.z() - T(points2D_[i].y());
    }
    return true;
  }

 private:
  const std::vector<Eigen::Vector2d>& points2D_;
  const std::vector<Eigen::Vector3d>& points3D_;
};

// Nonlinear refinement of a P4PF model (shared or separate focal lengths,
// selected by kSharedFocal) with TinySolver. Returns false and leaves *model
// unchanged if the solve produces a non-finite result.
template <bool kSharedFocal>
bool RefinePnPFPoseWithTinySolver(const std::vector<Eigen::Vector2d>& points2D,
                                  const std::vector<Eigen::Vector3d>& points3D,
                                  P4PFEstimator::M_t* model) {
  TinyPnPFCostFunctor<kSharedFocal> functor(points2D, points3D);
  typename TinyPnPFCostFunctor<kSharedFocal>::AutoDiffFunction f(functor);
  using Solver = TinySolver<decltype(f), PnPFManifold<kSharedFocal>>;
  Solver solver;
  typename Solver::Options options;
  options.max_num_iterations = 25;

  constexpr int kNumParams = kSharedFocal ? 8 : 9;
  Eigen::Matrix<double, kNumParams, 1> x;
  x.template head<4>() = model->cam_from_world.rotation().normalized().coeffs();
  x.template segment<3>(4) = model->cam_from_world.translation();
  x[7] = std::log(model->focal_lengths.x());
  if constexpr (!kSharedFocal) {
    x[8] = std::log(model->focal_lengths.y());
  }
  if (solver.Solve(f, &x, options).status == Solver::NUMERICAL_FAILURE) {
    return false;
  }
  model->cam_from_world.rotation() = Eigen::Quaterniond(x.data()).normalized();
  model->cam_from_world.translation() = x.template segment<3>(4);
  model->focal_lengths.x() = std::exp(x[7]);
  if constexpr (kSharedFocal) {
    model->focal_lengths.y() = std::exp(x[7]);
  } else {
    model->focal_lengths.y() = std::exp(x[8]);
  }
  return true;
}

}  // namespace

P3PEstimator::P3PEstimator(ImgFromCamFunc img_from_cam_func)
    : img_from_cam_func_(std::move(img_from_cam_func)) {}

void P3PEstimator::Estimate(const std::vector<X_t>& points2D,
                            const std::vector<Y_t>& points3D,
                            std::vector<M_t>* cams_from_world) const {
  THROW_CHECK_EQ(points2D.size(), 3);
  THROW_CHECK_EQ(points3D.size(), 3);
  THROW_CHECK_NOTNULL(cams_from_world);

  cams_from_world->clear();

  std::vector<Eigen::Vector3d> rays(3);
  for (int i = 0; i < 3; ++i) {
    rays[i] = points2D[i].camera_ray;
  }

  std::vector<poselib::CameraPose> poses;
  const int num_poses = poselib::p3p(rays, points3D, &poses);

  cams_from_world->resize(num_poses);
  for (int i = 0; i < num_poses; ++i) {
    (*cams_from_world)[i] = ConvertPoseLibPoseToRigid3d(poses[i]);
  }
}

void P3PEstimator::Residuals(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             const M_t& cam_from_world,
                             std::vector<double>* residuals) const {
  ComputeSquaredReprojectionError(points2D,
                                  points3D,
                                  cam_from_world.ToMatrix(),
                                  img_from_cam_func_,
                                  residuals);
}

bool P3PEstimator::Refine(const std::vector<X_t>& points2D,
                          const std::vector<Y_t>& points3D,
                          M_t* cam_from_world) const {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_NOTNULL(cam_from_world);
  if (points2D.size() < kMinNumSamples) {
    return false;
  }

  TinyPnPCostFunctor functor(points2D, points3D);
  TinyPnPCostFunctor::AutoDiffFunction f(functor);
  using Solver = TinySolver<decltype(f), Rigid3dManifold>;
  Solver solver;
  typename Solver::Options options;
  options.max_num_iterations = 25;

  Eigen::Matrix<double, 7, 1> x;
  x.head<4>() = cam_from_world->rotation().normalized().coeffs();
  x.tail<3>() = cam_from_world->translation();
  if (solver.Solve(f, &x, options).status == Solver::NUMERICAL_FAILURE) {
    return false;
  }
  cam_from_world->rotation() = Eigen::Quaterniond(x.data()).normalized();
  cam_from_world->translation() = x.tail<3>();
  return true;
}

P4PFEstimator::P4PFEstimator(bool share_focal_length)
    : share_focal_length_(share_focal_length) {}

void P4PFEstimator::Estimate(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             std::vector<M_t>* models) const {
  THROW_CHECK_EQ(points2D.size(), 4);
  THROW_CHECK_EQ(points3D.size(), 4);
  THROW_CHECK_NOTNULL(models);

  models->clear();

  std::vector<poselib::CameraPose> poses;
  std::vector<double> focals_x;
  std::vector<double> focals_y;
  int num_poses;
  if (share_focal_length_) {
    // Estimate a single shared focal length. filter_solutions additionally
    // removes solutions whose aspect ratio (fx/fy) is far from 1.
    num_poses = poselib::p4pf(
        points2D, points3D, &poses, &focals_x, /*filter_solutions=*/true);
    focals_y = focals_x;
  } else {
    // Estimate separate focal lengths for x and y. filter_solutions only
    // removes solutions with non-positive focal lengths.
    num_poses = poselib::p4pf(points2D,
                              points3D,
                              &poses,
                              &focals_x,
                              &focals_y,
                              /*filter_solutions=*/true);
  }

  models->resize(num_poses);
  for (int i = 0; i < num_poses; ++i) {
    (*models)[i].cam_from_world = ConvertPoseLibPoseToRigid3d(poses[i]);
    (*models)[i].focal_lengths = Eigen::Vector2d(focals_x[i], focals_y[i]);
  }
}

void P4PFEstimator::Residuals(const std::vector<X_t>& points2D,
                              const std::vector<Y_t>& points3D,
                              const M_t& model,
                              std::vector<double>* residuals) {
  const size_t num_points2D = points2D.size();
  CHECK_EQ(num_points2D, points3D.size());
  residuals->resize(num_points2D);
  const Eigen::Matrix3x4d cam_from_world_mat = model.cam_from_world.ToMatrix();
  for (size_t i = 0; i < num_points2D; ++i) {
    const Eigen::Vector3d point3D_in_cam =
        cam_from_world_mat * points3D[i].homogeneous();
    // Check if 3D point is in front of camera.
    if (point3D_in_cam.z() > std::numeric_limits<double>::epsilon()) {
      (*residuals)[i] =
          (model.focal_lengths.cwiseProduct(point3D_in_cam.hnormalized()) -
           points2D[i])
              .squaredNorm();
    } else {
      (*residuals)[i] = std::numeric_limits<double>::max();
    }
  }
}

bool P4PFEstimator::Refine(const std::vector<X_t>& points2D,
                           const std::vector<Y_t>& points3D,
                           M_t* model) const {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_NOTNULL(model);
  if (points2D.size() < kMinNumSamples) {
    return false;
  }
  if (model->focal_lengths.x() <= 0 ||
      (!share_focal_length_ && model->focal_lengths.y() <= 0)) {
    return false;
  }
  if (share_focal_length_) {
    return RefinePnPFPoseWithTinySolver<true>(points2D, points3D, model);
  } else {
    return RefinePnPFPoseWithTinySolver<false>(points2D, points3D, model);
  }
}

void ComputeSquaredReprojectionError(
    const std::vector<Point2DWithRay>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const ImgFromCamFunc& img_from_cam_func,
    std::vector<double>* residuals) {
  const size_t num_points = points2D.size();
  THROW_CHECK_EQ(num_points, points3D.size());
  residuals->resize(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    const Eigen::Vector3d point3D_in_cam =
        cam_from_world * points3D[i].homogeneous();
    const std::optional<Eigen::Vector2d> proj_image_point =
        img_from_cam_func(point3D_in_cam);
    if (proj_image_point) {
      (*residuals)[i] =
          (*proj_image_point - points2D[i].image_point).squaredNorm();
    } else {
      (*residuals)[i] = std::numeric_limits<double>::max();
    }
  }
}

}  // namespace colmap
