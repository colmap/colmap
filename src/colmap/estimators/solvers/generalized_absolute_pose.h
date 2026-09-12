// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/sim3.h"
#include "colmap/util/eigen_alignment.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Whether the given projection centers, e.g., of the cameras in a rig or of
// the observations in a sample, all coincide up to a fixed tolerance. Such a
// rig acts as a single central camera, for which the scale of the rig
// geometry is unobservable.
bool IsPanoramicRig(const std::vector<Eigen::Vector3d>& origins_in_rig);

// Solver for the Generalized P3P problem.
class GP3PEstimator {
 public:
  // The generalized image observations, which is composed of the relative pose
  // of a camera in the generalized camera and a ray in the camera frame.
  struct X_t {
    Eigen::Matrix3x4d cam_from_rig;  // Stored as matrix for fast residuals
    Eigen::Vector3d ray_in_cam;
  };

  // The observed 3D feature points in the world frame.
  using Y_t = Eigen::Vector3d;
  // The estimated rig_from_world pose of the generalized camera.
  using M_t = Rigid3d;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  // Whether to compute the cosine similarity or the reprojection error.
  // [WARNING] The reprojection error being in normalized coordinates,
  // the unique error threshold of RANSAC corresponds to different pixel values
  // in the different cameras of the rig if they have different intrinsics.
  enum class ResidualType {
    CosineDistance,
    ReprojectionError,
  };

  explicit GP3PEstimator(
      ResidualType residual_type = ResidualType::CosineDistance);

  // Estimate the most probable solution of the GP3P problem from a set of
  // three 2D-3D point correspondences.
  static void Estimate(const std::vector<X_t>& points2D,
                       const std::vector<Y_t>& points3D,
                       std::vector<M_t>* models);

  // Nonlinear local optimization of the rig pose over the given 2D-3D
  // correspondences, starting from *rig_from_world. It is required because
  // the minimal solver consumes exactly three points and has no non-minimal
  // counterpart. Minimizes the estimator's residual: the reprojection error
  // between the normalized rays, or the sine of the angle between the
  // observed and projected rays for cosine-distance scoring (same minimizer,
  // but with a non-vanishing Jacobian at zero). Observations that do not
  // project in front of their camera contribute a zero residual.
  //
  // Returns true and overwrites *rig_from_world with the refined transform on
  // success. Returns false and leaves *rig_from_world unchanged if fewer
  // than kMinNumSamples observations are given. Unlike the scaled variant,
  // the rigid pose stays observable for a single projection center.
  bool Refine(const std::vector<X_t>& points2D,
              const std::vector<Y_t>& points3D,
              M_t* rig_from_world) const;

  // Calculate the squared cosine distance error between the rays given a set of
  // 2D-3D point correspondences and the rig pose of the generalized camera.
  void Residuals(const std::vector<X_t>& points2D,
                 const std::vector<Y_t>& points3D,
                 const M_t& rig_from_world,
                 std::vector<double>* residuals) const;

 private:
  const ResidualType residual_type_;
};

// Solver for the Generalized P4P + scale (GP4PS) problem.
//
// Jointly estimates the pose of a generalized camera and the scale of its
// internal geometry from four 2D-3D point correspondences. The rig geometry
// (cams_from_rig) is treated as rigid but of unknown scale, e.g., camera
// poses from a monocular odometry trajectory in an arbitrarily scaled frame.
class GP4PSEstimator {
 public:
  // Same generalized image observations as in the GP3P problem.
  using X_t = GP3PEstimator::X_t;
  // The observed 3D feature points in the world frame.
  using Y_t = Eigen::Vector3d;
  // The estimated rig_from_world transform of the generalized camera,
  // mapping world points into the rig frame in which the given cams_from_rig
  // are valid. Its scale is the inverse scale of the rig geometry relative
  // to the world.
  using M_t = Sim3d;

  using ResidualType = GP3PEstimator::ResidualType;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 4;

  explicit GP4PSEstimator(
      ResidualType residual_type = ResidualType::CosineDistance);

  // Estimate the most probable solution of the GP4PS problem from a set of
  // four 2D-3D point correspondences. Produces no solutions for panoramic
  // rigs (coinciding camera centers), for which scale is unobservable.
  static void Estimate(const std::vector<X_t>& points2D,
                       const std::vector<Y_t>& points3D,
                       std::vector<M_t>* models);

  // Nonlinear local optimization of the scaled rig pose over the given 2D-3D
  // correspondences, starting from *rig_from_world. It is required because
  // the minimal solver consumes exactly four points and has no non-minimal
  // counterpart. Minimizes the estimator's residual: the reprojection error
  // between the normalized rays, or the sine of the angle between the
  // observed and projected rays for cosine-distance scoring (same minimizer,
  // but with a non-vanishing Jacobian at zero). The scale is optimized in
  // log-space so that it stays positive. Observations that do not project
  // in front of their camera contribute a zero residual.
  //
  // Returns true and overwrites *rig_from_world with the refined transform on
  // success. Returns false and leaves *rig_from_world unchanged if the
  // initial scale is not positive, if fewer than kMinNumSamples observations
  // are given, or if they share a single projection center, for which the
  // scale is unobservable.
  bool Refine(const std::vector<X_t>& points2D,
              const std::vector<Y_t>& points3D,
              M_t* rig_from_world) const;

  void Residuals(const std::vector<X_t>& points2D,
                 const std::vector<Y_t>& points3D,
                 const M_t& rig_from_world,
                 std::vector<double>* residuals) const;

 private:
  const ResidualType residual_type_;
};

}  // namespace colmap
