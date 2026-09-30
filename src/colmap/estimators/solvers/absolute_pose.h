// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <optional>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// Function mapping 3D point in the camera frame to 2D point in the image.
// Returns null if the point projection is invalid (e.g., behind the camera).
using ImgFromCamFunc =
    std::function<std::optional<Eigen::Vector2d>(const Eigen::Vector3d&)>;

struct Point2DWithRay {
  // The 2D image point in pixels.
  Eigen::Vector2d image_point;
  // The normalized 3D ray direction in the camera frame.
  Eigen::Vector3d camera_ray;
};

class P3PEstimator {
 public:
  // The 2D image feature observations.
  using X_t = Point2DWithRay;
  // The observed 3D features in the world frame.
  using Y_t = Eigen::Vector3d;
  // The transformation from the world to the camera frame.
  using M_t = Rigid3d;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  explicit P3PEstimator(ImgFromCamFunc img_from_cam_func);

  // Estimate the most probable solution of the P3P problem from a set of
  // three 2D-3D point correspondences.
  //
  // @param points2D         2D image observations with rays.
  // @param points3D         3D world points.
  // @param cams_from_world  Output vector of rigid transformations.
  void Estimate(const std::vector<X_t>& points2D,
                const std::vector<Y_t>& points3D,
                std::vector<M_t>* cams_from_world) const;

  // Calculate the squared reprojection error given a set of 2D-3D point
  // correspondences and a projection matrix.
  //
  // @param points2D        2D image observations with rays.
  // @param points3D        3D world points.
  // @param cam_from_world  Rigid transformation from world to camera frame.
  // @param residuals       Output vector of residuals.
  void Residuals(const std::vector<X_t>& points2D,
                 const std::vector<Y_t>& points3D,
                 const M_t& cam_from_world,
                 std::vector<double>* residuals) const;

  // Nonlinear local optimization of the pose over the given 2D-3D
  // correspondences, starting from *cam_from_world. Minimizes
  // normalized-plane reprojection errors with Levenberg-Marquardt
  // (colmap::TinySolver), with the rotation on the quaternion manifold and an
  // autodiff Jacobian. The refinement is independent of the camera model
  // (observations enter as rays); scoring uses the pixel reprojection error.
  //
  // Returns true and overwrites *cam_from_world with the refined transform on
  // success. Returns false and leaves *cam_from_world unchanged if fewer than
  // kMinNumSamples observations are given or the solve fails.
  //
  // @param points2D        2D image observations with rays.
  // @param points3D        3D world points.
  // @param cam_from_world  Rigid transformation, refined in place.
  bool Refine(const std::vector<X_t>& points2D,
              const std::vector<Y_t>& points3D,
              M_t* cam_from_world) const;

 private:
  const ImgFromCamFunc img_from_cam_func_;
};

// Minimal solver for 6-DOF pose and focal length.
class P4PFEstimator {
 public:
  // The 2D image feature observations.
  // Expected to be normalized by the principal point.
  using X_t = Eigen::Vector2d;
  // The observed 3D features in the world frame.
  using Y_t = Eigen::Vector3d;
  struct M_t {
    // The transformation from the world to the camera frame.
    Rigid3d cam_from_world;
    // The focal lengths (fx, fy) of the camera. Equal when the focal length is
    // shared (e.g. single-focal camera models).
    Eigen::Vector2d focal_lengths = Eigen::Vector2d::Zero();
  };

  static const int kMinNumSamples = 4;

  // If share_focal_length is true, a single shared focal length is estimated
  // (suitable for single-focal camera models, e.g. SIMPLE_PINHOLE). Otherwise,
  // separate focal lengths for x and y are estimated (e.g. PINHOLE, OPENCV).
  explicit P4PFEstimator(bool share_focal_length = true);

  void Estimate(const std::vector<X_t>& points2D,
                const std::vector<Y_t>& points3D,
                std::vector<M_t>* models) const;

  static void Residuals(const std::vector<X_t>& points2D,
                        const std::vector<Y_t>& points3D,
                        const M_t& model,
                        std::vector<double>* residuals);

  // Nonlinear local optimization of the pose and focal length(s) over the
  // given 2D-3D correspondences, starting from *model. Minimizes pixel
  // reprojection errors with Levenberg-Marquardt (colmap::TinySolver), with
  // the rotation on the quaternion manifold, an autodiff Jacobian, and the
  // focal length(s) in log-space so that they stay positive.
  //
  // Returns true and overwrites *model with the refined estimate on success.
  // Returns false and leaves *model unchanged if fewer than kMinNumSamples
  // observations are given, if the initial focal length(s) are not positive,
  // or if the solve fails.
  //
  // @param points2D  2D image feature observations, normalized by the
  //                  principal point.
  // @param points3D  3D world points.
  // @param model     Model to refine in place.
  bool Refine(const std::vector<X_t>& points2D,
              const std::vector<Y_t>& points3D,
              M_t* model) const;

 private:
  const bool share_focal_length_;
};

// Compute squared reprojection error in pixels.
void ComputeSquaredReprojectionError(
    const std::vector<Point2DWithRay>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const ImgFromCamFunc& img_from_cam_func,
    std::vector<double>* residuals);

}  // namespace colmap
