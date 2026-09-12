// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/camera.h"
#include "colmap/util/eigen_alignment.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Direct linear transformation algorithm to compute the homography between
// point pairs. This algorithm computes the least squares estimate for
// the homography from at least 4 correspondences.
class HomographyMatrixEstimator {
 public:
  using X_t = Eigen::Vector2d;
  using Y_t = Eigen::Vector2d;
  using M_t = Eigen::Matrix3d;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 4;

  // Estimate the projective transformation (homography).
  //
  // The number of corresponding points must be at least 4.
  //
  // @param points1    First set of corresponding points.
  // @param points2    Second set of corresponding points.
  //
  // @return         3x3 homogeneous transformation matrix.
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  // Calculate the transformation error for each corresponding point pair.
  //
  // Residuals are defined as the squared transformation error when
  // transforming the source to the destination coordinates.
  //
  // @param points1    First set of corresponding points.
  // @param points2    Second set of corresponding points.
  // @param H          3x3 projective matrix.
  // @param residuals  Output vector of residuals.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& H,
                        std::vector<double>* residuals);
};

// A ray together with the image point it was unprojected from. The ray drives
// the estimation, the image point is what the residual is measured against.
struct CamRayWithImgPoint {
  Eigen::Vector3d ray;
  Eigen::Vector2d img_point;
};

// Direct linear transformation algorithm to compute the homography between
// bearing rays. A world plane induces x2 ~ H x1 on the rays of any central
// camera, whereas image points are related by a 3x3 matrix only under a pinhole
// projection, which every model but SIMPLE_PINHOLE and PINHOLE breaks.
//
// The rays must come from cameras with known intrinsics. The estimated H maps
// rays to rays; conjugate it as K2 H K1^-1 for the pixel-space homography,
// which spherical cameras have no calibration matrix for.
class HomographyMatrixRayEstimator {
 public:
  using X_t = Eigen::Vector3d;
  using Y_t = CamRayWithImgPoint;
  using M_t = Eigen::Matrix3d;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 4;

  // RANSAC copies the estimator per thread, so this must stay cheap to copy.
  // The camera must outlive the estimator.
  explicit HomographyMatrixRayEstimator(const Camera* camera2 = nullptr)
      : camera2_(camera2) {}

  // Estimate the projective transformation (homography) between bearing rays.
  //
  // The number of corresponding rays must be at least 4.
  //
  // @param cam_rays1  First set of corresponding rays.
  // @param cam_rays2  Second set of corresponding rays and image points.
  //
  // @return         3x3 homogeneous transformation matrix.
  void Estimate(const std::vector<X_t>& cam_rays1,
                const std::vector<Y_t>& cam_rays2,
                std::vector<M_t>* models) const;

  // Calculate the transformation error for each corresponding ray pair.
  //
  // Residuals are defined as the squared transformation error in the second
  // camera's pixels: the first ray is transferred through H and projected back
  // into the image. Unlike the essential matrix, a homography transfers a point
  // to a point rather than to a line, so this is the exact geometric error and
  // needs no linearization. Rays with no image point, i.e. transferred behind a
  // perspective camera, are scored at the maximum residual.
  //
  // @param cam_rays1  First set of corresponding rays.
  // @param cam_rays2  Second set of corresponding rays and image points.
  // @param H          3x3 projective matrix mapping rays to rays.
  // @param residuals  Output vector of residuals.
  void Residuals(const std::vector<X_t>& cam_rays1,
                 const std::vector<Y_t>& cam_rays2,
                 const M_t& H,
                 std::vector<double>* residuals) const;

 private:
  const Camera* camera2_;
};

}  // namespace colmap
