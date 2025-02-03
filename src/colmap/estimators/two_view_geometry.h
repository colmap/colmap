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

#include "colmap/feature/types.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/logging.h"

namespace colmap {

// Estimation options.
struct TwoViewGeometryOptions {
  // Minimum number of inliers for non-degenerate two-view geometry.
  int min_num_inliers = 15;

  // In case both cameras are calibrated, the calibration is verified by
  // estimating an essential and fundamental matrix and comparing their
  // fractions of number of inliers. If the essential matrix produces
  // a similar number of inliers (`min_E_F_inlier_ratio * F_num_inliers`),
  // the calibration is assumed to be correct.
  double min_E_F_inlier_ratio = 0.95;

  // In case an epipolar geometry can be verified, it is checked whether
  // the geometry describes a planar scene or panoramic view (pure rotation)
  // described by a homography. This is a degenerate case, since epipolar
  // geometry is only defined for a moving camera. If the inlier ratio of
  // a homography comes close to the inlier ratio of the epipolar geometry,
  // a planar or panoramic configuration is assumed.
  double max_H_inlier_ratio = 0.8;

  // In case of valid two-view geometry, it is checked whether the geometry
  // describes a pure translation in the border region of the image. If more
  // than a certain ratio of inlier points conform with a pure image
  // translation, a watermark is assumed.
  double watermark_min_inlier_ratio = 0.7;

  // Watermark matches have to be in the border region of the image. The
  // border region is defined as the area around the image borders and
  // is defined as a fraction of the image diagonal.
  double watermark_border_size = 0.1;

  // Whether to enable watermark detection. A watermark causes a pure
  // translation in the image space with inliers in the border region.
  bool detect_watermark = true;

  // Whether to ignore watermark models in multiple model estimation.
  bool multiple_ignore_watermark = true;

  // In case the user asks for it, only going to estimate a Homography
  // between both cameras.
  bool force_H_use = false;

  // Whether to compute the relative pose between the two views.
  bool compute_relative_pose = false;

  // Recursively estimate multiple configurations by removing the previous set
  // of inliers from the matches until not enough inliers are found. Inlier
  // matches are concatenated and the configuration type is `MULTIPLE` if
  // multiple models could be estimated. This is useful to estimate the two-view
  // geometry for images with large distortion or multiple rigidly moving
  // objects in the scene.
  //
  // Note that in case the model type is `MULTIPLE`, only the `inlier_matches`
  // field will be initialized.
  bool multiple_models = false;

  // TwoViewGeometryOptions used to robustly estimate the geometry.
  RANSACOptions ransac_options;

  TwoViewGeometryOptions() {
    ransac_options.max_error = 4.0;
    ransac_options.confidence = 0.999;
    ransac_options.min_num_trials = 100;
    ransac_options.max_num_trials = 10000;
    ransac_options.min_inlier_ratio = 0.25;
  }

  bool Check() const;
};

// Estimate two-view geometry from calibrated or uncalibrated image pair,
// depending on whether a prior focal length is given or not.
//
// @param camera1         Camera of first image.
// @param points1         Feature points in first image.
// @param camera2         Camera of second image.
// @param points2         Feature points in second image.
// @param matches         Feature matches between first and second image.
// @param options         Two-view geometry estimation options.
TwoViewGeometry EstimateTwoViewGeometry(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches,
    const TwoViewGeometryOptions& options);

// Estimate relative pose for two-view geometry.
//
// @param camera1         Camera of first image.
// @param points1         Feature points in first image.
// @param camera2         Camera of second image.
// @param points2         Feature points in second image.
// @param matches         Feature matches between first and second image.
// @param options         Two-view geometry estimation options.
bool EstimateTwoViewGeometryPose(const Camera& camera1,
                                 const std::vector<Eigen::Vector2d>& points1,
                                 const Camera& camera2,
                                 const std::vector<Eigen::Vector2d>& points2,
                                 TwoViewGeometry* geometry);

// Estimate two-view geometry from calibrated image pair.
//
// @param camera1         Camera of first image.
// @param points1         Feature points in first image.
// @param camera2         Camera of second image.
// @param points2         Feature points in second image.
// @param matches         Feature matches between first and second image.
// @param options         Two-view geometry estimation options.
TwoViewGeometry EstimateCalibratedTwoViewGeometry(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches,
    const TwoViewGeometryOptions& options);

// Detect if inlier matches are caused by a watermark.
// A watermark causes a pure translation in the border are of the image.
bool DetectWatermark(const Camera& camera1,
                     const std::vector<Eigen::Vector2d>& points1,
                     const Camera& camera2,
                     const std::vector<Eigen::Vector2d>& points2,
                     size_t num_inliers,
                     const std::vector<char>& inlier_mask,
                     const TwoViewGeometryOptions& options);

}  // namespace colmap
