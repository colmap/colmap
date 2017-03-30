// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_ESTIMATORS_TWO_VIEW_GEOMETRY_H_
#define COLMAP_SRC_ESTIMATORS_TWO_VIEW_GEOMETRY_H_

#include "base/camera.h"
#include "base/feature.h"
#include "optim/ransac.h"
#include "util/alignment.h"
#include "util/logging.h"

namespace colmap {

// Two-view geometry estimator.
struct TwoViewGeometry {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // The configuration of the estimated two-view geometry.
  enum ConfigurationType {
    UNDEFINED = 0,
    // Degenerate configuration (e.g., no overlap or not enough inliers).
    DEGENERATE = 1,
    // Essential matrix.
    CALIBRATED = 2,
    // Fundamental matrix.
    UNCALIBRATED = 3,
    // Homography, planar scene with baseline.
    PLANAR = 4,
    // Homography, pure rotation without baseline.
    PANORAMIC = 5,
    // Homography, planar or panoramic.
    PLANAR_OR_PANORAMIC = 6,
    // Watermark, pure 2D translation in image borders.
    WATERMARK = 7,
    // Multi-model configuration, i.e. the inlier matches result from multiple
    // individual, non-degenerate configurations.
    MULTIPLE = 8,
  };

  // Estimation options.
  struct Options {
    // Minimum number of inliers for non-degenerate two-view geometry.
    size_t min_num_inliers = 15;

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

    // Options used to robustly estimate the geometry.
    RANSACOptions ransac_options;

    void Check() const {
      CHECK_GE(min_num_inliers, 0);
      CHECK_GE(min_E_F_inlier_ratio, 0);
      CHECK_LE(min_E_F_inlier_ratio, 1);
      CHECK_GE(max_H_inlier_ratio, 0);
      CHECK_LE(max_H_inlier_ratio, 1);
      CHECK_GE(watermark_min_inlier_ratio, 0);
      CHECK_LE(watermark_min_inlier_ratio, 1);
      CHECK_GE(watermark_border_size, 0);
      CHECK_LE(watermark_border_size, 1);
      ransac_options.Check();
    }
  };

  TwoViewGeometry()
      : config(ConfigurationType::UNDEFINED),
        E(Eigen::Matrix3d::Zero()),
        F(Eigen::Matrix3d::Zero()),
        H(Eigen::Matrix3d::Zero()),
        qvec(Eigen::Vector4d::Zero()),
        tvec(Eigen::Vector3d::Zero()),
        tri_angle(0),
        E_num_inliers(0),
        F_num_inliers(0),
        H_num_inliers(0) {}

  // Estimate two-view geometry from calibrated or uncalibrated image pair,
  // depending on whether a prior focal length is given or not.
  //
  // @param camera1         Camera of first image.
  // @param points1         Feature points in first image.
  // @param camera2         Camera of second image.
  // @param points2         Feature points in second image.
  // @param matches         Feature matches between first and second image.
  // @param options         Two-view geometry estimation options.
  void Estimate(const Camera& camera1,
                const std::vector<Eigen::Vector2d>& points1,
                const Camera& camera2,
                const std::vector<Eigen::Vector2d>& points2,
                const FeatureMatches& matches, const Options& options);

  // Recursively estimate multiple configurations by removing the previous set
  // of inliers from the matches until not enough inliers are found. Inlier
  // matches are concatenated and the configuration type is `MULTIPLE` if
  // multiple models could be estimated. This is useful to estimate the two-view
  // geometry for images with large distortion or multiple rigidly moving
  // objects in the scene.
  //
  // Note that in case the model type is `MULTIPLE`, only the `inlier_matches`
  // field will be initialized.
  //
  // @param camera1         Camera of first image.
  // @param points1         Feature points in first image.
  // @param camera2         Camera of second image.
  // @param points2         Feature points in second image.
  // @param matches         Feature matches between first and second image.
  // @param options         Two-view geometry estimation options.
  void EstimateMultiple(const Camera& camera1,
                        const std::vector<Eigen::Vector2d>& points1,
                        const Camera& camera2,
                        const std::vector<Eigen::Vector2d>& points2,
                        const FeatureMatches& matches, const Options& options);

  // Estimate two-view geometry and its relative pose from a calibrated or an
  // uncalibrated image pair.
  //
  // @param camera1         Camera of first image.
  // @param points1         Feature points in first image.
  // @param camera2         Camera of second image.
  // @param points2         Feature points in second image.
  // @param matches         Feature matches between first and second image.
  // @param options         Two-view geometry estimation options.
  void EstimateWithRelativePose(const Camera& camera1,
                                const std::vector<Eigen::Vector2d>& points1,
                                const Camera& camera2,
                                const std::vector<Eigen::Vector2d>& points2,
                                const FeatureMatches& matches,
                                const Options& options);

  // Estimate two-view geometry from calibrated image pair.
  //
  // @param camera1         Camera of first image.
  // @param points1         Feature points in first image.
  // @param camera2         Camera of second image.
  // @param points2         Feature points in second image.
  // @param matches         Feature matches between first and second image.
  // @param options         Two-view geometry estimation options.
  void EstimateCalibrated(const Camera& camera1,
                          const std::vector<Eigen::Vector2d>& points1,
                          const Camera& camera2,
                          const std::vector<Eigen::Vector2d>& points2,
                          const FeatureMatches& matches,
                          const Options& options);

  // Estimate two-view geometry from uncalibrated image pair.
  //
  // @param camera1         Camera of first image.
  // @param points1         Feature points in first image.
  // @param camera2         Camera of second image.
  // @param points2         Feature points in second image.
  // @param matches         Feature matches between first and second image.
  // @param options         Two-view geometry estimation options.
  void EstimateUncalibrated(const Camera& camera1,
                            const std::vector<Eigen::Vector2d>& points1,
                            const Camera& camera2,
                            const std::vector<Eigen::Vector2d>& points2,
                            const FeatureMatches& matches,
                            const Options& options);

  // Detect if inlier matches are caused by a watermark.
  // A watermark causes a pure translation in the border are of the image.
  static bool DetectWatermark(const Camera& camera1,
                              const std::vector<Eigen::Vector2d>& points1,
                              const Camera& camera2,
                              const std::vector<Eigen::Vector2d>& points2,
                              const size_t num_inliers,
                              const std::vector<char>& inlier_mask,
                              const Options& options);

  // One of `ConfigurationType`.
  int config;

  // Essential matrix.
  Eigen::Matrix3d E;
  // Fundamental matrix.
  Eigen::Matrix3d F;
  // Homography matrix.
  Eigen::Matrix3d H;

  // Relative pose.
  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;

  // Inlier matches of the configuration.
  FeatureMatches inlier_matches;
  std::vector<char> inlier_mask;

  // Median triangulation angle.
  double tri_angle;

  // Number of inliers.
  size_t E_num_inliers;
  size_t F_num_inliers;
  size_t H_num_inliers;
};

}  // namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(colmap::TwoViewGeometry)

#endif  // COLMAP_SRC_ESTIMATORS_TWO_VIEW_GEOMETRY_H_
