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

#include "colmap/estimators/two_view_geometry.h"

#include "colmap/estimators/essential_matrix.h"
#include "colmap/estimators/fundamental_matrix.h"
#include "colmap/estimators/homography_matrix.h"
#include "colmap/estimators/translation_transform.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/homography_matrix.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/math/random.h"
#include "colmap/optim/loransac.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/camera.h"

#include <unordered_set>

namespace colmap {
namespace {

FeatureMatches ExtractInlierMatches(const FeatureMatches& matches,
                                    const size_t num_inliers,
                                    const std::vector<char>& inlier_mask) {
  FeatureMatches inlier_matches(num_inliers);
  size_t j = 0;
  for (size_t i = 0; i < matches.size(); ++i) {
    if (inlier_mask[i]) {
      inlier_matches[j] = matches[i];
      j += 1;
    }
  }
  return inlier_matches;
}

FeatureMatches ExtractOutlierMatches(const FeatureMatches& matches,
                                     const FeatureMatches& inlier_matches) {
  THROW_CHECK_GE(matches.size(), inlier_matches.size());

  std::unordered_set<std::pair<point2D_t, point2D_t>> inlier_matches_set;
  inlier_matches_set.reserve(inlier_matches.size());
  for (const auto& match : inlier_matches) {
    inlier_matches_set.emplace(match.point2D_idx1, match.point2D_idx2);
  }

  FeatureMatches outlier_matches;
  outlier_matches.reserve(matches.size() - inlier_matches.size());

  for (const auto& match : matches) {
    if (inlier_matches_set.count(
            std::make_pair(match.point2D_idx1, match.point2D_idx2)) == 0) {
      outlier_matches.push_back(match);
    }
  }

  return outlier_matches;
}

inline bool IsImagePointInBoundingBox(const Eigen::Vector2d& point,
                                      const double minx,
                                      const double maxx,
                                      const double miny,
                                      const double maxy) {
  return point.x() >= minx && point.x() <= maxx && point.y() >= miny &&
         point.y() <= maxy;
}

TwoViewGeometry EstimateCalibratedHomography(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches,
    const TwoViewGeometryOptions& options) {
  TwoViewGeometry geometry;

  const size_t min_num_inliers = static_cast<size_t>(options.min_num_inliers);
  if (matches.size() < min_num_inliers) {
    geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
    return geometry;
  }

  // Extract corresponding points.
  std::vector<Eigen::Vector2d> matched_img_points1(matches.size());
  std::vector<Eigen::Vector2d> matched_img_points2(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    matched_img_points1[i] = points1[matches[i].point2D_idx1];
    matched_img_points2[i] = points2[matches[i].point2D_idx2];
  }

  // Estimate planar or panoramic model.

  LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(
      options.ransac_options);
  const auto H_report =
      H_ransac.Estimate(matched_img_points1, matched_img_points2);
  geometry.H = H_report.model;

  if (!H_report.success || H_report.support.num_inliers < min_num_inliers) {
    geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
    return geometry;
  } else {
    geometry.config = TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
  }

  geometry.inlier_matches = ExtractInlierMatches(
      matches, H_report.support.num_inliers, H_report.inlier_mask);
  if (options.detect_watermark && DetectWatermark(camera1,
                                                  matched_img_points1,
                                                  camera2,
                                                  matched_img_points2,
                                                  H_report.support.num_inliers,
                                                  H_report.inlier_mask,
                                                  options)) {
    geometry.config = TwoViewGeometry::ConfigurationType::WATERMARK;
  }

  if (options.compute_relative_pose) {
    EstimateTwoViewGeometryPose(camera1, points1, camera2, points2, &geometry);
  }

  return geometry;
}

TwoViewGeometry EstimateUncalibratedTwoViewGeometry(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches,
    const TwoViewGeometryOptions& options) {
  TwoViewGeometry geometry;

  const size_t min_num_inliers = static_cast<size_t>(options.min_num_inliers);
  if (matches.size() < static_cast<size_t>(min_num_inliers)) {
    geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
    return geometry;
  }

  // Extract corresponding points.
  std::vector<Eigen::Vector2d> matched_img_points1(matches.size());
  std::vector<Eigen::Vector2d> matched_img_points2(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    matched_img_points1[i] = points1[matches[i].point2D_idx1];
    matched_img_points2[i] = points2[matches[i].point2D_idx2];
  }

  // Estimate epipolar model.

  LORANSAC<FundamentalMatrixSevenPointEstimator,
           FundamentalMatrixEightPointEstimator>
      F_ransac(options.ransac_options);
  const auto F_report =
      F_ransac.Estimate(matched_img_points1, matched_img_points2);
  geometry.F = F_report.model;

  // Estimate planar or panoramic model.

  LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(
      options.ransac_options);
  const auto H_report =
      H_ransac.Estimate(matched_img_points1, matched_img_points2);
  geometry.H = H_report.model;

  if ((!F_report.success && !H_report.success) ||
      (F_report.support.num_inliers < min_num_inliers &&
       H_report.support.num_inliers < min_num_inliers)) {
    geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
    return geometry;
  }

  // Determine inlier ratios of different models.

  const double H_F_inlier_ratio =
      static_cast<double>(H_report.support.num_inliers) /
      F_report.support.num_inliers;

  const std::vector<char>* best_inlier_mask = &F_report.inlier_mask;
  int num_inliers = F_report.support.num_inliers;
  if (H_F_inlier_ratio > options.max_H_inlier_ratio) {
    geometry.config = TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
    if (H_report.support.num_inliers >= F_report.support.num_inliers) {
      num_inliers = H_report.support.num_inliers;
      best_inlier_mask = &H_report.inlier_mask;
    }
  } else {
    geometry.config = TwoViewGeometry::ConfigurationType::UNCALIBRATED;
  }

  geometry.inlier_matches =
      ExtractInlierMatches(matches, num_inliers, *best_inlier_mask);

  if (options.detect_watermark && DetectWatermark(camera1,
                                                  matched_img_points1,
                                                  camera2,
                                                  matched_img_points2,
                                                  num_inliers,
                                                  *best_inlier_mask,
                                                  options)) {
    geometry.config = TwoViewGeometry::ConfigurationType::WATERMARK;
  }

  if (options.compute_relative_pose) {
    EstimateTwoViewGeometryPose(camera1, points1, camera2, points2, &geometry);
  }

  return geometry;
}

TwoViewGeometry EstimateMultipleTwoViewGeometries(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches,
    const TwoViewGeometryOptions& options) {
  FeatureMatches remaining_matches = matches;
  TwoViewGeometry multi_geometry;
  std::vector<TwoViewGeometry> geometries;
  TwoViewGeometryOptions options_copy = options;
  // Set to false to prevent recursive calls to this function.
  options_copy.multiple_models = false;
  while (true) {
    TwoViewGeometry geometry = EstimateTwoViewGeometry(
        camera1, points1, camera2, points2, remaining_matches, options_copy);
    if (geometry.config == TwoViewGeometry::ConfigurationType::DEGENERATE) {
      break;
    }

    if (options.multiple_ignore_watermark) {
      if (geometry.config != TwoViewGeometry::ConfigurationType::WATERMARK) {
        geometries.push_back(geometry);
      }
    } else {
      geometries.push_back(geometry);
    }

    remaining_matches =
        ExtractOutlierMatches(remaining_matches, geometry.inlier_matches);
  }

  if (geometries.empty()) {
    multi_geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
  } else if (geometries.size() == 1) {
    multi_geometry = geometries[0];
  } else {
    multi_geometry.config = TwoViewGeometry::ConfigurationType::MULTIPLE;
    for (const auto& geometry : geometries) {
      multi_geometry.inlier_matches.insert(multi_geometry.inlier_matches.end(),
                                           geometry.inlier_matches.begin(),
                                           geometry.inlier_matches.end());
    }
  }

  return multi_geometry;
}

}  // namespace

bool TwoViewGeometryOptions::Check() const {
  CHECK_OPTION_GE(min_num_inliers, 0);
  CHECK_OPTION_GE(min_E_F_inlier_ratio, 0);
  CHECK_OPTION_LE(min_E_F_inlier_ratio, 1);
  CHECK_OPTION_GE(max_H_inlier_ratio, 0);
  CHECK_OPTION_LE(max_H_inlier_ratio, 1);
  CHECK_OPTION_GE(watermark_min_inlier_ratio, 0);
  CHECK_OPTION_LE(watermark_min_inlier_ratio, 1);
  CHECK_OPTION_GE(watermark_border_size, 0);
  CHECK_OPTION_LE(watermark_border_size, 1);
  CHECK_OPTION_GT(ransac_options.max_error, 0);
  CHECK_OPTION_GE(ransac_options.min_inlier_ratio, 0);
  CHECK_OPTION_LE(ransac_options.min_inlier_ratio, 1);
  CHECK_OPTION_GE(ransac_options.confidence, 0);
  CHECK_OPTION_LE(ransac_options.confidence, 1);
  CHECK_OPTION_LE(ransac_options.min_num_trials, ransac_options.max_num_trials);
  return true;
}

TwoViewGeometry EstimateTwoViewGeometry(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches,
    const TwoViewGeometryOptions& options) {
  if (options.multiple_models) {
    return EstimateMultipleTwoViewGeometries(
        camera1, points1, camera2, points2, matches, options);
  } else if (options.force_H_use) {
    return EstimateCalibratedHomography(
        camera1, points1, camera2, points2, matches, options);
  } else if (camera1.has_prior_focal_length && camera2.has_prior_focal_length) {
    return EstimateCalibratedTwoViewGeometry(
        camera1, points1, camera2, points2, matches, options);
  } else {
    return EstimateUncalibratedTwoViewGeometry(
        camera1, points1, camera2, points2, matches, options);
  }
}

bool EstimateTwoViewGeometryPose(const Camera& camera1,
                                 const std::vector<Eigen::Vector2d>& points1,
                                 const Camera& camera2,
                                 const std::vector<Eigen::Vector2d>& points2,
                                 TwoViewGeometry* geometry) {
  // We need a valid epopolar geometry to estimate the relative pose.
  if (geometry->config != TwoViewGeometry::ConfigurationType::CALIBRATED &&
      geometry->config != TwoViewGeometry::ConfigurationType::UNCALIBRATED &&
      geometry->config != TwoViewGeometry::ConfigurationType::PLANAR &&
      geometry->config != TwoViewGeometry::ConfigurationType::PANORAMIC &&
      geometry->config !=
          TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC) {
    return false;
  }

  // Extract normalized inlier points.
  const size_t num_inlier_matches = geometry->inlier_matches.size();
  if (num_inlier_matches == 0) {
    return false;
  }

  std::vector<Eigen::Vector3d> inlier_cam_rays1(num_inlier_matches);
  std::vector<Eigen::Vector3d> inlier_cam_rays2(num_inlier_matches);
  for (size_t i = 0; i < num_inlier_matches; ++i) {
    const FeatureMatch& match = geometry->inlier_matches[i];
    if (const std::optional<Eigen::Vector2d> cam_point1 =
            camera1.CamFromImg(points1[match.point2D_idx1]);
        cam_point1) {
      inlier_cam_rays1[i] = cam_point1->homogeneous().normalized();
    } else {
      inlier_cam_rays1[i].setZero();
    }
    if (const std::optional<Eigen::Vector2d> cam_point2 =
            camera2.CamFromImg(points2[match.point2D_idx2]);
        cam_point2) {
      inlier_cam_rays2[i] = cam_point2->homogeneous().normalized();
    } else {
      inlier_cam_rays2[i].setZero();
    }
  }

  std::vector<Eigen::Vector3d> points3D;

  if (geometry->config == TwoViewGeometry::ConfigurationType::CALIBRATED) {
    PoseFromEssentialMatrix(geometry->E,
                            inlier_cam_rays1,
                            inlier_cam_rays2,
                            &geometry->cam2_from_cam1,
                            &points3D);
    if (points3D.empty()) {
      return false;
    }
  } else if (geometry->config ==
             TwoViewGeometry::ConfigurationType::UNCALIBRATED) {
    const Eigen::Matrix3d E = EssentialFromFundamentalMatrix(
        camera2.CalibrationMatrix(), geometry->F, camera1.CalibrationMatrix());
    PoseFromEssentialMatrix(E,
                            inlier_cam_rays1,
                            inlier_cam_rays2,
                            &geometry->cam2_from_cam1,
                            &points3D);
    if (points3D.empty()) {
      return false;
    }
  } else if (geometry->config == TwoViewGeometry::ConfigurationType::PLANAR ||
             geometry->config ==
                 TwoViewGeometry::ConfigurationType::PANORAMIC ||
             geometry->config ==
                 TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC) {
    Eigen::Vector3d normal;
    PoseFromHomographyMatrix(geometry->H,
                             camera1.CalibrationMatrix(),
                             camera2.CalibrationMatrix(),
                             inlier_cam_rays1,
                             inlier_cam_rays2,
                             &geometry->cam2_from_cam1,
                             &normal,
                             &points3D);
    if (geometry->config ==
        TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC) {
      if (geometry->cam2_from_cam1.translation.squaredNorm() < 1e-12) {
        geometry->config = TwoViewGeometry::ConfigurationType::PANORAMIC;
      } else {
        geometry->config = TwoViewGeometry::ConfigurationType::PLANAR;
      }
    }

    if (geometry->config == TwoViewGeometry::ConfigurationType::PANORAMIC) {
      geometry->tri_angle = 0;
    }

    if (geometry->config == TwoViewGeometry::ConfigurationType::PLANAR &&
        points3D.empty()) {
      return false;
    }
  } else {
    return false;
  }

  if (!points3D.empty()) {
    const Eigen::Vector3d proj_center1 = Eigen::Vector3d::Zero();
    const Eigen::Vector3d proj_center2 =
        geometry->cam2_from_cam1.rotation.inverse() *
        -geometry->cam2_from_cam1.translation;
    geometry->tri_angle = Median(
        CalculateTriangulationAngles(proj_center1, proj_center2, points3D));
  }

  return true;
}

TwoViewGeometry EstimateCalibratedTwoViewGeometry(
    const Camera& camera1,
    const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2,
    const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches,
    const TwoViewGeometryOptions& options) {
  THROW_CHECK(options.Check());

  TwoViewGeometry geometry;

  const size_t min_num_inliers = static_cast<size_t>(options.min_num_inliers);
  if (matches.size() < min_num_inliers) {
    geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
    return geometry;
  }

  // Extract corresponding points.
  std::vector<Eigen::Vector2d> matched_img_points1(matches.size());
  std::vector<Eigen::Vector2d> matched_img_points2(matches.size());
  std::vector<Eigen::Vector3d> matched_cam_rays1(matches.size());
  std::vector<Eigen::Vector3d> matched_cam_rays2(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    const point2D_t idx1 = matches[i].point2D_idx1;
    const point2D_t idx2 = matches[i].point2D_idx2;
    matched_img_points1[i] = points1[idx1];
    matched_img_points2[i] = points2[idx2];
    if (const std::optional<Eigen::Vector2d> cam_point1 =
            camera1.CamFromImg(points1[idx1]);
        cam_point1) {
      matched_cam_rays1[i] = cam_point1->homogeneous().normalized();
    } else {
      matched_cam_rays1[i].setZero();
    }
    if (const std::optional<Eigen::Vector2d> cam_point2 =
            camera2.CamFromImg(points2[idx2]);
        cam_point2) {
      matched_cam_rays2[i] = cam_point2->homogeneous().normalized();
    } else {
      matched_cam_rays2[i].setZero();
    }
  }

  // Estimate epipolar models.

  auto E_ransac_options = options.ransac_options;
  E_ransac_options.max_error =
      (camera1.CamFromImgThreshold(options.ransac_options.max_error) +
       camera2.CamFromImgThreshold(options.ransac_options.max_error)) /
      2;

  LORANSAC<EssentialMatrixFivePointEstimator, EssentialMatrixFivePointEstimator>
      E_ransac(E_ransac_options);
  const auto E_report = E_ransac.Estimate(matched_cam_rays1, matched_cam_rays2);
  geometry.E = E_report.model;

  LORANSAC<FundamentalMatrixSevenPointEstimator,
           FundamentalMatrixEightPointEstimator>
      F_ransac(options.ransac_options);
  const auto F_report =
      F_ransac.Estimate(matched_img_points1, matched_img_points2);
  geometry.F = F_report.model;

  // Estimate planar or panoramic model.

  LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(
      options.ransac_options);
  const auto H_report =
      H_ransac.Estimate(matched_img_points1, matched_img_points2);
  geometry.H = H_report.model;

  if ((!E_report.success && !F_report.success && !H_report.success) ||
      (E_report.support.num_inliers < min_num_inliers &&
       F_report.support.num_inliers < min_num_inliers &&
       H_report.support.num_inliers < min_num_inliers)) {
    geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
    return geometry;
  }

  // Determine inlier ratios of different models.

  const double E_F_inlier_ratio =
      static_cast<double>(E_report.support.num_inliers) /
      F_report.support.num_inliers;
  const double H_F_inlier_ratio =
      static_cast<double>(H_report.support.num_inliers) /
      F_report.support.num_inliers;
  const double H_E_inlier_ratio =
      static_cast<double>(H_report.support.num_inliers) /
      E_report.support.num_inliers;

  const std::vector<char>* best_inlier_mask = nullptr;
  size_t num_inliers = 0;

  if (E_report.success && E_F_inlier_ratio > options.min_E_F_inlier_ratio &&
      E_report.support.num_inliers >= min_num_inliers) {
    // Calibrated configuration.

    // Always use the model with maximum matches.
    if (E_report.support.num_inliers >= F_report.support.num_inliers) {
      num_inliers = E_report.support.num_inliers;
      best_inlier_mask = &E_report.inlier_mask;
    } else {
      num_inliers = F_report.support.num_inliers;
      best_inlier_mask = &F_report.inlier_mask;
    }

    if (H_E_inlier_ratio > options.max_H_inlier_ratio) {
      geometry.config = TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
      if (H_report.support.num_inliers > num_inliers) {
        num_inliers = H_report.support.num_inliers;
        best_inlier_mask = &H_report.inlier_mask;
      }
    } else {
      geometry.config = TwoViewGeometry::ConfigurationType::CALIBRATED;
    }
  } else if (F_report.success &&
             F_report.support.num_inliers >= min_num_inliers) {
    // Uncalibrated configuration.

    num_inliers = F_report.support.num_inliers;
    best_inlier_mask = &F_report.inlier_mask;

    if (H_F_inlier_ratio > options.max_H_inlier_ratio) {
      geometry.config = TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
      if (H_report.support.num_inliers > num_inliers) {
        num_inliers = H_report.support.num_inliers;
        best_inlier_mask = &H_report.inlier_mask;
      }
    } else {
      geometry.config = TwoViewGeometry::ConfigurationType::UNCALIBRATED;
    }
  } else if (H_report.success &&
             H_report.support.num_inliers >= min_num_inliers) {
    num_inliers = H_report.support.num_inliers;
    best_inlier_mask = &H_report.inlier_mask;
    geometry.config = TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
  } else {
    geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
    return geometry;
  }

  if (best_inlier_mask != nullptr) {
    geometry.inlier_matches =
        ExtractInlierMatches(matches, num_inliers, *best_inlier_mask);

    if (options.detect_watermark && DetectWatermark(camera1,
                                                    matched_img_points1,
                                                    camera2,
                                                    matched_img_points2,
                                                    num_inliers,
                                                    *best_inlier_mask,
                                                    options)) {
      geometry.config = TwoViewGeometry::ConfigurationType::WATERMARK;
    }

    if (options.compute_relative_pose) {
      EstimateTwoViewGeometryPose(
          camera1, points1, camera2, points2, &geometry);
    }
  }

  return geometry;
}

bool DetectWatermark(const Camera& camera1,
                     const std::vector<Eigen::Vector2d>& points1,
                     const Camera& camera2,
                     const std::vector<Eigen::Vector2d>& points2,
                     const size_t num_inliers,
                     const std::vector<char>& inlier_mask,
                     const TwoViewGeometryOptions& options) {
  THROW_CHECK(options.Check());

  // Check if inlier points in border region and extract inlier matches.

  const double diagonal1 = std::sqrt(camera1.width * camera1.width +
                                     camera1.height * camera1.height);
  const double diagonal2 = std::sqrt(camera2.width * camera2.width +
                                     camera2.height * camera2.height);
  const double minx1 = options.watermark_border_size * diagonal1;
  const double miny1 = minx1;
  const double maxx1 = camera1.width - minx1;
  const double maxy1 = camera1.height - miny1;
  const double minx2 = options.watermark_border_size * diagonal2;
  const double miny2 = minx2;
  const double maxx2 = camera2.width - minx2;
  const double maxy2 = camera2.height - miny2;

  std::vector<Eigen::Vector2d> inlier_points1(num_inliers);
  std::vector<Eigen::Vector2d> inlier_points2(num_inliers);

  size_t num_matches_in_border = 0;

  size_t j = 0;
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const auto& point1 = points1[i];
      const auto& point2 = points2[i];

      inlier_points1[j] = point1;
      inlier_points2[j] = point2;
      j += 1;

      if (!IsImagePointInBoundingBox(point1, minx1, maxx1, miny1, maxy1) &&
          !IsImagePointInBoundingBox(point2, minx2, maxx2, miny2, maxy2)) {
        num_matches_in_border += 1;
      }
    }
  }

  const double matches_in_border_ratio =
      static_cast<double>(num_matches_in_border) / num_inliers;

  if (matches_in_border_ratio < options.watermark_min_inlier_ratio) {
    return false;
  }

  // Check if matches follow a translational model.

  RANSACOptions ransac_options = options.ransac_options;
  ransac_options.min_inlier_ratio = options.watermark_min_inlier_ratio;

  LORANSAC<TranslationTransformEstimator<2>, TranslationTransformEstimator<2>>
      ransac(ransac_options);
  const auto report = ransac.Estimate(inlier_points1, inlier_points2);

  const double inlier_ratio =
      static_cast<double>(report.support.num_inliers) / num_inliers;

  return inlier_ratio >= options.watermark_min_inlier_ratio;
}

}  // namespace colmap
