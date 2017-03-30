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

#include "estimators/two_view_geometry.h"

#include "base/camera.h"
#include "base/essential_matrix.h"
#include "base/homography_matrix.h"
#include "base/pose.h"
#include "base/projection.h"
#include "base/triangulation.h"
#include "estimators/essential_matrix.h"
#include "estimators/fundamental_matrix.h"
#include "estimators/homography_matrix.h"
#include "estimators/translation_transform.h"
#include "optim/loransac.h"
#include "optim/ransac.h"
#include "util/random.h"

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
                                     const size_t num_inliers,
                                     const std::vector<char>& inlier_mask) {
  FeatureMatches outlier_matches(matches.size() - num_inliers);
  size_t j = 0;
  for (size_t i = 0; i < matches.size(); ++i) {
    if (!inlier_mask[i]) {
      outlier_matches[j] = matches[i];
      j += 1;
    }
  }
  return outlier_matches;
}

inline bool IsImagePointInBoundingBox(const Eigen::Vector2d& point,
                                      const double minx, const double maxx,
                                      const double miny, const double maxy) {
  return point(0) >= minx && point(0) <= maxx && point(1) >= miny &&
         point(1) <= maxy;
}

}  // namespace

void TwoViewGeometry::Estimate(const Camera& camera1,
                               const std::vector<Eigen::Vector2d>& points1,
                               const Camera& camera2,
                               const std::vector<Eigen::Vector2d>& points2,
                               const FeatureMatches& matches,
                               const Options& options) {
  if (camera1.HasPriorFocalLength() && camera2.HasPriorFocalLength()) {
    EstimateCalibrated(camera1, points1, camera2, points2, matches, options);
  } else {
    EstimateUncalibrated(camera1, points1, camera2, points2, matches, options);
  }
}

void TwoViewGeometry::EstimateMultiple(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches, const Options& options) {
  FeatureMatches remaining_matches = matches;
  std::vector<TwoViewGeometry> two_view_geometries;
  while (true) {
    TwoViewGeometry two_view_geometry;
    two_view_geometry.Estimate(camera1, points1, camera2, points2,
                               remaining_matches, options);
    if (two_view_geometry.config == ConfigurationType::DEGENERATE) {
      break;
    }

    if (options.multiple_ignore_watermark) {
      if (two_view_geometry.config != ConfigurationType::WATERMARK) {
        two_view_geometries.push_back(two_view_geometry);
      }
    } else {
      two_view_geometries.push_back(two_view_geometry);
    }

    remaining_matches = ExtractOutlierMatches(
        remaining_matches, two_view_geometry.inlier_matches.size(),
        two_view_geometry.inlier_mask);
  }

  if (two_view_geometries.empty()) {
    config = ConfigurationType::DEGENERATE;
  } else if (two_view_geometries.size() == 1) {
    *this = two_view_geometries[0];
  } else {
    config = ConfigurationType::MULTIPLE;

    for (const auto& two_view_geometry : two_view_geometries) {
      inlier_matches.insert(inlier_matches.end(),
                            two_view_geometry.inlier_matches.begin(),
                            two_view_geometry.inlier_matches.end());
    }
  }
}

void TwoViewGeometry::EstimateWithRelativePose(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches, const Options& options) {
  // Warning: Do not change this call to another `Estimate*` method, since E is
  // need further down in this method.
  EstimateCalibrated(camera1, points1, camera2, points2, matches, options);

  // Extract normalized inlier points.
  std::vector<Eigen::Vector2d> inlier_points1_N;
  inlier_points1_N.reserve(inlier_matches.size());
  std::vector<Eigen::Vector2d> inlier_points2_N;
  inlier_points2_N.reserve(inlier_matches.size());
  for (const auto& match : inlier_matches) {
    const point2D_t idx1 = match.point2D_idx1;
    const point2D_t idx2 = match.point2D_idx2;
    inlier_points1_N.push_back(camera1.ImageToWorld(points1[idx1]));
    inlier_points2_N.push_back(camera2.ImageToWorld(points2[idx2]));
  }

  Eigen::Matrix3d R;
  std::vector<Eigen::Vector3d> points3D;

  if (config == CALIBRATED || config == UNCALIBRATED) {
    // Try to recover relative pose for calibrated and uncalibrated
    // configurations. In the uncalibrated case, this most likely leads to a
    // ill-defined reconstruction, but sometimes it succeeds anyways after e.g.
    // subsequent bundle-adjustment etc.
    PoseFromEssentialMatrix(E, inlier_points1_N, inlier_points2_N, &R, &tvec,
                            &points3D);
  } else {
    Eigen::Vector3d n;
    PoseFromHomographyMatrix(H, camera1.CalibrationMatrix(),
                             camera2.CalibrationMatrix(), inlier_points1_N,
                             inlier_points2_N, &R, &tvec, &n, &points3D);
  }

  qvec = RotationMatrixToQuaternion(R);

  // Determine triangulation angle.
  const Eigen::Matrix3x4d proj_matrix1 = Eigen::Matrix3x4d::Identity();
  const Eigen::Matrix3x4d proj_matrix2 = ComposeProjectionMatrix(R, tvec);

  if (points3D.empty()) {
    tri_angle = 0;
  } else {
    tri_angle = Median(
        CalculateTriangulationAngles(proj_matrix1, proj_matrix2, points3D));
  }

  if (config == PLANAR_OR_PANORAMIC) {
    if (tvec.norm() == 0) {
      config = PANORAMIC;
      tri_angle = 0;
    } else {
      config = PLANAR;
    }
  }
}

void TwoViewGeometry::EstimateCalibrated(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches, const Options& options) {
  options.Check();

  if (matches.size() < options.min_num_inliers) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // Extract corresponding points.
  std::vector<Eigen::Vector2d> matched_points1(matches.size());
  std::vector<Eigen::Vector2d> matched_points2(matches.size());
  std::vector<Eigen::Vector2d> matched_points1_N(matches.size());
  std::vector<Eigen::Vector2d> matched_points2_N(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    const point2D_t idx1 = matches[i].point2D_idx1;
    const point2D_t idx2 = matches[i].point2D_idx2;
    matched_points1[i] = points1[idx1];
    matched_points2[i] = points2[idx2];
    matched_points1_N[i] = camera1.ImageToWorld(points1[idx1]);
    matched_points2_N[i] = camera2.ImageToWorld(points2[idx2]);
  }

  // Estimate epipolar models.

  auto E_ransac_options = options.ransac_options;
  E_ransac_options.max_error =
      (camera1.ImageToWorldThreshold(options.ransac_options.max_error) +
       camera2.ImageToWorldThreshold(options.ransac_options.max_error)) /
      2;

  LORANSAC<EssentialMatrixFivePointEstimator, EssentialMatrixFivePointEstimator>
      E_ransac(E_ransac_options);
  const auto E_report = E_ransac.Estimate(matched_points1_N, matched_points2_N);
  E = E_report.model;
  E_num_inliers = E_report.support.num_inliers;

  LORANSAC<FundamentalMatrixSevenPointEstimator,
           FundamentalMatrixEightPointEstimator>
      F_ransac(options.ransac_options);
  const auto F_report = F_ransac.Estimate(matched_points1, matched_points2);
  F = F_report.model;
  F_num_inliers = F_report.support.num_inliers;

  // Estimate planar or panoramic model.

  LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(
      options.ransac_options);
  const auto H_report = H_ransac.Estimate(matched_points1, matched_points2);
  H = H_report.model;
  H_num_inliers = H_report.support.num_inliers;

  if ((!E_report.success && !F_report.success && !H_report.success) ||
      (E_num_inliers < options.min_num_inliers &&
       F_num_inliers < options.min_num_inliers &&
       H_num_inliers < options.min_num_inliers)) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // Determine inlier ratios of different models.

  const double E_F_inlier_ratio =
      static_cast<double>(E_num_inliers) / F_num_inliers;
  const double H_F_inlier_ratio =
      static_cast<double>(H_num_inliers) / F_num_inliers;
  const double H_E_inlier_ratio =
      static_cast<double>(H_num_inliers) / E_num_inliers;

  const std::vector<char>* best_inlier_mask = nullptr;
  size_t num_inliers = 0;

  if (E_report.success && E_F_inlier_ratio > options.min_E_F_inlier_ratio &&
      E_num_inliers >= options.min_num_inliers) {
    // Calibrated configuration.

    // Always use the model with maximum matches.
    if (E_num_inliers >= F_num_inliers) {
      num_inliers = E_num_inliers;
      best_inlier_mask = &E_report.inlier_mask;
    } else {
      num_inliers = F_num_inliers;
      best_inlier_mask = &F_report.inlier_mask;
    }

    if (H_E_inlier_ratio > options.max_H_inlier_ratio) {
      config = PLANAR_OR_PANORAMIC;
      if (H_num_inliers > num_inliers) {
        num_inliers = H_num_inliers;
        best_inlier_mask = &H_report.inlier_mask;
      }
    } else {
      config = ConfigurationType::CALIBRATED;
    }
  } else if (F_report.success && F_num_inliers >= options.min_num_inliers) {
    // Uncalibrated configuration.

    num_inliers = F_num_inliers;
    best_inlier_mask = &F_report.inlier_mask;

    if (H_F_inlier_ratio > options.max_H_inlier_ratio) {
      config = ConfigurationType::PLANAR_OR_PANORAMIC;
      if (H_num_inliers > num_inliers) {
        num_inliers = H_num_inliers;
        best_inlier_mask = &H_report.inlier_mask;
      }
    } else {
      config = ConfigurationType::UNCALIBRATED;
    }
  } else if (H_report.success && H_num_inliers >= options.min_num_inliers) {
    num_inliers = H_num_inliers;
    best_inlier_mask = &H_report.inlier_mask;
    config = ConfigurationType::PLANAR_OR_PANORAMIC;
  } else {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  if (best_inlier_mask != nullptr) {
    inlier_matches =
        ExtractInlierMatches(matches, num_inliers, *best_inlier_mask);
    inlier_mask = *best_inlier_mask;

    if (options.detect_watermark &&
        DetectWatermark(camera1, matched_points1, camera2, matched_points2,
                        num_inliers, *best_inlier_mask, options)) {
      config = ConfigurationType::WATERMARK;
    }
  }
}

void TwoViewGeometry::EstimateUncalibrated(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches, const Options& options) {
  options.Check();

  if (matches.size() < options.min_num_inliers) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // Extract corresponding points.
  std::vector<Eigen::Vector2d> matched_points1(matches.size());
  std::vector<Eigen::Vector2d> matched_points2(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    matched_points1[i] = points1[matches[i].point2D_idx1];
    matched_points2[i] = points2[matches[i].point2D_idx2];
  }

  // Estimate epipolar model.

  LORANSAC<FundamentalMatrixSevenPointEstimator,
           FundamentalMatrixEightPointEstimator>
      F_ransac(options.ransac_options);
  const auto F_report = F_ransac.Estimate(matched_points1, matched_points2);
  F = F_report.model;
  F_num_inliers = F_report.support.num_inliers;

  // Estimate planar or panoramic model.

  LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(
      options.ransac_options);
  const auto H_report = H_ransac.Estimate(matched_points1, matched_points2);
  H = H_report.model;
  H_num_inliers = H_report.support.num_inliers;

  if ((!F_report.success && !H_report.success) ||
      (F_num_inliers < options.min_num_inliers &&
       H_num_inliers < options.min_num_inliers)) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // Determine inlier ratios of different models.

  const double H_F_inlier_ratio =
      static_cast<double>(H_num_inliers) / F_num_inliers;

  if (H_F_inlier_ratio > options.max_H_inlier_ratio) {
    config = ConfigurationType::PLANAR_OR_PANORAMIC;
  } else {
    config = ConfigurationType::UNCALIBRATED;
  }

  inlier_matches =
      ExtractInlierMatches(matches, F_num_inliers, F_report.inlier_mask);
  inlier_mask = F_report.inlier_mask;

  if (options.detect_watermark &&
      DetectWatermark(camera1, matched_points1, camera2, matched_points2,
                      F_num_inliers, F_report.inlier_mask, options)) {
    config = ConfigurationType::WATERMARK;
  }
}

bool TwoViewGeometry::DetectWatermark(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const size_t num_inliers, const std::vector<char>& inlier_mask,
    const Options& options) {
  options.Check();

  // Check if inlier points in border region and extract inlier matches.

  const double diagonal1 = std::sqrt(camera1.Width() * camera1.Width() +
                                     camera1.Height() * camera1.Height());
  const double diagonal2 = std::sqrt(camera2.Width() * camera2.Width() +
                                     camera2.Height() * camera2.Height());
  const double minx1 = options.watermark_border_size * diagonal1;
  const double miny1 = minx1;
  const double maxx1 = camera1.Width() - minx1;
  const double maxy1 = camera1.Height() - miny1;
  const double minx2 = options.watermark_border_size * diagonal2;
  const double miny2 = minx2;
  const double maxx2 = camera2.Width() - minx2;
  const double maxy2 = camera2.Height() - miny2;

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
