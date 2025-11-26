#include "glomap/processors/image_pair_inliers.h"

#include "colmap/geometry/essential_matrix.h"

#include "glomap/math/two_view_geometry.h"

namespace glomap {

double ImagePairInliers::ScoreError() {
  // Count inliers base on the type
  if (image_pair.config == colmap::TwoViewGeometry::PLANAR ||
      image_pair.config == colmap::TwoViewGeometry::PANORAMIC ||
      image_pair.config == colmap::TwoViewGeometry::PLANAR_OR_PANORAMIC)
    return ScoreErrorHomography();
  else if (image_pair.config == colmap::TwoViewGeometry::UNCALIBRATED)
    return ScoreErrorFundamental();
  else if (image_pair.config == colmap::TwoViewGeometry::CALIBRATED)
    return ScoreErrorEssential();
  return 0;
}

double ImagePairInliers::ScoreErrorEssential() {
  const Eigen::Matrix3d E =
      colmap::EssentialMatrixFromPose(image_pair.cam2_from_cam1);

  // eij = camera i on image j
  Eigen::Vector3d epipole12 = image_pair.cam2_from_cam1.translation;
  Eigen::Vector3d epipole21 = Inverse(image_pair.cam2_from_cam1).translation;

  if (epipole12[2] < 0) epipole12 = -epipole12;
  if (epipole21[2] < 0) epipole21 = -epipole21;

  if (image_pair.inliers.size() > 0) {
    image_pair.inliers.clear();
  }

  const image_t image_id1 = image_pair.image_id1;
  const image_t image_id2 = image_pair.image_id2;

  const double thres =
      options.max_epipolar_error_E * 0.5 *
      (1. / cameras->at(images.at(image_id1).camera_id).MeanFocalLength() +
       1. / cameras->at(images.at(image_id2).camera_id).MeanFocalLength());

  // Square the threshold for faster computation
  const double sq_threshold = thres * thres;
  double score = 0.;
  Eigen::Vector3d pt1, pt2;

  // TODO: determine the best threshold for triangulation angle
  // double thres_angle = std::cos(DegToRad(1.));
  double thres_epipole = std::cos(DegToRad(3.));
  double thres_angle = 1;
  thres_angle += 1e-6;
  thres_epipole += 1e-6;
  for (size_t k = 0; k < image_pair.matches.rows(); ++k) {
    // Use the undistorted features
    pt1 = images.at(image_id1).features_undist[image_pair.matches(k, 0)];
    pt2 = images.at(image_id2).features_undist[image_pair.matches(k, 1)];
    const double r2 = SampsonError(E, pt1, pt2);

    if (r2 < sq_threshold) {
      bool cheirality =
          CheckCheirality(image_pair.cam2_from_cam1, pt1, pt2, 1e-2, 100.);

      // Check whether two image rays have small triangulation angle or are too
      // close to the epipoles
      bool not_denegerate = true;

      // Check whether two image rays are too close
      double diff_angle =
          pt1.dot(image_pair.cam2_from_cam1.rotation.inverse() * pt2);
      not_denegerate = (diff_angle < thres_angle);

      // Check whether two points are too close to the epipoles
      double diff_epipole1 = pt1.dot(epipole21);
      double diff_epipole2 = pt2.dot(epipole12);
      not_denegerate = not_denegerate && (diff_epipole1 < thres_epipole &&
                                          diff_epipole2 < thres_epipole);

      if (cheirality && not_denegerate) {
        score += r2;
        image_pair.inliers.push_back(k);
      } else {
        score += sq_threshold;
      }
    } else {
      score += sq_threshold;
    }
  }
  return score;
}

double ImagePairInliers::ScoreErrorFundamental() {
  if (image_pair.inliers.size() > 0) {
    image_pair.inliers.clear();
  }

  Eigen::Vector3d epipole = image_pair.F.row(0).cross(image_pair.F.row(2));

  bool status = false;
  for (auto i = 0; i < 3; i++) {
    if (std::abs(epipole(i)) > 1e-12) {
      status = true;
      break;
    }
  }
  if (!status) {
    epipole = image_pair.F.row(1).cross(image_pair.F.row(2));
  }

  // First, get the orientation signum for every point
  std::vector<double> signums;
  int positive_count = 0;
  int negative_count = 0;

  image_t image_id1 = image_pair.image_id1;
  image_t image_id2 = image_pair.image_id2;

  double thres = options.max_epipolar_error_F;
  double sq_threshold = thres * thres;

  double score = 0.;
  Eigen::Vector2d pt1, pt2;

  std::vector<int> inliers_pre;
  std::vector<double> errors;
  for (size_t k = 0; k < image_pair.matches.rows(); ++k) {
    pt1 = images.at(image_id1).features[image_pair.matches(k, 0)];
    pt2 = images.at(image_id2).features[image_pair.matches(k, 1)];
    const double r2 = SampsonError(image_pair.F, pt1, pt2);

    if (r2 < sq_threshold) {
      signums.push_back(GetOrientationSignum(image_pair.F, epipole, pt1, pt2));
      if (signums.back() > 0) {
        positive_count++;
      } else {
        negative_count++;
      }

      inliers_pre.push_back(k);
      errors.push_back(r2);
    } else {
      score += sq_threshold;
    }
  }
  bool is_positive = (positive_count > negative_count);

  // If cannot distinguish the signum, the pair should be invalid
  if (positive_count == negative_count) return 0;

  // Then, if the signum is not consistent with the cheirality, discard the
  // point
  for (int k = 0; k < inliers_pre.size(); k++) {
    bool cheirality = (signums[k] > 0) == is_positive;
    if (!cheirality) {
      score += sq_threshold;
    } else {
      image_pair.inliers.push_back(inliers_pre[k]);
      score += errors[k];
    }
  }
  return score;
}

double ImagePairInliers::ScoreErrorHomography() {
  if (image_pair.inliers.size() > 0) {
    image_pair.inliers.clear();
  }

  image_t image_id1 = image_pair.image_id1;
  image_t image_id2 = image_pair.image_id2;

  double thres = options.max_epipolar_error_H;
  double sq_threshold = thres * thres;
  double score = 0.;
  Eigen::Vector2d pt1, pt2;
  for (size_t k = 0; k < image_pair.matches.rows(); ++k) {
    pt1 = images.at(image_id1).features[image_pair.matches(k, 0)];
    pt2 = images.at(image_id2).features[image_pair.matches(k, 1)];
    const double r2 = HomographyError(image_pair.H, pt1, pt2);

    if (r2 < sq_threshold) {
      // TODO: cheirality check for homography. Is that a thing?
      bool cheirality = true;

      if (cheirality) {
        score += r2;
        image_pair.inliers.push_back(k);
      } else {
        score += sq_threshold;
      }
    } else {
      score += sq_threshold;
    }
  }
  return score;
}

void ImagePairsInlierCount(
    ViewGraph& view_graph,
    const std::unordered_map<camera_t, colmap::Camera>& cameras,
    const std::unordered_map<image_t, Image>& images,
    const InlierThresholdOptions& options,
    bool clean_inliers) {
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!clean_inliers && image_pair.inliers.size() > 0) continue;
    image_pair.inliers.clear();

    if (image_pair.is_valid == false) continue;
    ImagePairInliers inlier_finder(image_pair, images, options, &cameras);
    inlier_finder.ScoreError();
  }
}

}  // namespace glomap
