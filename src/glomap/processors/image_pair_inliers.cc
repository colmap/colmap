#include "glomap/processors/image_pair_inliers.h"

#include "colmap/estimators/utils.h"
#include "colmap/geometry/essential_matrix.h"

namespace glomap {
namespace {

// Cheirality check for essential matrix.
// Code from PoseLib by Viktor Larsson.
bool CheckCheirality(const Rigid3d& pose,
                     const Eigen::Vector3d& x1,
                     const Eigen::Vector3d& x2,
                     double min_depth = 0.,
                     double max_depth = 100.) {
  // This code assumes that x1 and x2 are unit vectors
  const Eigen::Vector3d Rx1 = pose.rotation * x1;

  // [1 a; a 1] * [lambda1; lambda2] = [b1; b2]
  // [lambda1; lambda2] = [1 -a; -a 1] * [b1; b2] / (1 - a*a)
  const double a = -Rx1.dot(x2);
  const double b1 = -Rx1.dot(pose.translation);
  const double b2 = x2.dot(pose.translation);

  // Note that we drop the factor 1.0/(1-a*a) since it is always positive.
  const double lambda1 = b1 - a * b2;
  const double lambda2 = -a * b1 + b2;

  min_depth = min_depth * (1 - a * a);
  max_depth = max_depth * (1 - a * a);

  bool status = lambda1 > min_depth && lambda2 > min_depth;
  status = status && (lambda1 < max_depth) && (lambda2 < max_depth);
  return status;
}

// Get the orientation signum for fundamental matrix.
// For cheirality check of fundamental matrix.
// Code from GC-RANSAC by Daniel Barath.
double GetOrientationSignum(const Eigen::Matrix3d& F,
                            const Eigen::Vector3d& epipole,
                            const Eigen::Vector2d& pt1,
                            const Eigen::Vector2d& pt2) {
  double signum1 = F(0, 0) * pt2[0] + F(1, 0) * pt2[1] + F(2, 0);
  double signum2 = epipole(1) - epipole(2) * pt1[1];
  return signum1 * signum2;
}

}  // namespace

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

  const colmap::Image& image1 = reconstruction.Image(image_id1);
  const colmap::Image& image2 = reconstruction.Image(image_id2);

  const double thres = options.max_epipolar_error_E * 0.5 *
                       (1. / image1.CameraPtr()->MeanFocalLength() +
                        1. / image2.CameraPtr()->MeanFocalLength());

  // Square the threshold for faster computation
  const double sq_threshold = thres * thres;
  double score = 0.;

  // TODO: determine the best threshold for triangulation angle
  double thres_epipole = std::cos(colmap::DegToRad(3.));
  double thres_angle = 1;
  thres_angle += 1e-6;
  thres_epipole += 1e-6;
  for (size_t k = 0; k < image_pair.matches.rows(); ++k) {
    const std::optional<Eigen::Vector2d> cam_point1 =
        image1.CameraPtr()->CamFromImg(
            image1.Point2D(image_pair.matches(k, 0)).xy);
    const std::optional<Eigen::Vector2d> cam_point2 =
        image2.CameraPtr()->CamFromImg(
            image2.Point2D(image_pair.matches(k, 1)).xy);
    if (!cam_point1.has_value() || !cam_point2.has_value()) {
      score += sq_threshold;
      continue;
    }

    const Eigen::Vector3d pt1 = cam_point1->homogeneous().normalized();
    const Eigen::Vector3d pt2 = cam_point2->homogeneous().normalized();

    const double r2 = colmap::ComputeSquaredSampsonError(pt1, pt2, E);

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

  const Image& image1 = reconstruction.Image(image_id1);
  const Image& image2 = reconstruction.Image(image_id2);

  double thres = options.max_epipolar_error_F;
  double sq_threshold = thres * thres;

  double score = 0.;

  std::vector<int> inliers_pre;
  std::vector<double> errors;
  for (size_t k = 0; k < image_pair.matches.rows(); ++k) {
    const Eigen::Vector2d& pt1 = image1.Point2D(image_pair.matches(k, 0)).xy;
    const Eigen::Vector2d& pt2 = image2.Point2D(image_pair.matches(k, 1)).xy;
    const double r2 = colmap::ComputeSquaredSampsonError(
        pt1.homogeneous(), pt2.homogeneous(), image_pair.F);

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

  const Image& image1 = reconstruction.Image(image_id1);
  const Image& image2 = reconstruction.Image(image_id2);

  double thres = options.max_epipolar_error_H;
  double sq_threshold = thres * thres;
  double score = 0.;
  for (size_t k = 0; k < image_pair.matches.rows(); ++k) {
    const Eigen::Vector2d& pt1 = image1.Point2D(image_pair.matches(k, 0)).xy;
    const Eigen::Vector2d& pt2 = image2.Point2D(image_pair.matches(k, 1)).xy;
    const double r2 =
        colmap::ComputeSquaredHomographyError(pt1, pt2, image_pair.H);

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

void ImagePairsInlierCount(ViewGraph& view_graph,
                           const colmap::Reconstruction& reconstruction,
                           const InlierThresholdOptions& options,
                           bool clean_inliers) {
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!clean_inliers && image_pair.inliers.size() > 0) continue;
    image_pair.inliers.clear();

    if (!image_pair.is_valid) continue;
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    ImagePairInliers inlier_finder(
        image_id1, image_id2, image_pair, reconstruction, options);
    inlier_finder.ScoreError();
  }
}

}  // namespace glomap
