// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/estimators/coordinate_frame.h"

#include "colmap/base/gps.h"
#include "colmap/base/line.h"
#include "colmap/base/pose.h"
#include "colmap/base/undistortion.h"
#include "colmap/estimators/utils.h"
#include "colmap/optim/ransac.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

namespace colmap {
namespace {

struct VanishingPointEstimator {
  // The line segments.
  typedef LineSegment X_t;
  // The line representation of the segments.
  typedef Eigen::Vector3d Y_t;
  // The vanishing point.
  typedef Eigen::Vector3d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 2;

  // Estimate the vanishing point from at least two line segments.
  static std::vector<M_t> Estimate(const std::vector<X_t>& line_segments,
                                   const std::vector<Y_t>& lines) {
    CHECK_EQ(line_segments.size(), 2);
    CHECK_EQ(lines.size(), 2);
    return {lines[0].cross(lines[1])};
  }

  // Calculate the squared distance of each line segment's end point to the line
  // connecting the vanishing point and the midpoint of the line segment.
  static void Residuals(const std::vector<X_t>& line_segments,
                        const std::vector<Y_t>& lines,
                        const M_t& vanishing_point,
                        std::vector<double>* residuals) {
    residuals->resize(line_segments.size());

    // Check if vanishing point is at infinity.
    if (vanishing_point[2] == 0) {
      std::fill(residuals->begin(),
                residuals->end(),
                std::numeric_limits<double>::max());
      return;
    }

    for (size_t i = 0; i < lines.size(); ++i) {
      const Eigen::Vector3d midpoint =
          (0.5 * (line_segments[i].start + line_segments[i].end)).homogeneous();
      const Eigen::Vector3d connecting_line = midpoint.cross(vanishing_point);
      const double signed_distance =
          connecting_line.dot(line_segments[i].end.homogeneous()) /
          connecting_line.head<2>().norm();
      (*residuals)[i] = signed_distance * signed_distance;
    }
  }
};

Eigen::Vector3d FindBestConsensusAxis(const std::vector<Eigen::Vector3d>& axes,
                                      const double max_distance) {
  if (axes.empty()) {
    return Eigen::Vector3d::Zero();
  }

  std::vector<int> inlier_idxs;
  inlier_idxs.reserve(axes.size());

  std::vector<int> best_inlier_idxs;
  best_inlier_idxs.reserve(axes.size());

  double best_inlier_distance_sum = std::numeric_limits<double>::max();

  for (size_t i = 0; i < axes.size(); ++i) {
    const Eigen::Vector3d ref_axis = axes[i];
    double inlier_distance_sum = 0;
    inlier_idxs.clear();
    for (size_t j = 0; j < axes.size(); ++j) {
      if (i == j) {
        inlier_idxs.push_back(j);
      } else {
        const double distance = 1 - ref_axis.dot(axes[j]);
        if (distance <= max_distance) {
          inlier_distance_sum += distance;
          inlier_idxs.push_back(j);
        }
      }
    }

    if (inlier_idxs.size() > best_inlier_idxs.size() ||
        (inlier_idxs.size() == best_inlier_idxs.size() &&
         inlier_distance_sum < best_inlier_distance_sum)) {
      best_inlier_distance_sum = inlier_distance_sum;
      best_inlier_idxs = inlier_idxs;
    }
  }

  if (best_inlier_idxs.empty()) {
    return Eigen::Vector3d::Zero();
  }

  Eigen::Vector3d best_axis(0, 0, 0);
  for (const auto idx : best_inlier_idxs) {
    best_axis += axes[idx];
  }
  best_axis /= best_inlier_idxs.size();

  return best_axis;
}

}  // namespace

Eigen::Vector3d EstimateGravityVectorFromImageOrientation(
    const Reconstruction& reconstruction, const double max_axis_distance) {
  std::vector<Eigen::Vector3d> downward_axes;
  downward_axes.reserve(reconstruction.NumRegImages());
  for (const auto image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    downward_axes.push_back(image.RotationMatrix().row(1));
  }
  return FindBestConsensusAxis(downward_axes, max_axis_distance);
}

Eigen::Matrix3d EstimateManhattanWorldFrame(
    const ManhattanWorldFrameEstimationOptions& options,
    const Reconstruction& reconstruction,
    const std::string& image_path) {
  std::vector<Eigen::Vector3d> rightward_axes;
  std::vector<Eigen::Vector3d> downward_axes;
  for (size_t i = 0; i < reconstruction.NumRegImages(); ++i) {
    const auto image_id = reconstruction.RegImageIds()[i];
    const auto& image = reconstruction.Image(image_id);
    const auto& camera = reconstruction.Camera(image.CameraId());

    PrintHeading1(StringPrintf("Processing image %s (%d / %d)",
                               image.Name().c_str(),
                               i + 1,
                               reconstruction.NumRegImages()));

    std::cout << "Reading image..." << std::endl;

    colmap::Bitmap bitmap;
    CHECK(bitmap.Read(colmap::JoinPaths(image_path, image.Name())));

    std::cout << "Undistorting image..." << std::endl;

    UndistortCameraOptions undistortion_options;
    undistortion_options.max_image_size = options.max_image_size;

    Bitmap undistorted_bitmap;
    Camera undistorted_camera;
    UndistortImage(undistortion_options,
                   bitmap,
                   camera,
                   &undistorted_bitmap,
                   &undistorted_camera);

    std::cout << "Detecting lines...";

    const std::vector<LineSegment> line_segments =
        DetectLineSegments(undistorted_bitmap, options.min_line_length);
    const std::vector<LineSegmentOrientation> line_orientations =
        ClassifyLineSegmentOrientations(line_segments,
                                        options.line_orientation_tolerance);

    std::cout << StringPrintf(" %d", line_segments.size());

    std::vector<LineSegment> horizontal_line_segments;
    std::vector<LineSegment> vertical_line_segments;
    std::vector<Eigen::Vector3d> horizontal_lines;
    std::vector<Eigen::Vector3d> vertical_lines;
    for (size_t i = 0; i < line_segments.size(); ++i) {
      const auto line_segment = line_segments[i];
      const Eigen::Vector3d line_segment_start =
          line_segment.start.homogeneous();
      const Eigen::Vector3d line_segment_end = line_segment.end.homogeneous();
      const Eigen::Vector3d line = line_segment_start.cross(line_segment_end);
      if (line_orientations[i] == LineSegmentOrientation::HORIZONTAL) {
        horizontal_line_segments.push_back(line_segment);
        horizontal_lines.push_back(line);
      } else if (line_orientations[i] == LineSegmentOrientation::VERTICAL) {
        vertical_line_segments.push_back(line_segment);
        vertical_lines.push_back(line);
      }
    }

    std::cout << StringPrintf(" (%d horizontal, %d vertical)",
                              horizontal_lines.size(),
                              vertical_lines.size())
              << std::endl;

    std::cout << "Estimating vanishing points...";

    RANSACOptions ransac_options;
    ransac_options.max_error = options.max_line_vp_distance;
    RANSAC<VanishingPointEstimator> ransac(ransac_options);
    const auto horizontal_report =
        ransac.Estimate(horizontal_line_segments, horizontal_lines);
    const auto vertical_report =
        ransac.Estimate(vertical_line_segments, vertical_lines);

    std::cout << StringPrintf(" (%d horizontal inliers, %d vertical inliers)",
                              horizontal_report.support.num_inliers,
                              vertical_report.support.num_inliers)
              << std::endl;

    std::cout << "Composing coordinate axes..." << std::endl;

    const Eigen::Matrix3d inv_calib_matrix =
        undistorted_camera.CalibrationMatrix().inverse();
    const Eigen::Vector4d inv_qvec = InvertQuaternion(image.Qvec());

    if (horizontal_report.success) {
      const Eigen::Vector3d horizontal_camera_axis =
          (inv_calib_matrix * horizontal_report.model).normalized();
      Eigen::Vector3d horizontal_axis =
          QuaternionRotatePoint(inv_qvec, horizontal_camera_axis).normalized();
      // Make sure all axes point into the same direction.
      if (rightward_axes.size() > 0 &&
          rightward_axes[0].dot(horizontal_axis) < 0) {
        horizontal_axis = -horizontal_axis;
      }
      rightward_axes.push_back(horizontal_axis);
      std::cout << "  Horizontal: " << horizontal_axis.transpose() << std::endl;
    }

    if (vertical_report.success) {
      const Eigen::Vector3d vertical_camera_axis =
          (inv_calib_matrix * vertical_report.model).normalized();
      Eigen::Vector3d vertical_axis =
          QuaternionRotatePoint(inv_qvec, vertical_camera_axis).normalized();
      // Make sure axis points downwards in the image, assuming that the image
      // was taken in upright orientation.
      if (vertical_camera_axis.dot(Eigen::Vector3d(0, 1, 0)) < 0) {
        vertical_axis = -vertical_axis;
      }
      downward_axes.push_back(vertical_axis);
      std::cout << "  Vertical: " << vertical_axis.transpose() << std::endl;
    }
  }

  PrintHeading1("Computing coordinate frame");

  Eigen::Matrix3d frame = Eigen::Matrix3d::Zero();

  if (rightward_axes.size() > 0) {
    frame.col(0) =
        FindBestConsensusAxis(rightward_axes, options.max_axis_distance);
  }

  std::cout << "Found rightward axis: " << frame.col(0).transpose()
            << std::endl;

  if (downward_axes.size() > 0) {
    frame.col(1) =
        FindBestConsensusAxis(downward_axes, options.max_axis_distance);
  }

  std::cout << "Found downward axis: " << frame.col(1).transpose() << std::endl;

  if (rightward_axes.size() > 0 && downward_axes.size() > 0) {
    frame.col(2) = frame.col(0).cross(frame.col(1));
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        frame, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const Eigen::Matrix3d orthonormal_frame =
        svd.matrixU() * Eigen::Matrix3d::Identity() * svd.matrixV().transpose();
    frame = orthonormal_frame;
  }

  std::cout << "Found orthonormal frame: " << std::endl;
  std::cout << frame << std::endl;

  return frame;
}

void AlignToPrincipalPlane(Reconstruction* recon, SimilarityTransform3* tform) {
  // Perform SVD on the 3D points to estimate the ground plane basis
  const Eigen::Vector3d centroid = recon->ComputeCentroid(0.0, 1.0);
  Eigen::MatrixXd points(3, recon->NumPoints3D());
  int pidx = 0;
  for (const auto& point : recon->Points3D()) {
    points.col(pidx++) = point.second.XYZ() - centroid;
  }
  const Eigen::Matrix3d basis =
      points.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).matrixU();
  Eigen::Matrix3d rot_mat;
  rot_mat << basis.col(0), basis.col(1), basis.col(0).cross(basis.col(1));
  rot_mat.transposeInPlace();

  *tform = SimilarityTransform3(
      1.0, RotationMatrixToQuaternion(rot_mat), -rot_mat * centroid);

  // if camera plane ends up below ground then flip basis vectors and create new
  // transform
  Image test_img = recon->Images().begin()->second;
  tform->TransformPose(&test_img.Qvec(), &test_img.Tvec());
  if (test_img.ProjectionCenter().z() < 0.0) {
    rot_mat << basis.col(0), -basis.col(1), basis.col(0).cross(-basis.col(1));
    rot_mat.transposeInPlace();
    *tform = SimilarityTransform3(
        1.0, RotationMatrixToQuaternion(rot_mat), -rot_mat * centroid);
  }

  recon->Transform(*tform);
}

void AlignToENUPlane(Reconstruction* recon,
                     SimilarityTransform3* tform,
                     bool unscaled) {
  const Eigen::Vector3d centroid = recon->ComputeCentroid(0.0, 1.0);
  GPSTransform gps_tform;
  const Eigen::Vector3d ell_centroid = gps_tform.XYZToEll({centroid}).at(0);

  // Create rotation matrix from ECEF to ENU coordinates
  const double sin_lat = sin(DegToRad(ell_centroid(0)));
  const double sin_lon = sin(DegToRad(ell_centroid(1)));
  const double cos_lat = cos(DegToRad(ell_centroid(0)));
  const double cos_lon = cos(DegToRad(ell_centroid(1)));

  // Create ECEF to ENU rotation matrix
  Eigen::Matrix3d rot_mat;
  rot_mat << -sin_lon, cos_lon, 0, -cos_lon * sin_lat, -sin_lon * sin_lat,
      cos_lat, cos_lon * cos_lat, sin_lon * cos_lat, sin_lat;

  const double scale = unscaled ? 1.0 / tform->Scale() : 1.0;
  *tform = SimilarityTransform3(scale,
                                RotationMatrixToQuaternion(rot_mat),
                                -(scale * rot_mat) * centroid);
  recon->Transform(*tform);
}

}  // namespace colmap
