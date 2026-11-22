// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/sim3.h"
#include "colmap/image/apriltag.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace colmap {

struct TagDistanceConstraint {
  int tag_id_a = -1;
  int tag_id_b = -1;
  double target_distance = 0.0;
};

struct TagPositionConstraint {
  int tag_id = -1;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
};

struct AprilTagTriangulationOptions {
  // Minimum number of registered views observing a tag center.
  int min_views = 2;

  // Minimum triangulation parallax angle in degrees between at least one pair
  // of observing rays.
  double min_parallax_angle_deg = 1.5;

  // Maximum pixel reprojection error allowed for an observation to remain an
  // inlier during triangulation refinement.
  double max_reprojection_error_px = 4.0;

  bool Check() const;
};

struct AprilTagScaleAlignmentOptions {
  // Minimum number of inlier distance constraints required.
  int min_inliers = 1;

  // Minimum fraction of evaluable distance constraints that must be inliers.
  double min_inlier_ratio = 0.5;

  // Relative distance error tolerance: |s * d - D| <= max_relative_error * D.
  double max_relative_error = 0.05;

  // Absolute distance error tolerance: |s * d - D| <= max_absolute_error.
  // A constraint is an inlier if it satisfies either the relative or the
  // absolute tolerance.
  double max_absolute_error = 0.0;

  bool Check() const;
};

struct AprilTagPositionAlignmentOptions {
  // Minimum number of common tags and inlier tags required for Sim3 estimation.
  int min_common_tags = 3;

  // Minimum fraction of matched tags that must be inliers.
  double min_inlier_ratio = 0.5;

  // RANSAC options used by EstimateSim3dRobust (max_error is in target units).
  RANSACOptions ransac_options;

  AprilTagPositionAlignmentOptions() { ransac_options.max_error = 0.05; }

  bool Check() const;
};

struct AprilTagImageObservation {
  image_t image_id = kInvalidImageId;
  std::string image_name;
  int tag_id = -1;
  // Center in camera pixel coordinates (scaled to match Camera::width/height).
  Eigen::Vector2d center_px = Eigen::Vector2d::Zero();
  double decision_margin = 0.0;
  int hamming = 0;
};

struct TriangulatedAprilTag {
  int tag_id = -1;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  size_t num_observations = 0;
  size_t num_inliers = 0;
  double mean_reprojection_error_px = 0.0;
  double max_reprojection_error_px = 0.0;
  double max_parallax_deg = 0.0;
};

struct RejectedAprilTag {
  int tag_id = -1;
  size_t num_observations = 0;
  std::string reason;
};

struct TagDistanceResidual {
  int tag_id_a = -1;
  int tag_id_b = -1;
  double target_distance = 0.0;
  double measured_distance = 0.0;
  double scaled_distance = 0.0;
  double error = 0.0;
  double relative_error = 0.0;
  bool inlier = false;
};

struct TagPositionResidual {
  int tag_id = -1;
  Eigen::Vector3d source_position = Eigen::Vector3d::Zero();
  Eigen::Vector3d aligned_position = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_position = Eigen::Vector3d::Zero();
  double error = 0.0;
  bool inlier = false;
};

struct AprilTagAlignmentErrorMetrics {
  size_t num_constraints = 0;
  size_t num_evaluable = 0;
  size_t num_inliers = 0;
  double inlier_ratio = 0.0;
  double rmse = 0.0;
  double mae = 0.0;
  double max_error = 0.0;
};

struct AprilTagAlignmentReport {
  bool success = false;
  std::string mode;
  Sim3d tgt_from_src;
  size_t num_images_processed = 0;
  size_t num_images_missing = 0;
  size_t num_raw_detections = 0;
  std::vector<TriangulatedAprilTag> triangulated_tags;
  std::vector<RejectedAprilTag> rejected_tags;
  AprilTagAlignmentErrorMetrics metrics;
  std::vector<TagDistanceResidual> distance_residuals;
  std::vector<TagPositionResidual> position_residuals;
};

// Parse tag distance constraints from a text file. Each non-empty, non-# line
// contains: <tag_id_a> <tag_id_b> <target_distance>.
std::vector<TagDistanceConstraint> ReadTagDistanceConstraints(
    const std::filesystem::path& path);

// Parse 3D tag position constraints from a text file. Each non-empty, non-#
// line contains: <tag_id> <x> <y> <z>.
std::vector<TagPositionConstraint> ReadTagPositionConstraints(
    const std::filesystem::path& path);

// Detect AprilTags across all registered images in the reconstruction using a
// thread pool. Scales pixel coordinates if the on-disk image dimensions differ
// from the camera dimensions, keeps the strongest detection per (image, tag),
// and returns observations deterministically sorted by (tag_id, image_id).
std::vector<AprilTagImageObservation> DetectAprilTagsInReconstruction(
    const Reconstruction& reconstruction,
    const std::filesystem::path& image_path,
    const AprilTagDetectionOptions& options,
    size_t* num_images_processed = nullptr,
    size_t* num_images_missing = nullptr);

// Triangulate 3D centers for all detected AprilTag IDs with sufficient parallax
// and reprojection consensus.
void TriangulateAprilTags(
    const Reconstruction& reconstruction,
    const std::vector<AprilTagImageObservation>& observations,
    const AprilTagTriangulationOptions& options,
    std::vector<TriangulatedAprilTag>* triangulated_tags,
    std::vector<RejectedAprilTag>* rejected_tags = nullptr);

// Estimate scale from triangulated tag positions and known pairwise distances
// using exact 1D interval sweep-line consensus followed by 2-pass weighted
// least squares refinement. Does not mutate any reconstruction.
bool EstimateAprilTagScale(
    const std::vector<TriangulatedAprilTag>& triangulated_tags,
    const std::vector<TagDistanceConstraint>& constraints,
    const AprilTagScaleAlignmentOptions& options,
    Sim3d* tgt_from_src,
    AprilTagAlignmentErrorMetrics* metrics = nullptr,
    std::vector<TagDistanceResidual>* residuals = nullptr);

// Estimate Sim3d alignment from triangulated tag positions and known 3D tag
// coordinates using EstimateSim3dRobust. Does not mutate any reconstruction.
bool EstimateAprilTagPositionAlignment(
    const std::vector<TriangulatedAprilTag>& triangulated_tags,
    const std::vector<TagPositionConstraint>& constraints,
    const AprilTagPositionAlignmentOptions& options,
    Sim3d* tgt_from_src,
    AprilTagAlignmentErrorMetrics* metrics = nullptr,
    std::vector<TagPositionResidual>* residuals = nullptr);

// End-to-end AprilTag detection, triangulation, scale estimation, and
// reconstruction transformation from pairwise tag distances.
bool AlignReconstructionWithAprilTagDistances(
    const std::filesystem::path& image_path,
    const std::vector<TagDistanceConstraint>& constraints,
    const AprilTagDetectionOptions& detection_options,
    const AprilTagTriangulationOptions& triangulation_options,
    const AprilTagScaleAlignmentOptions& scale_options,
    Reconstruction* reconstruction,
    AprilTagAlignmentReport* report = nullptr);

// End-to-end AprilTag detection, triangulation, Sim3d estimation, and
// reconstruction transformation from known 3D tag coordinates.
bool AlignReconstructionWithAprilTagPositions(
    const std::filesystem::path& image_path,
    const std::vector<TagPositionConstraint>& constraints,
    const AprilTagDetectionOptions& detection_options,
    const AprilTagTriangulationOptions& triangulation_options,
    const AprilTagPositionAlignmentOptions& position_options,
    Reconstruction* reconstruction,
    AprilTagAlignmentReport* report = nullptr);

// Write a JSON alignment report to disk.
void WriteAprilTagAlignmentReportJson(const std::filesystem::path& path,
                                      const AprilTagAlignmentReport& report);

}  // namespace colmap
