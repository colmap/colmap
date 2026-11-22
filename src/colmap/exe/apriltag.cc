// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/apriltag.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/estimators/apriltag_alignment.h"
#include "colmap/image/apriltag.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/string.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace colmap {

int RunAprilTagAligner(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::filesystem::path distances_path;
  std::filesystem::path positions_path;
  std::filesystem::path transform_path;
  std::filesystem::path report_path;

  AprilTagDetectionOptions detection_options;
  AprilTagTriangulationOptions triangulation_options;
  AprilTagScaleAlignmentOptions scale_options;
  AprilTagPositionAlignmentOptions position_options;
  std::string family_name(AprilTagFamilyToString(detection_options.family));

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption(
      "distances_path",
      &distances_path,
      "Path to text file with tag distance constraints: "
      "<tag_id_a> <tag_id_b> <target_distance>");
  options.AddDefaultOption(
      "positions_path",
      &positions_path,
      "Path to text file with known 3D tag positions: "
      "<tag_id> <x> <y> <z>");
  options.AddDefaultOption(
      "transform_path",
      &transform_path,
      "Optional path to save the estimated Sim3 transformation");
  options.AddDefaultOption(
      "report_path",
      &report_path,
      "Optional path to save a JSON alignment report");

  options.AddDefaultOption(
      "AprilTag.family",
      &family_name,
      "{" + VectorToCSV(AprilTagFamilyStrings()) + "}");
  options.AddDefaultOption("AprilTag.quad_decimate",
                           &detection_options.quad_decimate);
  options.AddDefaultOption("AprilTag.quad_sigma",
                           &detection_options.quad_sigma);
  options.AddDefaultOption("AprilTag.refine_edges",
                           &detection_options.refine_edges);
  options.AddDefaultOption("AprilTag.decode_sharpening",
                           &detection_options.decode_sharpening);
  options.AddDefaultOption("AprilTag.max_hamming",
                           &detection_options.max_hamming);
  options.AddDefaultOption("AprilTag.min_decision_margin",
                           &detection_options.min_decision_margin);
  options.AddDefaultOption("AprilTag.num_threads",
                           &detection_options.num_threads);

  options.AddDefaultOption("Triangulation.min_views",
                           &triangulation_options.min_views);
  options.AddDefaultOption("Triangulation.min_parallax_angle_deg",
                           &triangulation_options.min_parallax_angle_deg);
  options.AddDefaultOption("Triangulation.max_reprojection_error_px",
                           &triangulation_options.max_reprojection_error_px);

  options.AddDefaultOption("Alignment.min_distance_inliers",
                           &scale_options.min_inliers);
  options.AddDefaultOption("Alignment.min_common_tags",
                           &position_options.min_common_tags);
  options.AddDefaultOption("Alignment.min_inlier_ratio",
                           &scale_options.min_inlier_ratio);
  options.AddDefaultOption("Alignment.max_rel_distance_error",
                           &scale_options.max_relative_error);
  options.AddDefaultOption("Alignment.max_abs_distance_error",
                           &scale_options.max_absolute_error);
  options.AddDefaultOption("Alignment.max_position_error",
                           &position_options.ransac_options.max_error);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

#ifndef COLMAP_APRILTAG_ENABLED
  LOG(ERROR) << "AprilTag support is not enabled in this build.";
  return EXIT_FAILURE;
#else
  if (distances_path.empty() == positions_path.empty()) {
    LOG(ERROR) << "Exactly one of --distances_path or --positions_path must "
                  "be specified.";
    return EXIT_FAILURE;
  }

  if (options.image_path->empty() || !ExistsDir(*options.image_path)) {
    LOG(ERROR) << "`image_path` is not a valid directory: "
               << *options.image_path;
    return EXIT_FAILURE;
  }

  if (!ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory: " << input_path;
    return EXIT_FAILURE;
  }

  try {
    detection_options.family = ParseAprilTagFamily(family_name);
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return EXIT_FAILURE;
  }

  position_options.min_inlier_ratio = scale_options.min_inlier_ratio;
  if (!detection_options.Check() || !triangulation_options.Check() ||
      !scale_options.Check() || !position_options.Check()) {
    return EXIT_FAILURE;
  }

  const bool use_distances = !distances_path.empty();
  const std::filesystem::path& constraints_path =
      use_distances ? distances_path : positions_path;
  if (!ExistsFile(constraints_path)) {
    LOG(ERROR) << "Constraints file does not exist: " << constraints_path;
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  LOG_HEADING1("AprilTag reconstruction alignment");
  LOG(INFO) << "Family: " << AprilTagFamilyToString(detection_options.family)
            << ", mode: " << (use_distances ? "distances" : "positions");

  AprilTagAlignmentReport report;
  bool success = false;
  if (use_distances) {
    const std::vector<TagDistanceConstraint> constraints =
        ReadTagDistanceConstraints(distances_path);
    LOG(INFO) << "Loaded " << constraints.size() << " distance constraint(s)";
    success = AlignReconstructionWithAprilTagDistances(*options.image_path,
                                                      constraints,
                                                      detection_options,
                                                      triangulation_options,
                                                      scale_options,
                                                      &reconstruction,
                                                      &report);
  } else {
    const std::vector<TagPositionConstraint> constraints =
        ReadTagPositionConstraints(positions_path);
    LOG(INFO) << "Loaded " << constraints.size() << " tag position(s)";
    success = AlignReconstructionWithAprilTagPositions(*options.image_path,
                                                       constraints,
                                                       detection_options,
                                                       triangulation_options,
                                                       position_options,
                                                       &reconstruction,
                                                       &report);
  }

  LOG(INFO) << StringPrintf(
      "Processed %zu images (%zu missing), %zu detections, %zu triangulated "
      "tags (%zu rejected)",
      report.num_images_processed,
      report.num_images_missing,
      report.num_raw_detections,
      report.triangulated_tags.size(),
      report.rejected_tags.size());

  for (const RejectedAprilTag& rejected : report.rejected_tags) {
    LOG(INFO) << StringPrintf("  Rejected tag %d (%zu obs): %s",
                              rejected.tag_id,
                              rejected.num_observations,
                              rejected.reason.c_str());
  }

  if (!report_path.empty()) {
    WriteAprilTagAlignmentReportJson(report_path, report);
  }

  if (!success) {
    LOG(ERROR) << "=> AprilTag alignment failed";
    return EXIT_FAILURE;
  }

  LOG(INFO) << StringPrintf(
      "=> Alignment succeeded: scale=%.6f, inliers=%zu/%zu (%.1f%%), "
      "RMSE=%.6f, MAE=%.6f, max_error=%.6f",
      report.tgt_from_src.scale(),
      report.metrics.num_inliers,
      report.metrics.num_evaluable,
      100.0 * report.metrics.inlier_ratio,
      report.metrics.rmse,
      report.metrics.mae,
      report.metrics.max_error);

  CreateDirIfNotExists(output_path);
  reconstruction.Write(output_path);
  if (!transform_path.empty()) {
    report.tgt_from_src.ToFile(transform_path);
  }

  return EXIT_SUCCESS;
#endif
}

}  // namespace colmap
