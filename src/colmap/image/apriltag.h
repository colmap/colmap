// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/sensor/bitmap.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/enum_utils.h"

#include <array>
#include <memory>
#include <string_view>
#include <vector>

#include <Eigen/Core>

namespace colmap {

MAKE_ENUM_CLASS_OVERLOAD_STREAM(AprilTagFamily,
                                0,
                                TAG_36H11,
                                TAG_25H9,
                                TAG_16H5,
                                TAG_36H10,
                                TAG_STANDARD_41H12,
                                TAG_CIRCLE_21H7);

// Parse an AprilTag family name accepting both enum names (e.g. "TAG_36H11")
// and canonical AprilTag family names (e.g. "tag36h11"), case-insensitively.
AprilTagFamily ParseAprilTagFamily(std::string_view name);

struct AprilTagDetectionOptions {
  AprilTagFamily family = AprilTagFamily::TAG_36H11;

  // Decimation factor for quad detection (>= 1.0). Supported values are 1.0,
  // 1.5, or integer values >= 2 (e.g. 2.0). Higher values speed up detection
  // at the expense of small-tag sensitivity.
  double quad_decimate = 2.0;

  // Standard deviation in pixels of the Gaussian blur applied to the decimated
  // image before quad segmentation (negative values apply sharpening).
  double quad_sigma = 0.0;

  // Whether to refine quad edges by snapping to nearby image gradients.
  bool refine_edges = true;

  // Amount of sharpening applied to decoded binary payload images.
  double decode_sharpening = 0.25;

  // Maximum number of corrected bit errors allowed (supported range [0, 2]).
  int max_hamming = 1;

  // Minimum decision margin required to retain a tag detection.
  double min_decision_margin = 25.0;

  // Number of threads used by the detector (-1 uses all available cores).
  int num_threads = -1;

  bool Check() const;
};

struct AprilTagDetection {
  int id = -1;

  // Tag center in COLMAP pixel coordinates, where (0, 0) is the top-left
  // corner of the top-left pixel so pixel (i, j) has center (i + 0.5, j + 0.5).
  Eigen::Vector2d center = Eigen::Vector2d::Zero();

  // Four tag corners in counter-clockwise order in COLMAP pixel coordinates.
  std::array<Eigen::Vector2d, 4> corners{};

  double decision_margin = 0.0;
  int hamming = 0;
};

// Stateful single-threaded AprilTag detector wrapper. Reusing an instance
// across images avoids recreating the family decoding tables per image.
// Instances are not thread-safe; use one instance per worker thread with
// num_threads = 1.
class AprilTagDetector {
 public:
  explicit AprilTagDetector(const AprilTagDetectionOptions& options = {});
  ~AprilTagDetector();

  AprilTagDetector(const AprilTagDetector&) = delete;
  AprilTagDetector& operator=(const AprilTagDetector&) = delete;
  AprilTagDetector(AprilTagDetector&&) noexcept;
  AprilTagDetector& operator=(AprilTagDetector&&) noexcept;

  // Detect AprilTags in the given bitmap (converted to grayscale if RGB).
  std::vector<AprilTagDetection> Detect(const Bitmap& bitmap) const;

  const AprilTagDetectionOptions& Options() const { return options_; }

 private:
  struct Impl;
  AprilTagDetectionOptions options_;
  std::unique_ptr<Impl> impl_;
};

// Convenience helper for one-shot AprilTag detection on a single bitmap.
std::vector<AprilTagDetection> DetectAprilTags(
    const Bitmap& bitmap, const AprilTagDetectionOptions& options = {});

// Render a synthetic grayscale bitmap with the given AprilTag centered inside a
// white quiet-zone border.
Bitmap SynthesizeAprilTagBitmap(AprilTagFamily family,
                                int tag_id,
                                int quiet_zone_bits = 2,
                                int pixel_scale = 16);

}  // namespace colmap
