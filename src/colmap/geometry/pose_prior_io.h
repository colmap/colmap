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

#include "colmap/geometry/pose_prior.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// A pose-prior archive: measured sensor data for a capture, keyed by image
// name, read once at import time.
//
// There is exactly one accepted format, documented in doc/pose_priors.rst and
// summarized here:
//
//   {
//     "schema_version": 1,
//     "coordinate_system": "WGS84",
//     "sensor_type": "CAMERA",
//     "ellipsoid": "WGS84",
//     "height_datum": "ELLIPSOIDAL",
//     "position_covariance_frame": "LOCAL_ENU",
//     "gravity_frame": "CAMERA",
//     "gravity_direction": "DOWN",
//     "schema": ["NAME", "LAT", "LON", "ALT",
//                "STD_TX", "STD_TY", "STD_TZ", "GX", "GY", "GZ"],
//     "data": [["image.jpg", 45.0, -73.0, 40.0, 2.0, 2.0, 4.0, 0.0, 1.0, 0.0]]
//   }
//
// Reading is fail-closed in both directions. Nothing is optional except the
// two sensor groups below, nothing unrecognized is tolerated, and no value is
// repaired or inferred. An archive is produced by a trusted adapter that
// already knows what it measured; a reader that guesses at a malformed one
// turns a fixable producer bug into a wrong reconstruction that still looks
// plausible, which is the expensive kind.
struct PosePriorArchive {
  // The only version this build reads. An archive declaring any other version
  // is rejected rather than interpreted: a version bump exists precisely to
  // say "the meaning of these fields changed".
  static constexpr int kSchemaVersion = 1;

  // One fully-populated measurement. Position and its uncertainty are always
  // present -- a row that cannot say where the camera was has nothing this
  // workflow can use. Gravity and heading are the only optional groups, and
  // each is all-or-nothing per row.
  struct Row {
    std::string name;
    // Latitude in degrees, longitude in degrees, ellipsoidal altitude in
    // metres. Never a geoid/orthometric height; see `height_datum`.
    Eigen::Vector3d position_wgs84 = Eigen::Vector3d::Zero();
    // Symmetric positive-definite, expressed in the local ENU frame at this
    // row's own latitude and longitude.
    Eigen::Matrix3d position_covariance = Eigen::Matrix3d::Identity();
    // Measured down direction in camera coordinates, normalized on read.
    std::optional<Eigen::Vector3d> gravity;
    // Clockwise azimuth of the camera's horizontally-projected forward axis
    // from true north, and its one-sigma uncertainty, both in radians.
    std::optional<double> heading_rad;
    std::optional<double> heading_stddev_rad;
  };

  std::vector<Row> rows;
  // Whether the schema declared each optional group. A group is either in the
  // schema for the whole archive or absent from it; when present, individual
  // rows may still be null.
  bool schema_has_gravity = false;
  bool schema_has_heading = false;

  // Builds one complete PosePrior per row, in row order.
  //
  // `data_ids` must have one entry per row, already resolved by the caller.
  // Resolution lives with the caller because it is the caller that knows how
  // to report every unresolved name at once, before writing anything.
  std::vector<PosePrior> ToPosePriors(
      const std::vector<data_t>& data_ids) const;
};

// Reads and completely validates an archive, or throws with a message naming
// the offending key, row, or column. A returned archive needs no further
// checking.
PosePriorArchive ReadPosePriorArchive(const std::filesystem::path& path);

}  // namespace colmap
