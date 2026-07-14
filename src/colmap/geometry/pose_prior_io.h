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

#include "colmap/geometry/gps.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/util/enum_utils.h"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// In-memory representation of a pose-prior archive loaded from a file.
//
// An archive consists of three parts:
//   1. Metadata — describes the coordinate system, sensor type, reference
//      ellipsoid, and other global properties shared by all pose priors.
//   2. Schema — an ordered list of ColumnId values defining how each data
//      row maps to semantic fields (name, translation, covariance).
//   3. Data — a vector of rows, each with one cell per schema column.
//
// Schema flexibility:
//   - Column order is arbitrary; each column type appears at most once.
//   - Translation columns (LAT/LON/ALT or TX/TY/TZ) are optional. When absent,
//     the archive carries only NAME + uncertainty data.
//   - Uncertainty columns (STD_* or COV_*) are optional. STD and COV are
//     mutually exclusive.
//   - UNKNOWN columns are parsed but their values are discarded, allowing
//     forward-compatible schema evolution.
struct PosePriorArchive {
  MAKE_ENUM_CLASS(CartesianFrame, 0, LOCAL, ENU);
  MAKE_ENUM_CLASS(PoseConvention, 0, WORLD_FROM_CAM, CAM_FROM_WORLD);

  using cell_t = std::variant<std::monostate, std::string, double>;
  using row_t = std::vector<cell_t>;

  // Global metadata shared by all pose priors in the archive.
  struct Metadata {
    SensorType sensor_type = SensorType::CAMERA;

    PosePrior::CoordinateSystem coordinate_system =
        PosePrior::CoordinateSystem::UNDEFINED;
    std::optional<CartesianFrame> cartesian_frame = std::nullopt;

    // Convention for interpreting the prior translation.
    PoseConvention translation_convention = PoseConvention::WORLD_FROM_CAM;

    // Reference ellipsoid used by geographic and derived coordinate systems.
    std::optional<GPSTransform::Ellipsoid> ellipsoid = std::nullopt;
    // Origin (lat, lon, alt) of the ENU local tangent plane.
    std::optional<Eigen::Vector3d> enu_origin = std::nullopt;

    bool IsValid() const;
  };

  // clang-format off
  MAKE_ENUM_CLASS(ColumnId, -1,
      UNKNOWN,
      NAME,
      // Geographic translation
      LAT, LON, ALT,
      // Translation
      TX, TY, TZ,
      // Translation standard deviation
      STD_TX, STD_TY, STD_TZ,
      // Translation covariance
      COV_TXX, COV_TXY, COV_TXZ, COV_TYY, COV_TYZ, COV_TZZ)
  // clang-format on

  // To add a new ColumnId:
  //   1. Add the enumerator above.
  //   2. Specialize ColumnTraits<NewId> at the bottom of this file if its
  //      cell type is not double (the default).
  //   3. Add a COLMAP_COLUMN_PARSER_ENTRY for the new column in
  //      GetCellParsers() in the .cc file.
  //   4. Update AnalyzeColumnGroups() in the .cc file if the new column
  //      belongs to a group.
  //   5. Update the I/O logic, documentation, and tests.

  template <ColumnId Id>
  struct ColumnTraits {
    using value_type = double;
  };

  // Ordered list of column types describing the layout of each data row.
  struct Schema {
    std::vector<ColumnId> columns;

    bool IsValid(const Metadata& metadata) const;
  };

  Metadata metadata;
  Schema schema;
  std::vector<row_t> data;

  size_t NumColumns() const;
  size_t NumRows() const;

  bool IsValid() const;

  using data_id_resolver_t =
      std::function<std::optional<data_t>(const std::string&)>;

  // Update an existing vector of PosePrior objects with data from the archive.
  //
  // For each row in the archive, the data_id_from_name callback is called to
  // resolve the image name to a data_t. If the data_id already exists in
  // pose_priors, the matching entry is updated in-place — only fields whose
  // column types appear in this archive's schema are overwritten, all other
  // fields on the existing PosePrior are preserved. If the data_id does not
  // exist and allow_new_priors is true, a new PosePrior is appended.
  void UpdatePosePriors(const data_id_resolver_t& data_id_from_name,
                        bool allow_new_priors,
                        std::vector<PosePrior>& pose_priors) const;

  // Convert the archive data to a vector of PosePrior objects. Rows whose
  // name cannot be resolved are skipped with a warning.
  std::vector<PosePrior> ToPosePriors(
      const data_id_resolver_t& data_id_from_name) const;
};

// Read a pose prior archive from a file.
//
// JSON format:
//   {
//     "coordinate_system": "WGS84",
//     "sensor_type": "CAMERA",
//     "translation_convention": "WORLD_FROM_CAM",
//     "cartesian_frame": "ENU",
//     "ellipsoid": "WGS84",
//     "enu_origin": [47.0, 8.0, 500.0],
//     "schema": ["NAME", "LAT", "LON", "ALT"],
//     "data": [
//       ["img001.jpg", 47.3769, 8.5417, 500.0],
//       ["img002.jpg", 47.3770, 8.5418, 501.0]
//     ]
//   }
PosePriorArchive ReadPosePriorArchive(const std::filesystem::path& path);

// TODO: Implement WritePosePriorArchive

template <>
struct PosePriorArchive::ColumnTraits<PosePriorArchive::ColumnId::UNKNOWN> {
  using value_type = std::monostate;
};

template <>
struct PosePriorArchive::ColumnTraits<PosePriorArchive::ColumnId::NAME> {
  using value_type = std::string;
};

}  // namespace colmap
