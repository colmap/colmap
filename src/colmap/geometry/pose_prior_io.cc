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

#include "colmap/geometry/pose_prior_io.h"

#include "colmap/geometry/pose_prior.h"
#include "colmap/util/enum_utils.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/string.h"
#include "colmap/util/types.h"

#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#include <Eigen/Eigenvalues>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace colmap {
namespace {
MAKE_ENUM_CLASS(ColumnGroup,
                0,
                NAME,
                GEOGRAPHIC_POSITION,
                CARTESIAN_POSITION,
                TRANSLATION_STD,
                TRANSLATION_COV,
                GRAVITY,
                ROTATION,
                ROTATION_COV);

// Fixed input-validity tolerance for gravity/quaternion unit-norm checks. Not
// a CLI option: a nonzero vector/quaternion whose norm differs from one by no
// more than this is normalized; otherwise the row is rejected.
constexpr double kUnitNormTolerance = 1e-2;

// Small numerical negative tolerance for covariance PSD checks: an
// eigenvalue below this (not merely below zero) is rejected, to tolerate
// ordinary floating-point roundoff around zero.
constexpr double kCovarianceEigenvalueTolerance = -1e-9;

// A row's measurement group (gravity, quaternion, rotation covariance,
// translation STD/COV, Cartesian position) is present only when every cell
// is finite; these two helpers implement that "all or nothing" rule once
// instead of repeating an any/all boolean pair per group.
bool AnyFinite(std::initializer_list<double> values) {
  for (const double v : values) {
    if (std::isfinite(v)) {
      return true;
    }
  }
  return false;
}

bool AllFinite(std::initializer_list<double> values) {
  for (const double v : values) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  return true;
}

// Checks whether a symmetric matrix (already reconstructed from an upper
// triangle) has no eigenvalue below the numerical PSD tolerance.
template <typename MatrixType>
bool IsApproximatelyPSD(const MatrixType& matrix) {
  Eigen::SelfAdjointEigenSolver<MatrixType> solver(matrix,
                                                   Eigen::EigenvaluesOnly);
  return solver.eigenvalues().minCoeff() >= kCovarianceEigenvalueTolerance;
}

// Normalizes a nonzero vector/quaternion whose norm differs from one by no
// more than kUnitNormTolerance; returns false (leaving *coeffs unmodified)
// for a zero-norm or grossly non-unit input, which the caller must reject.
bool NormalizeIfNearUnit(Eigen::Ref<Eigen::VectorXd> coeffs) {
  const double norm = coeffs.norm();
  if (norm <= 0.0 || std::abs(norm - 1.0) > kUnitNormTolerance) {
    return false;
  }
  coeffs /= norm;
  return true;
}

struct GroupPresence {
  bool any = false;
  bool all = false;
  bool duplicate = false;
};

FlatHashMap<ColumnGroup, GroupPresence> AnalyzeColumnGroups(
    const std::vector<PosePriorArchive::ColumnId>& columns) {
  static const auto group_cols = [] {
    FlatHashMap<ColumnGroup, std::vector<PosePriorArchive::ColumnId>> m;
    m[ColumnGroup::NAME] = {PosePriorArchive::ColumnId::NAME};
    m[ColumnGroup::GEOGRAPHIC_POSITION] = {PosePriorArchive::ColumnId::LAT,
                                           PosePriorArchive::ColumnId::LON,
                                           PosePriorArchive::ColumnId::ALT};
    m[ColumnGroup::CARTESIAN_POSITION] = {PosePriorArchive::ColumnId::TX,
                                          PosePriorArchive::ColumnId::TY,
                                          PosePriorArchive::ColumnId::TZ};
    m[ColumnGroup::TRANSLATION_STD] = {PosePriorArchive::ColumnId::STD_TX,
                                       PosePriorArchive::ColumnId::STD_TY,
                                       PosePriorArchive::ColumnId::STD_TZ};
    m[ColumnGroup::TRANSLATION_COV] = {PosePriorArchive::ColumnId::COV_TXX,
                                       PosePriorArchive::ColumnId::COV_TXY,
                                       PosePriorArchive::ColumnId::COV_TXZ,
                                       PosePriorArchive::ColumnId::COV_TYY,
                                       PosePriorArchive::ColumnId::COV_TYZ,
                                       PosePriorArchive::ColumnId::COV_TZZ};
    m[ColumnGroup::GRAVITY] = {PosePriorArchive::ColumnId::GX,
                               PosePriorArchive::ColumnId::GY,
                               PosePriorArchive::ColumnId::GZ};
    m[ColumnGroup::ROTATION] = {PosePriorArchive::ColumnId::QW,
                                PosePriorArchive::ColumnId::QX,
                                PosePriorArchive::ColumnId::QY,
                                PosePriorArchive::ColumnId::QZ};
    m[ColumnGroup::ROTATION_COV] = {PosePriorArchive::ColumnId::ROT_COV_XX,
                                    PosePriorArchive::ColumnId::ROT_COV_XY,
                                    PosePriorArchive::ColumnId::ROT_COV_XZ,
                                    PosePriorArchive::ColumnId::ROT_COV_YY,
                                    PosePriorArchive::ColumnId::ROT_COV_YZ,
                                    PosePriorArchive::ColumnId::ROT_COV_ZZ};
    return m;
  }();

  FlatHashMap<PosePriorArchive::ColumnId, int> counts;
  for (const auto& col : columns) {
    if (col != PosePriorArchive::ColumnId::UNKNOWN) {
      counts[col]++;
    }
  }

  FlatHashMap<ColumnGroup, GroupPresence> result;
  for (const auto& [group, cols] : group_cols) {
    int present = 0;
    bool duplicate = false;
    for (const auto& col : cols) {
      auto it = counts.find(col);
      int c = (it != counts.end()) ? it->second : 0;
      if (c > 0) {
        ++present;
      }
      if (c > 1) {
        duplicate = true;
      }
    }
    GroupPresence p;
    p.any = present > 0;
    p.all = present == static_cast<int>(cols.size()) && !duplicate;
    p.duplicate = duplicate;
    result[group] = p;
  }
  return result;
}

template <typename EnumT>
bool MaybeGetEnumFromPropertyTree(const boost::property_tree::ptree& pt,
                                  const std::string& key,
                                  EnumT (*from_string)(std::string_view),
                                  EnumT& output) {
  const auto value = pt.get_optional<std::string>(key);
  if (value) {
    output = from_string(*value);
    return true;
  }
  return false;
}

template <typename EnumT>
bool MaybeGetEnumFromPropertyTree(const boost::property_tree::ptree& pt,
                                  const std::string& key,
                                  EnumT (*from_string)(std::string_view),
                                  std::optional<EnumT>& output) {
  EnumT value;
  if (MaybeGetEnumFromPropertyTree(pt, key, from_string, value)) {
    output = value;
    return true;
  }
  return false;
}

template <PosePriorArchive::ColumnId Id>
PosePriorArchive::cell_t ParseCell(const std::string& value) {
  using T = typename PosePriorArchive::ColumnTraits<Id>::value_type;
  if constexpr (std::is_same_v<T, std::string>) {
    return value;
  } else if constexpr (std::is_same_v<T, double>) {
    return StringToDouble(value);
  } else {
    return std::monostate{};
  }
}

using cell_parser_t =
    std::function<PosePriorArchive::cell_t(const std::string&)>;

#define COLMAP_COLUMN_PARSER_ENTRY(x) \
  {PosePriorArchive::ColumnId::x, ParseCell<PosePriorArchive::ColumnId::x>}

const FlatHashMap<PosePriorArchive::ColumnId, cell_parser_t>& GetCellParsers() {
  static const FlatHashMap<PosePriorArchive::ColumnId, cell_parser_t> parsers =
      {
          COLMAP_COLUMN_PARSER_ENTRY(UNKNOWN),
          COLMAP_COLUMN_PARSER_ENTRY(NAME),
          COLMAP_COLUMN_PARSER_ENTRY(LAT),
          COLMAP_COLUMN_PARSER_ENTRY(LON),
          COLMAP_COLUMN_PARSER_ENTRY(ALT),
          COLMAP_COLUMN_PARSER_ENTRY(TX),
          COLMAP_COLUMN_PARSER_ENTRY(TY),
          COLMAP_COLUMN_PARSER_ENTRY(TZ),
          COLMAP_COLUMN_PARSER_ENTRY(STD_TX),
          COLMAP_COLUMN_PARSER_ENTRY(STD_TY),
          COLMAP_COLUMN_PARSER_ENTRY(STD_TZ),
          COLMAP_COLUMN_PARSER_ENTRY(COV_TXX),
          COLMAP_COLUMN_PARSER_ENTRY(COV_TXY),
          COLMAP_COLUMN_PARSER_ENTRY(COV_TXZ),
          COLMAP_COLUMN_PARSER_ENTRY(COV_TYY),
          COLMAP_COLUMN_PARSER_ENTRY(COV_TYZ),
          COLMAP_COLUMN_PARSER_ENTRY(COV_TZZ),
          COLMAP_COLUMN_PARSER_ENTRY(GX),
          COLMAP_COLUMN_PARSER_ENTRY(GY),
          COLMAP_COLUMN_PARSER_ENTRY(GZ),
          COLMAP_COLUMN_PARSER_ENTRY(QW),
          COLMAP_COLUMN_PARSER_ENTRY(QX),
          COLMAP_COLUMN_PARSER_ENTRY(QY),
          COLMAP_COLUMN_PARSER_ENTRY(QZ),
          COLMAP_COLUMN_PARSER_ENTRY(ROT_COV_XX),
          COLMAP_COLUMN_PARSER_ENTRY(ROT_COV_XY),
          COLMAP_COLUMN_PARSER_ENTRY(ROT_COV_XZ),
          COLMAP_COLUMN_PARSER_ENTRY(ROT_COV_YY),
          COLMAP_COLUMN_PARSER_ENTRY(ROT_COV_YZ),
          COLMAP_COLUMN_PARSER_ENTRY(ROT_COV_ZZ),
      };
  return parsers;
}

#undef COLMAP_COLUMN_PARSER_ENTRY

PosePriorArchive ReadPosePriorArchiveFromJSON(
    const std::filesystem::path& path) {
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(path.string(), pt);

  PosePriorArchive archive;

  MaybeGetEnumFromPropertyTree(pt,
                               "coordinate_system",
                               PosePrior::CoordinateSystemFromString,
                               archive.metadata.coordinate_system);
  MaybeGetEnumFromPropertyTree(
      pt, "sensor_type", SensorTypeFromString, archive.metadata.sensor_type);
  MaybeGetEnumFromPropertyTree(pt,
                               "translation_convention",
                               PosePriorArchive::PoseConventionFromString,
                               archive.metadata.translation_convention);
  MaybeGetEnumFromPropertyTree(pt,
                               "cartesian_frame",
                               PosePriorArchive::CartesianFrameFromString,
                               archive.metadata.cartesian_frame);

  MaybeGetEnumFromPropertyTree(pt,
                               "ellipsoid",
                               GPSTransform::EllipsoidFromString,
                               archive.metadata.ellipsoid);

  if (const auto v = pt.get_optional<std::string>("height_datum")) {
    archive.metadata.height_datum = *v;
  }
  if (const auto v =
          pt.get_optional<std::string>("position_covariance_frame")) {
    archive.metadata.position_covariance_frame = *v;
  }
  if (const auto v = pt.get_optional<std::string>("rotation_convention")) {
    archive.metadata.rotation_convention = *v;
  }
  if (const auto v = pt.get_optional<std::string>("rotation_world_frame")) {
    archive.metadata.rotation_world_frame = *v;
  }
  if (const auto v =
          pt.get_optional<std::string>("rotation_covariance_convention")) {
    archive.metadata.rotation_covariance_convention = *v;
  }

  const auto enu_origin_node = pt.get_child_optional("enu_origin");
  if (enu_origin_node) {
    THROW_CHECK_EQ(enu_origin_node->size(), 3)
        << "enu_origin must be an array of 3 values";

    Eigen::Vector3d origin;
    int index = 0;
    for (const auto& value : *enu_origin_node) {
      origin(index++) = value.second.get_value<double>();
    }
    archive.metadata.enu_origin = origin;
  }

  // Parse schema: ordered list of column type strings.
  const auto schema_node = pt.get_child("schema");
  THROW_CHECK(!schema_node.empty())
      << "PosePriorArchive JSON must contain a non-empty schema array";
  for (const auto& item : schema_node) {
    std::string column_name = item.second.get_value<std::string>();
    archive.schema.columns.push_back(
        PosePriorArchive::ColumnIdFromString(column_name));
  }

  THROW_CHECK(archive.IsValid())
      << "PosePriorArchive metadata or schema is invalid";

  // Parse data rows via the type-dispatch table. Every row must have exactly
  // the schema's cell count: a producer that needs an extra field must
  // declare an unknown schema column rather than append an undeclared cell,
  // and a shifted/truncated row must not silently attach a value to the
  // wrong measurement.
  const auto data_node = pt.get_child("data");
  const auto& cell_parsers = GetCellParsers();
  for (const auto& row_node : data_node) {
    THROW_CHECK_EQ(row_node.second.size(), archive.schema.columns.size())
        << StringPrintf(
               "Row %zu has %zu cells but the schema defines %zu "
               "columns",
               archive.data.size(),
               row_node.second.size(),
               archive.schema.columns.size());

    PosePriorArchive::row_t row;
    row.reserve(archive.schema.columns.size());
    size_t column_index = 0;
    for (const auto& cell_node : row_node.second) {
      const PosePriorArchive::ColumnId column_id =
          archive.schema.columns[column_index];
      const auto value = cell_node.second.get_value<std::string>("");

      if (column_id == PosePriorArchive::ColumnId::NAME) {
        THROW_CHECK(!value.empty()) << StringPrintf(
            "Row %zu has an empty NAME cell", archive.data.size());
        row.push_back(value);
      } else if (value.empty()) {
        // A JSON `null` (or an empty measurement cell) is the absent-cell
        // state; it must never reach StringToDouble.
        row.push_back(std::monostate{});
      } else {
        const auto it = cell_parsers.find(column_id);
        row.push_back(it != cell_parsers.end() ? it->second(value)
                                               : PosePriorArchive::cell_t{});
      }
      column_index++;
    }
    archive.data.push_back(std::move(row));
  }

  return archive;
}

// TODO: Implement ReadPosePriorArchiveFromCSV

}  // namespace

namespace {
bool RequireLiteral(const std::optional<std::string>& value,
                    const char* expected,
                    const char* key) {
  if (value.has_value() && *value != expected) {
    LOG(ERROR) << "Unsupported value for `" << key << "`: `" << *value
               << "` (only `" << expected << "` is supported)";
    return false;
  }
  return true;
}
}  // namespace

bool PosePriorArchive::Metadata::IsValid() const {
  if (sensor_type != SensorType::CAMERA) {
    LOG(ERROR) << "Only sensor_type=CAMERA is supported";
    return false;
  }

  if (coordinate_system == PosePrior::CoordinateSystem::UNDEFINED) {
    return false;
  } else if (coordinate_system == PosePrior::CoordinateSystem::WGS84) {
    if (cartesian_frame.has_value()) {
      return false;
    }
    if (!ellipsoid.has_value() ||
        *ellipsoid != GPSTransform::Ellipsoid::WGS84) {
      LOG(ERROR) << "WGS84 archives require ellipsoid=WGS84";
      return false;
    }
  } else if (coordinate_system == PosePrior::CoordinateSystem::CARTESIAN) {
    if (cartesian_frame.has_value() &&
        *cartesian_frame == PosePriorArchive::CartesianFrame::ENU) {
      if (!enu_origin.has_value()) {
        return false;
      }
    }
  }

  if (!RequireLiteral(height_datum, "ELLIPSOIDAL", "height_datum") ||
      !RequireLiteral(
          rotation_convention, "SENSOR_FROM_WORLD", "rotation_convention") ||
      !RequireLiteral(rotation_covariance_convention,
                      "RIGHT_MULTIPLICATIVE_WORLD",
                      "rotation_covariance_convention")) {
    return false;
  }

  if (position_covariance_frame.has_value()) {
    const std::string& v = *position_covariance_frame;
    if (coordinate_system == PosePrior::CoordinateSystem::WGS84 &&
        v != "LOCAL_ENU") {
      LOG(ERROR) << "WGS84 position_covariance_frame must be LOCAL_ENU";
      return false;
    }
    if (coordinate_system == PosePrior::CoordinateSystem::CARTESIAN &&
        v != "CARTESIAN") {
      LOG(ERROR) << "Cartesian position_covariance_frame must be CARTESIAN";
      return false;
    }
  }

  if (rotation_world_frame.has_value()) {
    const std::string& v = *rotation_world_frame;
    if (coordinate_system == PosePrior::CoordinateSystem::WGS84) {
      if (v != "ENU") {
        LOG(ERROR) << "WGS84 rotation_world_frame must be ENU";
        return false;
      }
      if (!enu_origin.has_value()) {
        LOG(ERROR) << "WGS84 rotation_world_frame=ENU requires enu_origin";
        return false;
      }
    } else if (coordinate_system == PosePrior::CoordinateSystem::CARTESIAN) {
      if (!cartesian_frame.has_value() ||
          v != CartesianFrameToString(*cartesian_frame)) {
        LOG(ERROR) << "Cartesian rotation_world_frame must equal "
                      "cartesian_frame";
        return false;
      }
      if (*cartesian_frame == PosePriorArchive::CartesianFrame::ENU &&
          !enu_origin.has_value()) {
        LOG(ERROR) << "Cartesian rotation_world_frame=ENU requires "
                      "enu_origin";
        return false;
      }
    }
  }

  return true;
}

bool PosePriorArchive::Schema::IsValid(const Metadata& metadata) const {
  if (columns.empty()) {
    LOG(ERROR) << "Schema is empty";
    return false;
  }

  // Count and skip UNKNOWN columns.
  size_t num_skipped = 0;
  std::vector<ColumnId> known_columns;
  for (const auto& col : columns) {
    if (col == ColumnId::UNKNOWN) {
      num_skipped++;
    } else {
      known_columns.push_back(col);
    }
  }
  LOG_IF(WARNING, num_skipped > 0)
      << "Skipped " << num_skipped << " UNKNOWN columns";

  auto groups = AnalyzeColumnGroups(known_columns);
  const auto cs = metadata.coordinate_system;

  // Reject duplicate columns within any group.
  for (auto& [group, presence] : groups) {
    if (presence.duplicate) {
      LOG(ERROR) << "Duplicate columns in group " << ColumnGroupToString(group);
      return false;
    }
  }

  const auto& name_presence = groups.at(ColumnGroup::NAME);
  const auto& geographic_presence = groups.at(ColumnGroup::GEOGRAPHIC_POSITION);
  const auto& cartesian_presence = groups.at(ColumnGroup::CARTESIAN_POSITION);
  const auto& translation_std_presence =
      groups.at(ColumnGroup::TRANSLATION_STD);
  const auto& translation_cov_presence =
      groups.at(ColumnGroup::TRANSLATION_COV);

  if (!name_presence.all) {
    LOG(ERROR) << "Schema must contain a NAME column";
    return false;
  }

  if (cs == PosePrior::CoordinateSystem::WGS84) {
    if (geographic_presence.any && !geographic_presence.all) {
      LOG(ERROR) << "Incomplete translation: all of LAT/LON/ALT are required";
      return false;
    }
    if (cartesian_presence.any) {
      LOG(WARNING) << "WGS84 coordinate system ignores TX/TY/TZ columns";
    }
  } else if (cs == PosePrior::CoordinateSystem::CARTESIAN) {
    if (cartesian_presence.any && !cartesian_presence.all) {
      LOG(ERROR) << "Incomplete translation: all of TX/TY/TZ are required";
      return false;
    }
    if (geographic_presence.any) {
      LOG(WARNING) << "CARTESIAN coordinate system ignores LAT/LON/ALT columns";
    }
  }

  if (translation_std_presence.all && translation_cov_presence.all) {
    LOG(ERROR) << "Schema must not contain both STD and COV columns";
    return false;
  }

  if (translation_std_presence.any && !translation_std_presence.all) {
    LOG(ERROR) << "Incomplete STD columns: all of STD_TX, STD_TY, STD_TZ "
                  "are required";
    return false;
  }
  if (translation_cov_presence.any && !translation_cov_presence.all) {
    LOG(ERROR) << "Incomplete COV columns: all of COV_TXX, COV_TXY, "
                  "COV_TXZ, COV_TYY, COV_TYZ, COV_TZZ are required";
    return false;
  }

  const auto& gravity_presence = groups.at(ColumnGroup::GRAVITY);
  const auto& rotation_presence = groups.at(ColumnGroup::ROTATION);
  const auto& rotation_cov_presence = groups.at(ColumnGroup::ROTATION_COV);

  if (gravity_presence.any && !gravity_presence.all) {
    LOG(ERROR) << "Incomplete GX/GY/GZ columns: all three are required";
    return false;
  }
  if (rotation_presence.any && !rotation_presence.all) {
    LOG(ERROR) << "Incomplete QW/QX/QY/QZ columns: all four are required";
    return false;
  }
  if (rotation_cov_presence.any && !rotation_cov_presence.all) {
    LOG(ERROR) << "Incomplete ROT_COV_* columns: all six are required";
    return false;
  }

  const bool has_position_cov =
      translation_std_presence.all || translation_cov_presence.all;
  if (has_position_cov && !metadata.position_covariance_frame.has_value()) {
    LOG(ERROR) << "position_covariance_frame is required when a position "
                  "covariance group is present";
    return false;
  }
  if (rotation_presence.all) {
    if (!metadata.rotation_convention.has_value()) {
      LOG(ERROR) << "rotation_convention is required when a quaternion "
                    "group is present";
      return false;
    }
    if (!metadata.rotation_world_frame.has_value()) {
      LOG(ERROR) << "rotation_world_frame is required when a quaternion "
                    "group is present";
      return false;
    }
  }
  if (rotation_cov_presence.all &&
      !metadata.rotation_covariance_convention.has_value()) {
    LOG(ERROR) << "rotation_covariance_convention is required when a "
                  "rotation covariance group is present";
    return false;
  }
  if (cs == PosePrior::CoordinateSystem::WGS84 && geographic_presence.all &&
      !metadata.height_datum.has_value()) {
    LOG(ERROR) << "height_datum is required when a WGS84 position schema "
                  "is present";
    return false;
  }

  return true;
}

size_t PosePriorArchive::NumColumns() const { return schema.columns.size(); }
size_t PosePriorArchive::NumRows() const { return data.size(); }

bool PosePriorArchive::IsValid() const {
  return metadata.IsValid() && schema.IsValid(metadata);
}

PosePriorArchive ReadPosePriorArchive(const std::filesystem::path& path) {
  const std::string ext = path.extension().string();
  if (ext == ".json") {
    return ReadPosePriorArchiveFromJSON(path);
  }
  LOG(FATAL_THROW) << "Unsupported pose prior archive format: " << ext;
  // Unreachable, silence -Wreturn-type for non-MSVC compilers.
  return {};
}

void PosePriorArchive::UpdatePosePriors(
    const data_id_resolver_t& data_id_from_name,
    bool allow_new_priors,
    std::vector<PosePrior>& pose_priors) const {
  THROW_CHECK(IsValid()) << "Invalid PosePriorArchive";

  // Currently only support prior translations in world frame.
  if (metadata.translation_convention !=
      PosePriorArchive::PoseConvention::WORLD_FROM_CAM) {
    LOG(FATAL_THROW)
        << "Only PosePriorArchive::PoseConvention::WORLD_FROM_CAM is supported";
  }

  FlatHashMap<ColumnId, size_t> column_indices;
  for (size_t i = 0; i < schema.columns.size(); ++i) {
    column_indices[schema.columns[i]] = i;
  }

  const auto groups = AnalyzeColumnGroups(schema.columns);
  const auto& name_presence = groups.at(ColumnGroup::NAME);
  const auto& geographic_presence = groups.at(ColumnGroup::GEOGRAPHIC_POSITION);
  const auto& cartesian_presence = groups.at(ColumnGroup::CARTESIAN_POSITION);
  const auto& translation_std_presence =
      groups.at(ColumnGroup::TRANSLATION_STD);
  const auto& translation_cov_presence =
      groups.at(ColumnGroup::TRANSLATION_COV);
  const auto& gravity_presence = groups.at(ColumnGroup::GRAVITY);
  const auto& rotation_presence = groups.at(ColumnGroup::ROTATION);
  const auto& rotation_cov_presence = groups.at(ColumnGroup::ROTATION_COV);

  if (!name_presence.all) {
    LOG(ERROR) << "Schema is missing NAME column: cannot convert "
                  "pose prior archive to PosePrior objects";
    return;
  }

  const bool is_geographic =
      metadata.coordinate_system == PosePrior::CoordinateSystem::WGS84;
  const bool has_translation =
      is_geographic ? geographic_presence.all : cartesian_presence.all;
  const bool has_uncertainty =
      translation_std_presence.all || translation_cov_presence.all;

  const auto get_double = [](const PosePriorArchive::cell_t& cell) -> double {
    if (std::holds_alternative<double>(cell)) {
      return std::get<double>(cell);
    }
    return PosePrior::kNaN;
  };

  const size_t name_index = column_indices.at(ColumnId::NAME);

  if (allow_new_priors && pose_priors.capacity() < data.size()) {
    pose_priors.reserve(data.size());
  }

  FlatHashMap<data_t, size_t> data_id_to_index;
  for (size_t i = 0; i < pose_priors.size(); ++i) {
    data_id_to_index[pose_priors[i].corr_data_id] = i;
  }

  for (const auto& row : data) {
    if (name_index >= row.size()) {
      continue;
    }
    const std::string& name = std::get<std::string>(row[name_index]);

    const auto data_id = data_id_from_name(name);
    if (!data_id) {
      LOG(WARNING) << "Cannot resolve name: " << name;
      continue;
    }

    const bool is_prior_exist = data_id_to_index.count(*data_id);
    if (!is_prior_exist && !allow_new_priors) {
      LOG(WARNING) << "No existing pose prior for " << name
                   << " and allow_new_priors is false, skipping";
      continue;
    }

    PosePrior* prior_ptr = nullptr;
    if (is_prior_exist) {
      prior_ptr = &pose_priors[data_id_to_index.at(*data_id)];
    } else {
      pose_priors.push_back(PosePrior());
      prior_ptr = &pose_priors.back();
      prior_ptr->corr_data_id = *data_id;
    }

    PosePrior& prior = *prior_ptr;
    const auto row_has_all = [&](std::initializer_list<ColumnId> columns) {
      for (const ColumnId column : columns) {
        if (!std::isfinite(get_double(row[column_indices.at(column)]))) {
          return false;
        }
      }
      return true;
    };
    const bool row_has_position =
        has_translation &&
        (is_geographic
             ? row_has_all({ColumnId::LAT, ColumnId::LON})
             : row_has_all({ColumnId::TX, ColumnId::TY, ColumnId::TZ}));
    const bool row_has_position_covariance =
        (translation_std_presence.all &&
         row_has_all({ColumnId::STD_TX, ColumnId::STD_TY, ColumnId::STD_TZ})) ||
        (translation_cov_presence.all && row_has_all({ColumnId::COV_TXX,
                                                      ColumnId::COV_TXY,
                                                      ColumnId::COV_TXZ,
                                                      ColumnId::COV_TYY,
                                                      ColumnId::COV_TYZ,
                                                      ColumnId::COV_TZZ}));
    const bool row_has_rotation =
        rotation_presence.all &&
        row_has_all({ColumnId::QW, ColumnId::QX, ColumnId::QY, ColumnId::QZ});
    const bool row_has_rotation_covariance =
        rotation_cov_presence.all && row_has_all({ColumnId::ROT_COV_XX,
                                                  ColumnId::ROT_COV_XY,
                                                  ColumnId::ROT_COV_XZ,
                                                  ColumnId::ROT_COV_YY,
                                                  ColumnId::ROT_COV_YZ,
                                                  ColumnId::ROT_COV_ZZ});
    const bool row_has_coordinate_measurement =
        row_has_position || row_has_position_covariance || row_has_rotation ||
        row_has_rotation_covariance;
    const bool existing_has_coordinate_measurement =
        prior.HasPosition() || prior.HasPositionCov() || prior.HasRotation() ||
        prior.HasRotationCov();
    THROW_CHECK(!is_prior_exist || !row_has_coordinate_measurement ||
                !existing_has_coordinate_measurement ||
                prior.coordinate_system == metadata.coordinate_system)
        << "Cannot merge coordinate-bearing pose-prior groups expressed in "
           "different coordinate systems for "
        << name;
    if (!is_prior_exist || row_has_coordinate_measurement) {
      prior.coordinate_system = metadata.coordinate_system;
    }

    // A group is present for a row only when at least one of its cells is
    // finite (schema-level presence, checked above via has_translation, only
    // says the columns exist; a specific row may still leave them all
    // empty, in which case this group must be left untouched on `prior` so
    // merge semantics preserve whatever was already there).
    if (has_translation) {
      if (is_geographic) {
        const double lat = get_double(row[column_indices.at(ColumnId::LAT)]);
        const double lon = get_double(row[column_indices.at(ColumnId::LON)]);
        const double alt = get_double(row[column_indices.at(ColumnId::ALT)]);
        // The sole partial subgroup: LAT/LON may be finite while ALT is
        // empty (horizontal-only), but LAT and LON are always finite or
        // absent together.
        THROW_CHECK_EQ(std::isfinite(lat), std::isfinite(lon))
            << "Row for " << name
            << " has only one of LAT/LON finite; both or neither are "
               "required";
        if (std::isfinite(lat)) {
          THROW_CHECK(lat >= -90.0 && lat <= 90.0)
              << "Latitude out of range for " << name << ": " << lat;
          THROW_CHECK(lon >= -180.0 && lon <= 180.0)
              << "Longitude out of range for " << name << ": " << lon;
          prior.position.x() = lat;
          prior.position.y() = lon;
          prior.position.z() = alt;
        }
      } else {
        const double tx = get_double(row[column_indices.at(ColumnId::TX)]);
        const double ty = get_double(row[column_indices.at(ColumnId::TY)]);
        const double tz = get_double(row[column_indices.at(ColumnId::TZ)]);
        const bool any_finite = AnyFinite({tx, ty, tz});
        const bool all_finite = AllFinite({tx, ty, tz});
        THROW_CHECK(!any_finite || all_finite)
            << "Partially populated Cartesian position group for " << name;
        if (all_finite) {
          prior.position.x() = tx;
          prior.position.y() = ty;
          prior.position.z() = tz;
        }
      }

      if (!prior.HasPosition()) {
        LOG(WARNING) << "Pose prior for " << name
                     << " has no valid translation data";
      }
    }

    // Read covariance from either STD columns (diagonal) or full COV matrix.
    if (has_uncertainty) {
      if (translation_std_presence.all) {
        const double std_x =
            get_double(row[column_indices.at(ColumnId::STD_TX)]);
        const double std_y =
            get_double(row[column_indices.at(ColumnId::STD_TY)]);
        const double std_z =
            get_double(row[column_indices.at(ColumnId::STD_TZ)]);
        const bool any_finite = AnyFinite({std_x, std_y, std_z});
        const bool all_finite = AllFinite({std_x, std_y, std_z});
        THROW_CHECK(!any_finite || all_finite)
            << "Partially populated STD group for " << name;
        if (all_finite) {
          prior.position_covariance = Eigen::DiagonalMatrix<double, 3>(
              std_x * std_x, std_y * std_y, std_z * std_z);
        }
      } else if (translation_cov_presence.all) {
        const double cxx =
            get_double(row[column_indices.at(ColumnId::COV_TXX)]);
        const double cxy =
            get_double(row[column_indices.at(ColumnId::COV_TXY)]);
        const double cxz =
            get_double(row[column_indices.at(ColumnId::COV_TXZ)]);
        const double cyy =
            get_double(row[column_indices.at(ColumnId::COV_TYY)]);
        const double cyz =
            get_double(row[column_indices.at(ColumnId::COV_TYZ)]);
        const double czz =
            get_double(row[column_indices.at(ColumnId::COV_TZZ)]);
        const bool any_finite = AnyFinite({cxx, cxy, cxz, cyy, cyz, czz});
        const bool all_finite = AllFinite({cxx, cxy, cxz, cyy, cyz, czz});
        THROW_CHECK(!any_finite || all_finite)
            << "Partially populated COV group for " << name;
        if (all_finite) {
          prior.position_covariance << cxx, cxy, cxz, cxy, cyy, cyz, cxz, cyz,
              czz;
          THROW_CHECK(IsApproximatelyPSD(prior.position_covariance))
              << "Position covariance for " << name
              << " is not positive semi-definite";
        }
      }

      if (!prior.HasPositionCov()) {
        LOG(WARNING) << "Pose prior for " << name
                     << " has no valid translation covariance data";
      }
    }

    // Gravity (down, in sensor coordinates).
    if (gravity_presence.all) {
      const double gx = get_double(row[column_indices.at(ColumnId::GX)]);
      const double gy = get_double(row[column_indices.at(ColumnId::GY)]);
      const double gz = get_double(row[column_indices.at(ColumnId::GZ)]);
      const bool any_finite = AnyFinite({gx, gy, gz});
      const bool all_finite = AllFinite({gx, gy, gz});
      THROW_CHECK(!any_finite || all_finite)
          << "Partially populated gravity group for " << name;
      if (all_finite) {
        Eigen::Vector3d gravity(gx, gy, gz);
        THROW_CHECK(NormalizeIfNearUnit(gravity))
            << "Gravity vector for " << name
            << " is zero-norm or not approximately unit-norm";
        prior.gravity = gravity;
      }
    }

    // Import-time orientation normalization (WGS84 only): the archive
    // quaternion is sensor_from_archive_enu, expressed in the archive's
    // single global ENU frame (metadata.enu_origin). The database stores a
    // row-local rotation (sensor_from_local_enu_at_position) so every row is
    // self-contained; convert here, once, at import time:
    //   archive_from_local = archive_from_ecef * ecef_from_local
    //   sensor_from_local = sensor_from_archive * archive_from_local
    // and, for right-multiplicative covariance:
    //   Cov_local = local_from_archive * Cov_archive * archive_from_local
    //             = archive_from_local^T * Cov_archive * archive_from_local
    // Cartesian archives have no row-local concept (rotation is stored as
    // one shared sensor_from_cartesian_world) and need no conversion here.
    Eigen::Matrix3d archive_from_local = Eigen::Matrix3d::Identity();
    bool has_archive_to_local_basis = false;
    if (is_geographic && (rotation_presence.all || rotation_cov_presence.all)) {
      THROW_CHECK(metadata.enu_origin.has_value())
          << "WGS84 rotation group requires metadata.enu_origin";
      const Eigen::Matrix3d archive_from_ecef = GPSTransform::ENUFromECEF(
          metadata.enu_origin->x(), metadata.enu_origin->y());
      const Eigen::Matrix3d ecef_from_local =
          GPSTransform::ECEFFromENU(prior.position.x(), prior.position.y());
      archive_from_local = archive_from_ecef * ecef_from_local;
      has_archive_to_local_basis = true;
    }

    // Absolute orientation quaternion.
    if (rotation_presence.all) {
      const double qw = get_double(row[column_indices.at(ColumnId::QW)]);
      const double qx = get_double(row[column_indices.at(ColumnId::QX)]);
      const double qy = get_double(row[column_indices.at(ColumnId::QY)]);
      const double qz = get_double(row[column_indices.at(ColumnId::QZ)]);
      const bool any_finite = AnyFinite({qw, qx, qy, qz});
      const bool all_finite = AllFinite({qw, qx, qy, qz});
      THROW_CHECK(!any_finite || all_finite)
          << "Partially populated quaternion group for " << name;
      if (all_finite) {
        if (is_geographic) {
          THROW_CHECK(std::isfinite(prior.position.x()) &&
                      std::isfinite(prior.position.y()))
              << "Full WGS84 orientation for " << name
              << " requires finite latitude and longitude on the same row";
        }
        Eigen::Vector4d wxyz(qw, qx, qy, qz);
        THROW_CHECK(NormalizeIfNearUnit(wxyz))
            << "Quaternion for " << name
            << " is zero-norm or not approximately unit-norm";
        const Eigen::Quaterniond sensor_from_archive(
            wxyz(0), wxyz(1), wxyz(2), wxyz(3));
        prior.rotation =
            has_archive_to_local_basis
                ? (sensor_from_archive * Eigen::Quaterniond(archive_from_local))
                      .normalized()
                : sensor_from_archive;
      }
    }

    // Rotation covariance, right-multiplicative in the same world basis as
    // the quaternion above.
    if (rotation_cov_presence.all) {
      const double rxx =
          get_double(row[column_indices.at(ColumnId::ROT_COV_XX)]);
      const double rxy =
          get_double(row[column_indices.at(ColumnId::ROT_COV_XY)]);
      const double rxz =
          get_double(row[column_indices.at(ColumnId::ROT_COV_XZ)]);
      const double ryy =
          get_double(row[column_indices.at(ColumnId::ROT_COV_YY)]);
      const double ryz =
          get_double(row[column_indices.at(ColumnId::ROT_COV_YZ)]);
      const double rzz =
          get_double(row[column_indices.at(ColumnId::ROT_COV_ZZ)]);
      const bool any_finite = AnyFinite({rxx, rxy, rxz, ryy, ryz, rzz});
      const bool all_finite = AllFinite({rxx, rxy, rxz, ryy, ryz, rzz});
      THROW_CHECK(!any_finite || all_finite)
          << "Partially populated rotation covariance group for " << name;
      if (all_finite) {
        Eigen::Matrix3d rotation_covariance;
        rotation_covariance << rxx, rxy, rxz, rxy, ryy, ryz, rxz, ryz, rzz;
        THROW_CHECK(IsApproximatelyPSD(rotation_covariance))
            << "Rotation covariance for " << name
            << " is not positive semi-definite";
        if (has_archive_to_local_basis) {
          // Cov_local = archive_from_local^T * Cov_archive *
          // archive_from_local.
          rotation_covariance = archive_from_local.transpose() *
                                rotation_covariance * archive_from_local;
        }
        prior.rotation_covariance = rotation_covariance;
      }
    }
  }
}

std::vector<PosePrior> PosePriorArchive::ToPosePriors(
    const data_id_resolver_t& data_id_from_name) const {
  std::vector<PosePrior> pose_priors;
  UpdatePosePriors(data_id_from_name, /*allow_new_priors=*/true, pose_priors);
  return pose_priors;
}

bool PosePriorArchive::HasDuplicateResolvedNames(
    const data_id_resolver_t& data_id_from_name) const {
  FlatHashMap<ColumnId, size_t> column_indices;
  for (size_t i = 0; i < schema.columns.size(); ++i) {
    column_indices[schema.columns[i]] = i;
  }
  const auto it = column_indices.find(ColumnId::NAME);
  if (it == column_indices.end()) {
    return false;
  }
  const size_t name_index = it->second;

  FlatHashSet<data_t> seen;
  for (const auto& row : data) {
    if (name_index >= row.size()) {
      continue;
    }
    const std::string& name = std::get<std::string>(row[name_index]);
    const auto data_id = data_id_from_name(name);
    if (!data_id) {
      continue;
    }
    if (!seen.insert(*data_id).second) {
      return true;
    }
  }
  return false;
}

}  // namespace colmap
