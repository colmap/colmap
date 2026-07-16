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

#include <string>
#include <type_traits>
#include <vector>

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
                TRANSLATION_COV);

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

  // Parse data rows via the type-dispatch table.
  const auto data_node = pt.get_child("data");
  const auto& cell_parsers = GetCellParsers();
  for (const auto& row_node : data_node) {
    PosePriorArchive::row_t row;
    size_t column_index = 0;
    for (const auto& cell_node : row_node.second) {
      if (column_index >= archive.schema.columns.size()) {
        LOG(WARNING) << StringPrintf(
            "Row %zu has more cells than the schema defines (%zu). "
            "Extra cells will be ignored.",
            archive.data.size(),
            archive.schema.columns.size());
        break;
      }

      const auto value = cell_node.second.get_value<std::string>("");
      const auto it = cell_parsers.find(archive.schema.columns[column_index]);
      if (it != cell_parsers.end()) {
        row.push_back(it->second(value));
      }
      column_index++;
    }
    if (row.size() < archive.schema.columns.size()) {
      LOG(WARNING) << StringPrintf(
          "Row %zu has fewer cells (%zu) than the schema defines (%zu). "
          "Missing cells will be treated as empty.",
          archive.data.size(),
          row.size(),
          archive.schema.columns.size());
    }

    while (row.size() < archive.schema.columns.size()) {
      row.push_back(std::monostate{});
    }
    archive.data.push_back(std::move(row));
  }

  return archive;
}

// TODO: Implement ReadPosePriorArchiveFromCSV

}  // namespace

bool PosePriorArchive::Metadata::IsValid() const {
  if (sensor_type == SensorType::INVALID) {
    return false;
  }

  if (coordinate_system == PosePrior::CoordinateSystem::UNDEFINED) {
    return false;
  } else if (coordinate_system == PosePrior::CoordinateSystem::WGS84) {
    if (cartesian_frame.has_value()) {
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
    prior.coordinate_system = metadata.coordinate_system;

    if (has_translation) {
      if (is_geographic) {
        prior.position.x() = get_double(row[column_indices.at(ColumnId::LAT)]);
        prior.position.y() = get_double(row[column_indices.at(ColumnId::LON)]);
        prior.position.z() = get_double(row[column_indices.at(ColumnId::ALT)]);
      } else {
        prior.position.x() = get_double(row[column_indices.at(ColumnId::TX)]);
        prior.position.y() = get_double(row[column_indices.at(ColumnId::TY)]);
        prior.position.z() = get_double(row[column_indices.at(ColumnId::TZ)]);
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
        prior.position_covariance = Eigen::DiagonalMatrix<double, 3>(
            std_x * std_x, std_y * std_y, std_z * std_z);
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

        prior.position_covariance << cxx, cxy, cxz, cxy, cyy, cyz, cxz, cyz,
            czz;
      }

      if (!prior.HasPositionCov()) {
        LOG(WARNING) << "Pose prior for " << name
                     << " has no valid translation covariance data";
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

}  // namespace colmap
