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

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/string.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Eigenvalues>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace colmap {
namespace {

// Tolerance on the gravity direction's norm. Action cameras and phones expose
// gravity as an already device-fused unit direction, so this admits ordinary
// serialization rounding and nothing else. It is deliberately not compared
// against 9.80665: a normalized direction has no acceleration magnitude left
// in it to compare.
constexpr double kGravityUnitNormTolerance = 1e-2;

constexpr double kPi = 3.14159265358979323846;

// Every column this build understands. Anything else in `schema` is an error,
// not a field to skip: an unrecognized name is a producer typo far more often
// than it is a deliberate extension, and skipping it would silently drop the
// measurement the operator believed they had supplied.
enum class Column {
  NAME,
  LAT,
  LON,
  ALT,
  STD_TX,
  STD_TY,
  STD_TZ,
  COV_TXX,
  COV_TXY,
  COV_TXZ,
  COV_TYY,
  COV_TYZ,
  COV_TZZ,
  GX,
  GY,
  GZ,
  HEADING_DEG,
  HEADING_STD_DEG,
};

const std::map<std::string, Column>& ColumnsByName() {
  static const std::map<std::string, Column> kColumns = {
      {"NAME", Column::NAME},
      {"LAT", Column::LAT},
      {"LON", Column::LON},
      {"ALT", Column::ALT},
      {"STD_TX", Column::STD_TX},
      {"STD_TY", Column::STD_TY},
      {"STD_TZ", Column::STD_TZ},
      {"COV_TXX", Column::COV_TXX},
      {"COV_TXY", Column::COV_TXY},
      {"COV_TXZ", Column::COV_TXZ},
      {"COV_TYY", Column::COV_TYY},
      {"COV_TYZ", Column::COV_TYZ},
      {"COV_TZZ", Column::COV_TZZ},
      {"GX", Column::GX},
      {"GY", Column::GY},
      {"GZ", Column::GZ},
      {"HEADING_DEG", Column::HEADING_DEG},
      {"HEADING_STD_DEG", Column::HEADING_STD_DEG},
  };
  return kColumns;
}

const std::vector<Column> kStdColumns = {
    Column::STD_TX, Column::STD_TY, Column::STD_TZ};
const std::vector<Column> kCovColumns = {Column::COV_TXX,
                                         Column::COV_TXY,
                                         Column::COV_TXZ,
                                         Column::COV_TYY,
                                         Column::COV_TYZ,
                                         Column::COV_TZZ};
const std::vector<Column> kGravityColumns = {
    Column::GX, Column::GY, Column::GZ};
const std::vector<Column> kHeadingColumns = {Column::HEADING_DEG,
                                             Column::HEADING_STD_DEG};

// Maps a schema column to its index in each data row.
using ColumnIndex = std::map<Column, size_t>;

// How a JSON `null` reaches us. Boost's property_tree has no null type: it
// renders one as the literal text "null", and an empty JSON string as "".
// Neither is a number, so for the numeric columns -- the only ones a group may
// omit -- both unambiguously mean "no reading". Accepting both rather than
// adding a second JSON dependency to tell them apart is deliberate; the
// distinction does not exist for any value this schema can hold.
bool IsNullCell(const std::string& cell) {
  return cell.empty() || cell == "null";
}

double ParseNumericCell(const std::string& cell,
                        const std::string& column_name,
                        size_t row_index) {
  THROW_CHECK(!IsNullCell(cell))
      << "row " << row_index << ": column " << column_name
      << " is null, but only the gravity and heading groups may be null";
  try {
    size_t consumed = 0;
    const double value = std::stod(cell, &consumed);
    THROW_CHECK_EQ(consumed, cell.size())
        << "row " << row_index << ": column " << column_name
        << " is not a number: `" << cell << "`";
    return value;
  } catch (const std::invalid_argument&) {
    LOG(FATAL_THROW) << "row " << row_index << ": column " << column_name
                     << " is not a number: `" << cell << "`";
  } catch (const std::out_of_range&) {
    LOG(FATAL_THROW) << "row " << row_index << ": column " << column_name
                     << " is out of range for a double: `" << cell << "`";
  }
  return 0.0;
}

// Reads a JSON array of strings. Rejects a nested object or array, which
// property_tree would otherwise silently present as an empty value.
std::vector<std::string> ReadStringArray(
    const boost::property_tree::ptree& node, const std::string& what) {
  std::vector<std::string> values;
  for (const auto& [key, child] : node) {
    THROW_CHECK(key.empty()) << what << " must be a JSON array";
    THROW_CHECK(child.empty())
        << what << " must contain scalars, not nested objects or arrays";
    values.push_back(child.get_value<std::string>());
  }
  return values;
}

// Requires a top-level string key to be present and exactly equal to the one
// supported value. A single supported value is still validated rather than
// assumed, so that an archive from a future producer that means something
// different by it fails instead of being misread.
void RequireExactString(const boost::property_tree::ptree& root,
                        const std::string& key,
                        const std::string& expected) {
  const auto value = root.get_optional<std::string>(key);
  THROW_CHECK(value.has_value()) << "archive is missing required key `" << key
                                 << "` (expected \"" << expected << "\")";
  THROW_CHECK_EQ(*value, expected)
      << "archive key `" << key << "` must be \"" << expected << "\"";
}

void RequireAbsent(const std::set<std::string>& keys,
                   const std::string& key,
                   const std::string& because) {
  THROW_CHECK(keys.count(key) == 0)
      << "archive key `" << key << "` is only meaningful " << because
      << "; remove it";
}

Eigen::Matrix3d CovarianceFromStd(const Eigen::Vector3d& stddev,
                                  size_t row_index) {
  for (int i = 0; i < 3; ++i) {
    THROW_CHECK(std::isfinite(stddev[i]) && stddev[i] > 0.0)
        << "row " << row_index
        << ": every standard deviation must be finite and strictly positive, "
           "got "
        << stddev.transpose();
  }
  return stddev.cwiseProduct(stddev).asDiagonal();
}

Eigen::Matrix3d CovarianceFromUpperTriangle(double xx,
                                            double xy,
                                            double xz,
                                            double yy,
                                            double yz,
                                            double zz,
                                            size_t row_index) {
  Eigen::Matrix3d cov;
  // Symmetric by construction: only the upper triangle is stated, so an
  // asymmetric input is not representable rather than silently averaged.
  cov << xx, xy, xz, xy, yy, yz, xz, yz, zz;
  THROW_CHECK(cov.allFinite())
      << "row " << row_index << ": position covariance is not finite";

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov,
                                                        Eigen::EigenvaluesOnly);
  THROW_CHECK_EQ(solver.info(), Eigen::Success)
      << "row " << row_index
      << ": position covariance eigendecomposition "
         "failed";
  THROW_CHECK_GT(solver.eigenvalues().minCoeff(), 0.0)
      << "row " << row_index
      << ": position covariance must be strictly positive definite; its "
         "smallest eigenvalue is "
      << solver.eigenvalues().minCoeff()
      << ". A zero, singular, or indefinite covariance states a direction of "
         "perfect certainty, which weighted fitting reads as an infinite "
         "weight.";
  return cov;
}

}  // namespace

std::vector<PosePrior> PosePriorArchive::ToPosePriors(
    const std::vector<data_t>& data_ids) const {
  THROW_CHECK_EQ(data_ids.size(), rows.size())
      << "ToPosePriors needs one resolved data id per archive row";

  std::vector<PosePrior> priors;
  priors.reserve(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row& row = rows[i];
    PosePrior prior;
    prior.corr_data_id = data_ids[i];
    prior.coordinate_system = PosePrior::CoordinateSystem::WGS84;
    prior.position = row.position_wgs84;
    prior.position_covariance = row.position_covariance;
    if (row.gravity.has_value()) {
      prior.gravity = *row.gravity;
    }
    if (row.heading_rad.has_value()) {
      prior.heading_rad = *row.heading_rad;
      prior.heading_stddev_rad = *row.heading_stddev_rad;
    }
    priors.push_back(prior);
  }
  return priors;
}

PosePriorArchive ReadPosePriorArchive(const std::filesystem::path& path) {
  boost::property_tree::ptree root;
  try {
    boost::property_tree::read_json(path.string(), root);
  } catch (const boost::property_tree::json_parser_error& error) {
    LOG(FATAL_THROW) << "Failed to parse pose prior archive `" << path.string()
                     << "`: " << error.what();
  }

  // ---- Schema columns -----------------------------------------------------
  const auto schema_node = root.get_child_optional("schema");
  THROW_CHECK(schema_node.has_value()) << "archive is missing `schema`";
  const std::vector<std::string> schema_names =
      ReadStringArray(*schema_node, "`schema`");
  THROW_CHECK(!schema_names.empty()) << "`schema` must not be empty";

  ColumnIndex column_index;
  for (size_t i = 0; i < schema_names.size(); ++i) {
    const auto it = ColumnsByName().find(schema_names[i]);
    THROW_CHECK(it != ColumnsByName().end())
        << "`schema` column " << i << " is `" << schema_names[i]
        << "`, which this build does not recognize";
    const auto [_, inserted] = column_index.emplace(it->second, i);
    THROW_CHECK(inserted) << "`schema` names column `" << schema_names[i]
                          << "` more than once";
  }

  const auto has_all = [&](const std::vector<Column>& columns) {
    return std::all_of(columns.begin(), columns.end(), [&](Column column) {
      return column_index.count(column) > 0;
    });
  };
  const auto has_any = [&](const std::vector<Column>& columns) {
    return std::any_of(columns.begin(), columns.end(), [&](Column column) {
      return column_index.count(column) > 0;
    });
  };
  const auto require_whole_group = [&](const std::vector<Column>& columns,
                                       const std::string& group) {
    THROW_CHECK(!has_any(columns) || has_all(columns))
        << "the " << group
        << " column group is partially present in `schema`; it must be "
           "wholly present or wholly absent";
  };

  for (const Column required :
       {Column::NAME, Column::LAT, Column::LON, Column::ALT}) {
    THROW_CHECK(column_index.count(required) > 0)
        << "`schema` must contain NAME, LAT, LON and ALT";
  }

  require_whole_group(kStdColumns, "STD_T*");
  require_whole_group(kCovColumns, "COV_T*");
  require_whole_group(kGravityColumns, "gravity (GX/GY/GZ)");
  require_whole_group(kHeadingColumns, "heading (HEADING_DEG/HEADING_STD_DEG)");

  const bool has_std = has_all(kStdColumns);
  const bool has_cov = has_all(kCovColumns);
  THROW_CHECK(has_std != has_cov)
      << "`schema` must state position uncertainty exactly once, as either "
         "STD_TX/STD_TY/STD_TZ or the six COV_T* columns";

  const bool has_gravity = has_all(kGravityColumns);
  const bool has_heading = has_all(kHeadingColumns);
  THROW_CHECK(!has_heading || has_gravity)
      << "the heading group requires the gravity group: a heading is an "
         "azimuth in the horizontal plane that the measured camera-frame down "
         "vector establishes";

  // ---- Metadata -----------------------------------------------------------
  std::set<std::string> present_keys;
  for (const auto& [key, child] : root) {
    THROW_CHECK(!key.empty()) << "archive must be a JSON object";
    const auto [_, inserted] = present_keys.insert(key);
    THROW_CHECK(inserted) << "archive names key `" << key << "` more than once";
  }

  const auto schema_version = root.get_optional<int>("schema_version");
  THROW_CHECK(schema_version.has_value())
      << "archive is missing `schema_version`";
  THROW_CHECK_EQ(*schema_version, PosePriorArchive::kSchemaVersion)
      << "this build reads pose prior archive schema_version "
      << PosePriorArchive::kSchemaVersion << " only";

  RequireExactString(root, "coordinate_system", "WGS84");
  RequireExactString(root, "sensor_type", "CAMERA");
  RequireExactString(root, "ellipsoid", "WGS84");
  RequireExactString(root, "height_datum", "ELLIPSOIDAL");
  RequireExactString(root, "position_covariance_frame", "LOCAL_ENU");

  std::set<std::string> allowed_keys = {"schema_version",
                                        "coordinate_system",
                                        "sensor_type",
                                        "ellipsoid",
                                        "height_datum",
                                        "position_covariance_frame",
                                        "schema",
                                        "data"};
  if (has_gravity) {
    RequireExactString(root, "gravity_frame", "CAMERA");
    RequireExactString(root, "gravity_direction", "DOWN");
    allowed_keys.insert("gravity_frame");
    allowed_keys.insert("gravity_direction");
  } else {
    RequireAbsent(present_keys, "gravity_frame", "with a gravity column group");
    RequireAbsent(
        present_keys, "gravity_direction", "with a gravity column group");
  }
  if (has_heading) {
    RequireExactString(root, "heading_reference", "TRUE_NORTH");
    RequireExactString(
        root, "heading_axis", "CAMERA_FORWARD_PROJECTED_HORIZONTAL");
    RequireExactString(root, "heading_rotation", "CLOCKWISE_FROM_NORTH");
    allowed_keys.insert("heading_reference");
    allowed_keys.insert("heading_axis");
    allowed_keys.insert("heading_rotation");
  } else {
    RequireAbsent(
        present_keys, "heading_reference", "with a heading column group");
    RequireAbsent(present_keys, "heading_axis", "with a heading column group");
    RequireAbsent(
        present_keys, "heading_rotation", "with a heading column group");
  }

  for (const std::string& key : present_keys) {
    THROW_CHECK(allowed_keys.count(key) > 0)
        << "archive contains unknown key `" << key << "`";
  }

  // ---- Rows ---------------------------------------------------------------
  const auto data_node = root.get_child_optional("data");
  THROW_CHECK(data_node.has_value()) << "archive is missing `data`";

  PosePriorArchive archive;
  archive.schema_has_gravity = has_gravity;
  archive.schema_has_heading = has_heading;

  std::set<std::string> seen_names;
  size_t row_index = 0;
  for (const auto& [key, row_node] : *data_node) {
    THROW_CHECK(key.empty()) << "`data` must be a JSON array of rows";
    const std::vector<std::string> cells =
        ReadStringArray(row_node, StringPrintf("`data` row %zu", row_index));
    THROW_CHECK_EQ(cells.size(), schema_names.size())
        << "row " << row_index << " has " << cells.size() << " cells but the "
        << "schema declares " << schema_names.size() << " columns";

    const auto cell_of = [&](Column column) -> const std::string& {
      return cells[column_index.at(column)];
    };
    const auto number_of = [&](Column column,
                               const std::string& column_name) -> double {
      return ParseNumericCell(cell_of(column), column_name, row_index);
    };

    PosePriorArchive::Row row;

    row.name = cell_of(Column::NAME);
    THROW_CHECK(!row.name.empty())
        << "row " << row_index << " has an empty NAME";
    THROW_CHECK(seen_names.insert(row.name).second)
        << "archive names image `" << row.name
        << "` more than once; which row wins would be arbitrary";

    const double lat = number_of(Column::LAT, "LAT");
    const double lon = number_of(Column::LON, "LON");
    const double alt = number_of(Column::ALT, "ALT");
    THROW_CHECK(std::isfinite(lat) && std::isfinite(lon) && std::isfinite(alt))
        << "row " << row_index << " (" << row.name
        << "): LAT/LON/ALT must all be finite";
    THROW_CHECK(lat >= -90.0 && lat <= 90.0)
        << "row " << row_index << " (" << row.name << "): LAT " << lat
        << " is outside [-90, 90]";
    THROW_CHECK(lon >= -180.0 && lon <= 180.0)
        << "row " << row_index << " (" << row.name << "): LON " << lon
        << " is outside [-180, 180]";
    // A geographically distant but otherwise valid row is not rejected here.
    // Whether it is an outlier is a question about the capture, which robust
    // fitting answers with the other rows in hand; the parser only knows
    // whether the row is well-formed.
    row.position_wgs84 = Eigen::Vector3d(lat, lon, alt);

    if (has_std) {
      const Eigen::Vector3d stddev(number_of(Column::STD_TX, "STD_TX"),
                                   number_of(Column::STD_TY, "STD_TY"),
                                   number_of(Column::STD_TZ, "STD_TZ"));
      row.position_covariance = CovarianceFromStd(stddev, row_index);
    } else {
      row.position_covariance =
          CovarianceFromUpperTriangle(number_of(Column::COV_TXX, "COV_TXX"),
                                      number_of(Column::COV_TXY, "COV_TXY"),
                                      number_of(Column::COV_TXZ, "COV_TXZ"),
                                      number_of(Column::COV_TYY, "COV_TYY"),
                                      number_of(Column::COV_TYZ, "COV_TYZ"),
                                      number_of(Column::COV_TZZ, "COV_TZZ"),
                                      row_index);
    }

    bool row_has_gravity = false;
    if (has_gravity) {
      const size_t num_null =
          static_cast<size_t>(IsNullCell(cell_of(Column::GX))) +
          static_cast<size_t>(IsNullCell(cell_of(Column::GY))) +
          static_cast<size_t>(IsNullCell(cell_of(Column::GZ)));
      THROW_CHECK(num_null == 0 || num_null == 3)
          << "row " << row_index << " (" << row.name
          << "): GX/GY/GZ must be all present or all null";
      if (num_null == 0) {
        Eigen::Vector3d gravity(number_of(Column::GX, "GX"),
                                number_of(Column::GY, "GY"),
                                number_of(Column::GZ, "GZ"));
        THROW_CHECK(gravity.allFinite())
            << "row " << row_index << " (" << row.name
            << "): gravity is not finite";
        const double norm = gravity.norm();
        THROW_CHECK(norm > 0.0 &&
                    std::abs(norm - 1.0) <= kGravityUnitNormTolerance)
            << "row " << row_index << " (" << row.name
            << "): gravity must be a unit direction, got norm " << norm
            << ". This column carries a normalized device-fused down "
               "direction, not an acceleration in m/s^2.";
        row.gravity = (gravity / norm).eval();
        row_has_gravity = true;
      }
    }

    if (has_heading) {
      const size_t num_null =
          static_cast<size_t>(IsNullCell(cell_of(Column::HEADING_DEG))) +
          static_cast<size_t>(IsNullCell(cell_of(Column::HEADING_STD_DEG)));
      THROW_CHECK(num_null == 0 || num_null == 2)
          << "row " << row_index << " (" << row.name
          << "): HEADING_DEG/HEADING_STD_DEG must be both present or both "
             "null";
      if (num_null == 0) {
        THROW_CHECK(row_has_gravity)
            << "row " << row_index << " (" << row.name
            << "): a heading requires a gravity reading on the same row, "
               "which establishes the horizontal plane the azimuth is "
               "measured in";
        const double heading_deg =
            number_of(Column::HEADING_DEG, "HEADING_DEG");
        const double heading_stddev_deg =
            number_of(Column::HEADING_STD_DEG, "HEADING_STD_DEG");
        THROW_CHECK(heading_deg >= 0.0 && heading_deg < 360.0)
            << "row " << row_index << " (" << row.name << "): HEADING_DEG "
            << heading_deg << " is outside [0, 360)";
        THROW_CHECK(heading_stddev_deg > 0.0 && heading_stddev_deg <= 180.0)
            << "row " << row_index << " (" << row.name << "): HEADING_STD_DEG "
            << heading_stddev_deg
            << " is outside (0, 180]. Every heading row states its own "
               "uncertainty; there is no global fallback.";
        row.heading_rad = heading_deg * kPi / 180.0;
        row.heading_stddev_rad = heading_stddev_deg * kPi / 180.0;
      }
    }

    archive.rows.push_back(std::move(row));
    ++row_index;
  }

  THROW_CHECK(!archive.rows.empty())
      << "archive `data` contains no rows; there is nothing to import";
  return archive;
}

}  // namespace colmap
