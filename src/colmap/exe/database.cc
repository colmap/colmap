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

#include "colmap/exe/database.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/geometry/pose.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/rig.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace colmap {

namespace {

bool HasFields(const std::unordered_map<std::string, size_t>& col_idx,
               const std::vector<std::string>& fields) {
  for (const auto& f : fields) {
    if (col_idx.find(f) == col_idx.end()) return false;
  }
  return true;
}
}  // namespace

int RunDatabaseCleaner(int argc, char** argv) {
  std::string type;

  OptionManager options;
  options.AddRequiredOption("type", &type, "{all, images, features, matches}");
  options.AddDatabaseOptions();
  options.Parse(argc, argv);

  StringToLower(&type);
  Database database(*options.database_path);
  PrintHeading1("Clearing database");
  {
    DatabaseTransaction transaction(&database);
    if (type == "all") {
      PrintHeading2("Clearing all tables");
      database.ClearAllTables();
    } else if (type == "images") {
      PrintHeading2("Clearing Images and all dependent tables");
      database.ClearImages();
      database.ClearTwoViewGeometries();
      database.ClearMatches();
    } else if (type == "features") {
      PrintHeading2("Clearing image features and matches");
      database.ClearDescriptors();
      database.ClearKeypoints();
      database.ClearTwoViewGeometries();
      database.ClearMatches();
    } else if (type == "matches") {
      PrintHeading2("Clearing image matches");
      database.ClearTwoViewGeometries();
      database.ClearMatches();
    } else {
      LOG(ERROR) << "Invalid cleanup type; no changes in database";
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

int RunDatabaseCreator(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.Parse(argc, argv);

  Database database(*options.database_path);

  return EXIT_SUCCESS;
}

int RunDatabaseMerger(int argc, char** argv) {
  std::string database_path1;
  std::string database_path2;
  std::string merged_database_path;

  OptionManager options;
  options.AddRequiredOption("database_path1", &database_path1);
  options.AddRequiredOption("database_path2", &database_path2);
  options.AddRequiredOption("merged_database_path", &merged_database_path);
  options.Parse(argc, argv);

  if (ExistsFile(merged_database_path)) {
    LOG(ERROR) << "Merged database file must not exist.";
    return EXIT_FAILURE;
  }

  Database database1(database_path1);
  Database database2(database_path2);
  Database merged_database(merged_database_path);
  Database::Merge(database1, database2, &merged_database);

  return EXIT_SUCCESS;
}

int RunRigConfigurator(int argc, char** argv) {
  std::string database_path;
  std::string rig_config_path;
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("database_path", &database_path);
  options.AddRequiredOption("rig_config_path",
                            &rig_config_path,
                            "Rig configuration as a .json file.");
  options.AddDefaultOption("input_path",
                           &input_path,
                           "Optional input reconstruction to automatically "
                           "derive the (average) rig and camera calibrations. "
                           "If not provided, the rig intrinsics and extrinsics "
                           "must be specified in the provided config.");
  options.AddDefaultOption(
      "output_path",
      &output_path,
      "Optional output reconstruction with configured rigs/frames.");
  options.Parse(argc, argv);

  std::optional<Reconstruction> reconstruction;
  if (!input_path.empty()) {
    reconstruction = std::make_optional<Reconstruction>();
    reconstruction->Read(input_path);
  }

  Database database(database_path);

  ApplyRigConfig(
      ReadRigConfig(rig_config_path),
      database,
      reconstruction.has_value() ? &reconstruction.value() : nullptr);

  if (reconstruction.has_value() && !output_path.empty()) {
    reconstruction->Write(output_path);
  }

  return EXIT_SUCCESS;
}

// Pose prior input file format:
// Each line corresponds to one image's pose prior and can include position,
// rotation, and their uncertainties. Can be NaN or omitted if unknown.
//
// Columns:
//   name          - Image name, must match database image name.
//   x, y, z       - Position in world coordinates.
//   cs            - World coordinate system
//   std_x, std_y, std_z
//                 - Standard deviations of position prior.
//                   Used to construct the position covariance matrix.
//   qw, qx, qy, qz
//                 - Quaternion representing world-to-camera rotation.
//                   Follows the Hamilton convention: q = [qw, qx, qy, qz].
//   std_rx, std_ry, std_rz
//                 - Standard deviations of rotation prior in radians.
//                   Represent uncertainties in the axis-angle representation
//                   around x, y, z axes respectively.
//
int RunPosePriorImporter(int argc, char** argv) {
  std::string input_path;
  bool clear_existing = false;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption(
      "input_path",
      &input_path,
      "Pose prior input file supportted columns:\n"
      "# name x y z cs std_x std_y std_z qw qx qy qz std_rx std_ry std_rz\n");
  options.AddDefaultOption("clear_existing",
                           &clear_existing,
                           "Remove all existing pose priors from the database "
                           "before importing new ones.");
  options.Parse(argc, argv);

  Database database(*options.database_path);

  std::ifstream infile(input_path);
  if (!infile.is_open()) {
    LOG(ERROR) << "Could not open file: " << input_path;
    return EXIT_FAILURE;
  }

  std::unordered_map<std::string, size_t> col_idx;
  std::vector<std::string> column_names;
  std::string line;

  // Read header
  while (std::getline(infile, line)) {
    if (line.empty() || line[0] != '#') continue;
    std::istringstream ss(line.substr(1));
    std::string field;
    size_t idx = 0;
    while (ss >> field) {
      col_idx[field] = idx++;
      column_names.push_back(field);
    }
    break;
  }

  if (col_idx.empty() || col_idx.find("name") == col_idx.end()) {
    LOG(ERROR) << "Invalid or missing header line with # name ...";
    return EXIT_FAILURE;
  }

  std::unordered_map<std::string, image_t> name_to_id;
  for (const auto& image : database.ReadAllImages()) {
    name_to_id[image.Name()] = image.ImageId();
  }

  size_t num_imported = 0;
  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (ss >> token) tokens.push_back(token);
    if (tokens.size() < 1) continue;

    const std::string& name = tokens[col_idx["name"]];
    if (name_to_id.find(name) == name_to_id.end()) {
      LOG(WARNING) << "Image name not found in database: " << name;
      continue;
    }
    image_t image_id = name_to_id[name];

    PosePrior prior;

    if (HasFields(col_idx, {"x", "y", "z"})) {
      prior.position = Eigen::Vector3d(std::stod(tokens[col_idx["x"]]),
                                       std::stod(tokens[col_idx["y"]]),
                                       std::stod(tokens[col_idx["z"]]));
    }

    if (HasFields(col_idx, {"cs"})) {
      std::string system = tokens[col_idx["cs"]];
      StringToUpper(&system);
      prior.coordinate_system = PosePrior::CoordinateSystemFromString(system);
    } else {
      // Default to CARTESIAN if no coordinate system is specified
      prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
    }

    if (HasFields(col_idx, {"std_x", "std_y", "std_z"})) {
      Eigen::Vector3d stddev(std::stod(tokens[col_idx["std_x"]]),
                             std::stod(tokens[col_idx["std_y"]]),
                             std::stod(tokens[col_idx["std_z"]]));
      prior.position_covariance = stddev.cwiseProduct(stddev).asDiagonal();
    }

    if (HasFields(col_idx, {"qw", "qx", "qy", "qz"})) {
      prior.rotation = Eigen::Quaterniond(std::stod(tokens[col_idx["qw"]]),
                                          std::stod(tokens[col_idx["qx"]]),
                                          std::stod(tokens[col_idx["qy"]]),
                                          std::stod(tokens[col_idx["qz"]]))
                           .normalized();
    }

    if (HasFields(col_idx, {"std_rx", "std_ry", "std_rz"})) {
      Eigen::Vector3d stddev(std::stod(tokens[col_idx["std_rx"]]),
                             std::stod(tokens[col_idx["std_ry"]]),
                             std::stod(tokens[col_idx["std_rz"]]));
      prior.rotation_covariance = stddev.cwiseProduct(stddev).asDiagonal();
    }

    if (!database.ExistsPosePrior(image_id)) {
      database.WritePosePrior(image_id, prior);
    } else {
      LOG(WARNING) << "Overwriting exsiting pose prior for image #" << image_id;
      database.UpdatePosePrior(image_id, prior);
    }

    num_imported++;
  }

  LOG(INFO) << "Imported pose priors for " << num_imported
            << " images with fields: " << VectorToCSV(column_names);
  return EXIT_SUCCESS;
}

}  // namespace colmap
