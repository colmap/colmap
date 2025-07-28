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
// Each line specifies the pose prior for one image. Fields may include
// position, translation, rotation, and their corresponding uncertainties.
// Missing or unknown values can be omitted or marked as NaN.
//
// Columns:
//   name                  - Image name, must match the image name in the
//                           database.
//   cs                    - World coordinate system: WGS84 or CARTESIAN.
//
//   px py pz              - Camera position in the world coordinate system.
//                           If cs = WGS84, the order is longitude, latitude and
//                           altitude
//   stddev_px stddev_py stddev_pz
//                         - Standard deviations of the position prior (in world
//                           frame).
//
//   tx ty tz              - Camera-from-world translation vector.
//                           Cannot be used together with px, py, pz.
//   stddev_tx stddev_ty stddev_tz
//                         - Standard deviations of the translation prior.
//
//   qw qx qy qz           - Quaternion representing camera-from-world rotation.
//                           Only valid if cs = CARTESIAN.
//   stddev_rx stddev_ry stddev_rz
//                         - Standard deviations of rotation in radians.
//                           Represent uncertainty in the axis-angle
//                           representation around x, y, z axes respectively.
//
// Example 1: WGS84 with position uncertainty
//   # name cs px py pz stddev_px stddev_py stddev_pz
//   image1.jpg WGS84 48.1476954472 11.5695882694 561.1509 0.1 0.1 0.5
//
// Example 2: CARTESIAN with position and rotation priors:
//   # name cs px py pz stddev_px stddev_py stddev_pz qw qx qy qz stddev_rx
//   stddev_ry stddev_rz
//   image2.jpg CARTESIAN -1.0 2.0 0.5 0.2 0.2 0.2 0.7071 0.7071 0.0 0.0 0.02
//   0.02 0.02
//
// Example 3: CARTESIAN with translation and rotation priors:
//   # name cs tx ty tz stddev_tx stddev_ty stddev_tz qw qx qy qz stddev_rx
//   stddev_ry stddev_rz
//   image3.jpg CARTESIAN -1.0 2.0 0.5 0.2 0.2 0.2 0.7071 0.7071 0.0 0.0 0.02
//   0.02 0.02
//
// Example 4: CARTESIAN with rotation priors and invalid translation:
//   # name cs tx ty tz stddev_tx stddev_ty stddev_tz qw qx qy qz stddev_rx
//   stddev_ry stddev_rz
//   image4.jpg CARTESIAN NaN NaN NaN NaN NaN NaN 0.7071 0.7071 0.0 0.0 0.02
//   0.02 0.02
//
int RunPosePriorImporter(int argc, char** argv) {
  std::string input_path;
  bool clear_existing = false;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("input_path",
                            &input_path,
                            "Pose prior input file supported columns:\n"
                            "# name cs px py pz stddev_px stddev_py stddev_pz\n"
                            "tx ty tz stddev_tx stddev_ty stddev_tz\n"
                            "qw qx qy qz stddev_rx stddev_ry stddev_rz");
  options.AddDefaultOption("clear_existing",
                           &clear_existing,
                           "Remove all existing pose priors from the database "
                           "before importing new ones.");
  options.Parse(argc, argv);

  Database database(*options.database_path);

  if (clear_existing) {
    database.ClearPosePriors();
  }

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

  const bool has_coordinate_system = col_idx.count("cs");
  const bool has_position = HasFields(col_idx, {"px", "py", "pz"});
  const bool has_position_stddev =
      HasFields(col_idx, {"stddev_px", "stddev_py", "stddev_pz"});
  const bool has_translation = HasFields(col_idx, {"tx", "ty", "tz"});
  const bool has_translation_stddev =
      HasFields(col_idx, {"stddev_tx", "stddev_ty", "stddev_tz"});
  const bool has_rotation = HasFields(col_idx, {"qw", "qx", "qy", "qz"});
  const bool has_rotation_stddev =
      HasFields(col_idx, {"stddev_rx", "stddev_ry", "stddev_rz"});

  if (has_position && has_translation) {
    LOG(ERROR) << "Both position and translation provided";
    return EXIT_FAILURE;
  }

  std::unordered_map<std::string, image_t> name_to_id;
  for (const auto& image : database.ReadAllImages()) {
    const image_t image_id = image.ImageId();
    name_to_id[image.Name()] = image_id;
  }

  size_t num_imported = 0;

  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::vector<std::string> tokens;
    std::istringstream ss(line);
    std::copy(std::istream_iterator<std::string>(ss),
              std::istream_iterator<std::string>(),
              std::back_inserter(tokens));

    if (tokens.size() < col_idx.size()) {
      LOG(WARNING) << "Skipping line (insufficient fields): " << line;
      continue;
    }

    const std::string& name = tokens[col_idx["name"]];
    const auto name_iter = name_to_id.find(name);
    if (name_iter == name_to_id.end()) {
      LOG(WARNING) << "Skipping image not found in database: " << name;
      continue;
    }
    const image_t image_id = name_iter->second;

    const bool prior_exists = database.ExistsPosePrior(image_id);
    PosePrior prior =
        prior_exists ? database.ReadPosePrior(image_id) : PosePrior();

    if (has_coordinate_system) {
      std::string system = tokens[col_idx["cs"]];
      StringToUpper(&system);
      prior.coordinate_system = PosePrior::CoordinateSystemFromString(system);
    } else {
      prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
    }

    if (has_position) {
      prior.SetPosition({std::stod(tokens[col_idx["px"]]),
                         std::stod(tokens[col_idx["py"]]),
                         std::stod(tokens[col_idx["pz"]])});
    } else if (has_translation) {
      prior.SetTranslation({std::stod(tokens[col_idx["tx"]]),
                            std::stod(tokens[col_idx["ty"]]),
                            std::stod(tokens[col_idx["tz"]])});
    }
    if (has_position_stddev) {
      const Eigen::Vector3d stddev{std::stod(tokens[col_idx["stddev_px"]]),
                                   std::stod(tokens[col_idx["stddev_py"]]),
                                   std::stod(tokens[col_idx["stddev_pz"]])};
      prior.SetPositionCovariance(stddev.cwiseProduct(stddev).asDiagonal());
    } else if (has_translation_stddev) {
      const Eigen::Vector3d stddev{std::stod(tokens[col_idx["stddev_tx"]]),
                                   std::stod(tokens[col_idx["stddev_ty"]]),
                                   std::stod(tokens[col_idx["stddev_tz"]])};
      prior.SetTranslationCovariance(stddev.cwiseProduct(stddev).asDiagonal());
    }

    if (has_rotation) {
      if (prior.coordinate_system == PosePrior::CoordinateSystem::WGS84) {
        LOG(WARNING) << "Ignoring rotation for WGS84 image: " << name;
      } else {
        prior.SetRotationFromCoeffs({std::stod(tokens[col_idx["qx"]]),
                                     std::stod(tokens[col_idx["qy"]]),
                                     std::stod(tokens[col_idx["qz"]]),
                                     std::stod(tokens[col_idx["qw"]])});
      }
    }

    if (has_rotation_stddev) {
      if (prior.coordinate_system == PosePrior::CoordinateSystem::WGS84) {
        LOG(WARNING) << "Ignoring rotation uncertainty for WGS84 image: "
                     << name;
      } else {
        const Eigen::Vector3d stddev{std::stod(tokens[col_idx["stddev_rx"]]),
                                     std::stod(tokens[col_idx["stddev_ry"]]),
                                     std::stod(tokens[col_idx["stddev_rz"]])};
        prior.SetRotationCovariance(stddev.cwiseProduct(stddev).asDiagonal());
      }
    }

    if (prior_exists) {
      database.UpdatePosePrior(image_id, prior);
    } else {
      database.WritePosePrior(image_id, prior);
    }

    num_imported++;
  }

  LOG(INFO) << "Imported pose priors for " << num_imported
            << " images with fields: " << VectorToCSV(column_names);
  return EXIT_SUCCESS;
}

}  // namespace colmap
