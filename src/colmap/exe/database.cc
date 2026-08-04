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
#include "colmap/geometry/pose_prior_io.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/rig.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace colmap {

int RunDatabaseCleaner(int argc, char** argv) {
  std::string type;

  OptionManager options;
  options.AddRequiredOption(
      "type", &type, "{all, images, features, matches, two_view_geometries}");
  options.AddDatabaseOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  StringToLower(&type);
  auto database = Database::Open(*options.database_path);

  DatabaseTransaction transaction(database.get());
  if (type == "all") {
    LOG(INFO) << "Clearing all tables";
    database->ClearAllTables();
  } else if (type == "images") {
    LOG(INFO) << "Clearing images and all dependent tables";
    database->ClearImages();
    database->ClearMatches();
    database->ClearTwoViewGeometries();
  } else if (type == "features") {
    LOG(INFO) << "Clearing features, matches, and two-view geometries";
    database->ClearDescriptors();
    database->ClearKeypoints();
    database->ClearMatches();
    database->ClearTwoViewGeometries();
  } else if (type == "matches") {
    LOG(INFO) << "Clearing matches and two-view geometries";
    database->ClearMatches();
    database->ClearTwoViewGeometries();
  } else if (type == "two_view_geometries") {
    LOG(INFO) << "Clearing two-view geometries";
    database->ClearTwoViewGeometries();
  } else {
    LOG(ERROR) << "Invalid cleanup type; no changes in database";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int RunDatabaseCreator(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  auto database = Database::Open(*options.database_path);

  return EXIT_SUCCESS;
}

int RunDatabaseMerger(int argc, char** argv) {
  std::filesystem::path database_path1;
  std::filesystem::path database_path2;
  std::filesystem::path merged_database_path;

  OptionManager options;
  options.AddRequiredOption("database_path1", &database_path1);
  options.AddRequiredOption("database_path2", &database_path2);
  options.AddRequiredOption("merged_database_path", &merged_database_path);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (ExistsFile(merged_database_path)) {
    LOG(ERROR) << "Merged database file must not exist.";
    return EXIT_FAILURE;
  }

  auto database1 = Database::Open(database_path1);
  auto database2 = Database::Open(database_path2);
  auto merged_database = Database::Open(merged_database_path);
  Database::Merge(*database1, *database2, merged_database.get());

  return EXIT_SUCCESS;
}

int RunRigConfigurator(int argc, char** argv) {
  std::filesystem::path database_path;
  std::filesystem::path rig_config_path;
  std::filesystem::path input_path;
  std::filesystem::path output_path;

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
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  std::optional<Reconstruction> reconstruction;
  if (!input_path.empty()) {
    reconstruction = std::make_optional<Reconstruction>();
    reconstruction->Read(input_path);
  }

  auto database = Database::Open(database_path);

  ApplyRigConfig(
      ReadRigConfig(rig_config_path),
      *database,
      reconstruction.has_value() ? &reconstruction.value() : nullptr);

  if (reconstruction.has_value() && !output_path.empty()) {
    reconstruction->Write(output_path);
  }

  return EXIT_SUCCESS;
}

int RunPosePriorImporter(int argc, char** argv) {
  std::filesystem::path pose_prior_path;
  std::string existing_policy;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("pose_prior_path", &pose_prior_path);
  options.AddRequiredOption(
      "existing",
      &existing_policy,
      "One of error|replace: `error` aborts if any resolved image already "
      "has a prior; `replace` writes the row's complete prior over it, so a "
      "null gravity or heading group clears any previously stored value. "
      "There is no partial merge: a prior assembled from two archives has no "
      "single provenance, and no way to tell which field came from where.");
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (existing_policy != "error" && existing_policy != "replace") {
    LOG(ERROR) << "`existing` must be one of error|replace, got: `"
               << existing_policy << "`";
    return EXIT_FAILURE;
  }

  if (!ExistsFile(pose_prior_path)) {
    LOG(ERROR) << "`pose_prior_path` is not a file.";
    return EXIT_FAILURE;
  }

  // Validate everything -- the archive, every image binding, and the existing
  // state -- before the first write. An import that fails halfway leaves a
  // database that is neither the old one nor the new one, and nothing records
  // which rows made it in.
  const PosePriorArchive archive = ReadPosePriorArchive(pose_prior_path);

  auto database = Database::Open(*options.database_path);

  std::vector<data_t> data_ids;
  std::vector<std::string> unresolved_names;
  data_ids.reserve(archive.rows.size());
  for (const PosePriorArchive::Row& row : archive.rows) {
    const auto image = database->ReadImageWithName(row.name);
    if (!image) {
      unresolved_names.push_back(row.name);
      continue;
    }
    data_ids.emplace_back(sensor_t(SensorType::CAMERA, image->CameraId()),
                          image->ImageId());
  }
  if (!unresolved_names.empty()) {
    // Report all of them at once. Finding these one run at a time, with a
    // partially-written database after each, is how an operator loses an
    // afternoon to a naming mismatch.
    LOG(ERROR) << unresolved_names.size() << " of " << archive.rows.size()
               << " archive rows name an image that is not in the database; "
                  "nothing was written. Unresolved names: "
               << VectorToCSV(unresolved_names);
    return EXIT_FAILURE;
  }

  NodeHashMap<data_t, pose_prior_t> existing_prior_ids;
  for (const auto& prior : database->ReadAllPosePriors()) {
    existing_prior_ids.emplace(prior.corr_data_id, prior.pose_prior_id);
  }

  if (existing_policy == "error") {
    size_t num_conflicts = 0;
    for (const data_t& data_id : data_ids) {
      if (existing_prior_ids.count(data_id) > 0) {
        ++num_conflicts;
      }
    }
    if (num_conflicts > 0) {
      LOG(ERROR) << num_conflicts
                 << " resolved images already have a pose prior and "
                    "`--existing=error` was given; nothing was written. Pass "
                    "`--existing=replace` to overwrite them.";
      return EXIT_FAILURE;
    }
  }

  std::vector<PosePrior> priors = archive.ToPosePriors(data_ids);

  size_t num_inserted = 0;
  size_t num_replaced = 0;
  {
    DatabaseTransaction transaction(database.get());
    for (PosePrior& prior : priors) {
      const auto it = existing_prior_ids.find(prior.corr_data_id);
      if (it != existing_prior_ids.end()) {
        prior.pose_prior_id = it->second;
        database->UpdatePosePrior(prior);
        ++num_replaced;
      } else {
        database->WritePosePrior(prior);
        ++num_inserted;
      }
    }
  }

  LOG(INFO) << StringPrintf(
      "Pose-prior import: archive_rows=%zu, resolved_images=%zu, "
      "inserted=%zu, replaced=%zu, gravity_rows=%zu, heading_rows=%zu",
      archive.rows.size(),
      data_ids.size(),
      num_inserted,
      num_replaced,
      std::count_if(archive.rows.begin(),
                    archive.rows.end(),
                    [](const PosePriorArchive::Row& row) {
                      return row.gravity.has_value();
                    }),
      std::count_if(archive.rows.begin(),
                    archive.rows.end(),
                    [](const PosePriorArchive::Row& row) {
                      return row.heading_rad.has_value();
                    }));
  return EXIT_SUCCESS;
}

}  // namespace colmap
