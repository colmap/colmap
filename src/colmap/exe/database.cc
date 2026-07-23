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
  std::string unknown_column_policy_str = "error";

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("pose_prior_path", &pose_prior_path);
  options.AddRequiredOption(
      "existing",
      &existing_policy,
      "One of error|replace|merge: `error` aborts if any incoming resolved "
      "image already has a prior; `replace` replaces the complete prior for "
      "that image (groups absent from the row become absent); `merge` "
      "updates only the groups present in the row and preserves the rest.");
  options.AddDefaultOption(
      "unknown_column_policy",
      &unknown_column_policy_str,
      "One of error|ignore: `error` (default) fails the import if the "
      "archive's schema contains a column name this build does not "
      "recognize; `ignore` discards that column's cells (preserving row "
      "width) and logs one warning naming every ignored column. Forward "
      "compatibility with a producer's extra columns is opt-in, not "
      "automatic.");
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (existing_policy != "error" && existing_policy != "replace" &&
      existing_policy != "merge") {
    LOG(ERROR) << "`existing` must be one of error|replace|merge, got: `"
               << existing_policy << "`";
    return EXIT_FAILURE;
  }

  PosePriorArchiveReadOptions read_options;
  if (unknown_column_policy_str == "error") {
    read_options.unknown_column_policy = UnknownColumnPolicy::ERROR;
  } else if (unknown_column_policy_str == "ignore") {
    read_options.unknown_column_policy = UnknownColumnPolicy::IGNORE;
  } else {
    LOG(ERROR) << "`unknown_column_policy` must be one of error|ignore, got: `"
               << unknown_column_policy_str << "`";
    return EXIT_FAILURE;
  }

  if (!ExistsFile(pose_prior_path)) {
    LOG(ERROR) << "`pose_prior_path` is not a file.";
    return EXIT_FAILURE;
  }

  const auto archive = ReadPosePriorArchive(pose_prior_path, read_options);

  auto database = Database::Open(*options.database_path);

  const auto data_id_from_name =
      [&database](const std::string& name) -> std::optional<data_t> {
    const auto image = database->ReadImageWithName(name);
    if (!image) {
      return std::nullopt;
    }
    return data_t(sensor_t(SensorType::CAMERA, image->CameraId()),
                  image->ImageId());
  };

  THROW_CHECK(!archive.HasDuplicateResolvedNames(data_id_from_name))
      << "Archive names the same resolved image more than once";

  if (existing_policy == "merge") {
    auto priors = database->ReadAllPosePriors();
    const size_t num_existing = priors.size();

    // Resolve every incoming row *before* mutating anything, so only rows
    // that actually target an existing prior cause a database write to it.
    // Duplicate resolved names are already rejected above, so each
    // resolved data_t appears in at most one row.
    NodeHashMap<data_t, size_t> existing_index_by_data_id;
    for (size_t i = 0; i < priors.size(); ++i) {
      existing_index_by_data_id.emplace(priors[i].corr_data_id, i);
    }
    const std::vector<std::optional<data_t>> resolved_data_ids =
        archive.ResolveRowDataIds(data_id_from_name);

    const size_t rows_read = archive.NumRows();
    size_t rows_unresolved = 0;
    FlatHashSet<size_t> touched_existing_indices;
    for (const auto& data_id : resolved_data_ids) {
      if (!data_id.has_value()) {
        ++rows_unresolved;
        continue;
      }
      const auto it = existing_index_by_data_id.find(*data_id);
      if (it != existing_index_by_data_id.end()) {
        touched_existing_indices.insert(it->second);
      }
    }
    const size_t rows_resolved = rows_read - rows_unresolved;

    archive.UpdatePosePriors(
        data_id_from_name, /*allow_new_priors=*/true, priors);

    size_t rows_added = 0;
    size_t rows_updated = 0;
    {
      DatabaseTransaction transaction(database.get());
      for (size_t i = 0; i < priors.size(); ++i) {
        if (i < num_existing) {
          // Only rewrite existing rows an incoming resolved row actually
          // targeted; every other existing prior is left byte-for-byte
          // unchanged in the database.
          if (touched_existing_indices.count(i) > 0) {
            database->UpdatePosePrior(priors[i]);
            ++rows_updated;
          }
        } else {
          database->WritePosePrior(priors[i]);
          ++rows_added;
        }
      }
    }

    LOG(INFO) << StringPrintf(
        "Pose-prior merge: rows_read=%zu, rows_resolved=%zu, "
        "rows_unresolved=%zu, rows_added=%zu, rows_updated=%zu",
        rows_read,
        rows_resolved,
        rows_unresolved,
        rows_added,
        rows_updated);
    return EXIT_SUCCESS;
  }

  // existing_policy == "error" or "replace": each row's full prior replaces
  // any existing prior for that image (fresh PosePrior with only the row's
  // groups set); the two policies differ only in whether an existing prior
  // is permitted at all.
  auto priors = archive.ToPosePriors(data_id_from_name);
  if (priors.empty()) {
    LOG(WARNING) << "No pose priors were imported.";
    return EXIT_FAILURE;
  }

  // We cannot use ExistsPosePrior(pose_prior_t pose_prior_id) here
  NodeHashMap<data_t, pose_prior_t> existing_prior_ids;
  for (const auto& prior : database->ReadAllPosePriors()) {
    existing_prior_ids.emplace(prior.corr_data_id, prior.pose_prior_id);
  }

  if (existing_policy == "error") {
    for (const auto& prior : priors) {
      THROW_CHECK(existing_prior_ids.find(prior.corr_data_id) ==
                  existing_prior_ids.end())
          << "A pose prior already exists for a resolved image and "
             "`existing=error` was specified; the database was not "
             "modified";
    }
  }

  size_t num_imported = 0;
  {
    DatabaseTransaction transaction(database.get());
    for (auto& prior : priors) {
      const auto it = existing_prior_ids.find(prior.corr_data_id);
      if (it != existing_prior_ids.end()) {
        prior.pose_prior_id = it->second;
        database->UpdatePosePrior(prior);
      } else {
        database->WritePosePrior(prior);
      }
      ++num_imported;
    }
  }

  LOG(INFO) << "Imported " << num_imported << " pose priors.";
  return EXIT_SUCCESS;
}

}  // namespace colmap
