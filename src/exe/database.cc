// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "exe/database.h"

#include "base/camera_models.h"
#include "base/database.h"
#include "util/misc.h"
#include "util/option_manager.h"

namespace colmap {

// Updates cameras in database using a CSV file formatted as:
// CAMERA_ID, MODEL_NAME, PARAM_1, PARAM_2, ..., PARAM_N
//
// The CAMERA_ID must already exist in the database (no new cameras added), the
// MODEL_NAME must be one of the valid model names, and the number N of
// parameters provided must match the number of parameters for the specific
// model.
//
// Example:
// 1, SIMPLE_RADIAL, 700, 320, 240, 0.005
// 2, PINHOLE, 700, 680, 320, 240
// 3, OPENCV, 700, 700, 320, 240, 0.0, 0.0, 0.0, 0.0
int RunCameraUpdater(int argc, char** argv) {
  std::string params_path;
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("params_path", &params_path);
  options.Parse(argc, argv);

  if (!ExistsFile(params_path)) {
    std::cout << "WARN: Camera params file not found; skipping update"
              << std::endl;
    return EXIT_SUCCESS;
  }

  PrintHeading1("Updating database cameras from file");
  const std::vector<std::string> lines = ReadTextFileLines(params_path);
  {
    Database database(*options.database_path);
    DatabaseTransaction transaction(&database);
    for (std::string line : lines) {
      const std::vector<std::string> parts = CSVToVector<std::string>(line);
      if (parts.size() < 5) {
        std::cout << "WARN: Malformed camera parameters line: " << line
                  << std::endl;
        continue;
      }

      const camera_t camera_id = (camera_t)std::stoi(parts[0]);
      Camera camera = database.ReadCamera(camera_id);
      if (camera.CameraId() == kInvalidCameraId) {
        std::cout << "WARN: Camera id " << camera_id
                  << " not found in database; skipping line: " << line
                  << std::endl;
        continue;
      }

      const std::string model_name = parts[1];
      if (!ExistsCameraModelWithName(model_name)) {
        std::cout << "WARN: Invalid model name " << model_name
                  << "; skipping line: " << line << std::endl;
        continue;
      }

      camera.SetModelIdFromName(model_name);
      std::vector<double> params;
      for (int i = 2; i < parts.size(); ++i) {
        params.push_back(std::stod(parts[i]));
      }
      camera.SetParams(params);
      camera.SetPriorFocalLength(true);

      if (camera.VerifyParams() && !camera.HasBogusParams(0.1, 10.0, 1.0)) {
        database.UpdateCamera(camera);
      } else {
        std::cout << "WARN: Invalid camera parameters in line: " << line
                  << std::endl;
      }
    }
  }

  PrintHeading2("Cameras updated successfully");
  return EXIT_SUCCESS;
}

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
      std::cout << "ERROR: Invalid cleanup type; no changes in database"
                << std::endl;
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
    std::cout << "ERROR: Merged database file must not exist." << std::endl;
    return EXIT_FAILURE;
  }

  Database database1(database_path1);
  Database database2(database_path2);
  Database merged_database(merged_database_path);
  Database::Merge(database1, database2, &merged_database);

  return EXIT_SUCCESS;
}

}  // namespace colmap
