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

#include "base/database.h"
#include "util/misc.h"
#include "util/option_manager.h"

namespace colmap {

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
