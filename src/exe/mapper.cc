// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <boost/filesystem.hpp>

#include "sfm/controllers.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string import_path;
  std::string export_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddSparseMapperOptions();
  options.AddDefaultOption("import_path", import_path, &import_path);
  options.AddRequiredOption("export_path", &export_path);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  if (!boost::filesystem::is_directory(export_path)) {
    std::cerr << "ERROR: `export_path` is not a directory." << std::endl;
    return EXIT_FAILURE;
  }

  ReconstructionManager reconstruction_manager;
  if (import_path != "") {
    if (!boost::filesystem::is_directory(import_path)) {
      std::cerr << "ERROR: `import_path` is not a directory." << std::endl;
      return EXIT_FAILURE;
    }
    reconstruction_manager.Read(import_path);
  }

  IncrementalMapperController mapper(options, &reconstruction_manager);
  mapper.Start();
  mapper.Wait();

  if (import_path == "") {
    reconstruction_manager.Write(export_path, &options);
  } else {
    CHECK_LE(reconstruction_manager.Size(), 1);
    reconstruction_manager.Get(0).Write(export_path);
  }

  return EXIT_SUCCESS;
}
