// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#include <glog/logging.h>

#include "sfm/controllers.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

using namespace colmap;

namespace config = boost::program_options;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string import_path;
  std::string export_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddMapperOptions();
  options.AddBundleAdjustmentOptions();
  options.desc->add_options()(
      "import_path", config::value<std::string>(&import_path)->required());
  options.desc->add_options()(
      "export_path", config::value<std::string>(&export_path)->required());

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  if (!boost::filesystem::is_directory(import_path)) {
    std::cerr << "ERROR: `import_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (!boost::filesystem::is_directory(export_path)) {
    std::cerr << "ERROR: `export_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  google::InitGoogleLogging(argv[0]);

  //////////////////////////////////////////////////////////////////////////////
  // Image registration
  //////////////////////////////////////////////////////////////////////////////

  PrintHeading1("Loading database");

  DatabaseCache database_cache;

  {
    Database database;
    database.Open(*options.database_path);
    Timer timer;
    timer.Start();
    const size_t min_num_matches =
        static_cast<size_t>(options.mapper_options->min_num_matches);
    database_cache.Load(database, min_num_matches,
                        options.mapper_options->ignore_watermarks);
    std::cout << std::endl;
    timer.PrintMinutes();
  }

  std::cout << std::endl;

  Reconstruction reconstruction;
  reconstruction.Read(import_path);

  IncrementalMapper mapper(&database_cache);
  mapper.BeginReconstruction(&reconstruction);

  const IncrementalMapper::Options inc_mapper_options =
      options.mapper_options->IncrementalMapperOptions();

  for (const auto& image : reconstruction.Images()) {
    if (image.second.IsRegistered()) {
      continue;
    }

    PrintHeading1("Registering image #" + std::to_string(image.first) + " (" +
                  std::to_string(reconstruction.NumRegImages() + 1) + ")");

    std::cout << "  => Image sees " << image.second.NumVisiblePoints3D()
              << " / " << image.second.NumObservations() << " points"
              << std::endl;

    mapper.RegisterNextImage(inc_mapper_options, image.first);
  }

  const bool kDiscardReconstruction = false;
  mapper.EndReconstruction(kDiscardReconstruction);

  //////////////////////////////////////////////////////////////////////////////
  // Export models
  //////////////////////////////////////////////////////////////////////////////

  export_path = EnsureTrailingSlash(export_path);
  reconstruction.Write(export_path);

  return EXIT_SUCCESS;
}
