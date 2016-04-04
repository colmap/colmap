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
      "import_path",
      config::value<std::string>(&import_path)->default_value(""));
  options.desc->add_options()(
      "export_path", config::value<std::string>(&export_path)->required());

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

  IncrementalMapperController* mapper_controller =
      new IncrementalMapperController(options);

  if (import_path != "") {
    if (!boost::filesystem::is_directory(import_path)) {
      std::cerr << "ERROR: `import_path` is not a directory." << std::endl;
      return EXIT_FAILURE;
    }

    const size_t model_idx = mapper_controller->AddModel();
    mapper_controller->Model(model_idx).Read(import_path);
  }

  mapper_controller->start();
  mapper_controller->wait();

  export_path = EnsureTrailingSlash(export_path);

  for (size_t i = 0; i < mapper_controller->NumModels(); ++i) {
    const std::string model_path = export_path + std::to_string(i);

    if (!boost::filesystem::is_directory(model_path)) {
      boost::filesystem::create_directory(model_path);
    }

    options.Write(model_path + "/project.ini");
    mapper_controller->Model(i).Write(model_path);
  }

  return EXIT_SUCCESS;
}
