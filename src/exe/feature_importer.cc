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

#include <QtGui>

#include "base/camera_models.h"
#include "base/feature_extraction.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

using namespace colmap;

namespace config = boost::program_options;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string import_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddExtractionOptions();
  options.desc->add_options()(
      "import_path", config::value<std::string>(&import_path)->required());

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  const std::vector<double> camera_params =
      CSVToVector<double>(options.extraction_options->camera_params);
  const int camera_model_id =
      CameraModelNameToId(options.extraction_options->camera_model);

  if (camera_params.size() > 0 &&
      !CameraModelVerifyParams(camera_model_id, camera_params)) {
    std::cerr << "ERROR: Invalid camera parameters" << std::endl;
    return EXIT_FAILURE;
  }

  FeatureImporter* feature_importer = new FeatureImporter(
      options.extraction_options->Options(), *options.database_path,
      *options.image_path, import_path);

  feature_importer->start();
  feature_importer->wait();

  return EXIT_SUCCESS;
}
