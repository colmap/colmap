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

#include "base/reconstruction.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string input_path;
  std::string output_path;
  std::string output_type;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("output_type", &output_type,
                            "{'NVM', 'Bundler', 'VRML', 'PLY'}");

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  if (output_type == "NVM") {
    reconstruction.ExportNVM(output_path);
  } else if (output_type == "Bundler") {
    reconstruction.ExportBundler(output_path + ".bundle.out",
                                 output_path + ".list.txt");
  } else if (output_type == "PLY") {
    reconstruction.ExportPLY(output_path);
  } else if (output_type == "VRML") {
    const auto base_path = output_path.substr(0, output_path.find_last_of("."));
    reconstruction.ExportVRML(base_path + ".images.wrl",
                              base_path + ".points3D.wrl", 1,
                              Eigen::Vector3d(1, 0, 0));
  } else {
    std::cerr << "ERROR: Invalid `output_type`" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
