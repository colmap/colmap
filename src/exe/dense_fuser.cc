// COLMAP - Structure-from-Motion and Multi-View Stereo.
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

#include "mvs/fusion.h"
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char* argv[]) {
  InitializeGlog(argv);

  std::string workspace_path;
  std::string input_type;
  std::string workspace_format;
  std::string output_path;
  std::string config_path;

  OptionManager options;
  options.AddDenseMapperOptions();
  options.AddRequiredOption("workspace_path", &workspace_path);
  options.AddRequiredOption("workspace_format", &workspace_format);
  options.AddRequiredOption("input_type", &input_type);
  options.AddRequiredOption("output_path", &output_path);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  if (input_type != "photometric" && input_type != "geometric") {
    std::cout << "ERROR: Invalid input type - supported values are "
                 "'photometric' and 'geometric'."
              << std::endl;
    return EXIT_FAILURE;
  }

  mvs::StereoFusion fuser(options.dense_mapper_options->fusion, workspace_path,
                          workspace_format, input_type);

  fuser.Start();
  fuser.Wait();

  std::cout << "Writing output: " << output_path << std::endl;
  WritePlyBinary(output_path, fuser.GetFusedPoints());

  return EXIT_SUCCESS;
}
