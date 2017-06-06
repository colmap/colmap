// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "mvs/patch_match.h"
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char* argv[]) {
  InitializeGlog(argv);

  std::string workspace_path;
  std::string workspace_format = "COLMAP";
  std::string pmvs_option_name = "option-all";

  OptionManager options;
  options.AddRequiredOption("workspace_path", &workspace_path);
  options.AddDefaultOption("workspace_format", &workspace_format,
                           "{COLMAP, PMVS}");
  options.AddDefaultOption("pmvs_option_name", &pmvs_option_name);
  options.AddDenseStereoOptions();
  options.Parse(argc, argv);

  StringToLower(&workspace_format);
  if (workspace_format != "colmap" && workspace_format != "pmvs") {
    std::cout << "ERROR: Invalid `workspace_format` - supported values are "
                 "'COLMAP' or 'PMVS'."
              << std::endl;
    return EXIT_FAILURE;
  }

  mvs::PatchMatchController controller(*options.dense_stereo, workspace_path,
                                       workspace_format, pmvs_option_name);

  controller.Start();
  controller.Wait();

  return EXIT_SUCCESS;
}
