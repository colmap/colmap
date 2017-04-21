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

#include "controllers/automatic_reconstruction.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  AutomaticReconstructionController::Options reconstruction_options;
  std::string data_type;

  OptionManager options;
  options.AddRequiredOption("workspace_path",
                            &reconstruction_options.workspace_path);
  options.AddRequiredOption("image_path", &reconstruction_options.image_path);
  options.AddRequiredOption("vocab_tree_path",
                            &reconstruction_options.vocab_tree_path);
  options.AddRequiredOption("data_type", &data_type, "{VIDEO, DSLR, INTERNET}");
  options.AddDefaultOption("sparse", &reconstruction_options.sparse);
  options.AddDefaultOption("dense", &reconstruction_options.dense);
  options.AddDefaultOption("use_gpu", &reconstruction_options.use_gpu);
  options.AddDefaultOption("use_opengl", &reconstruction_options.use_opengl);
  options.Parse(argc, argv);

  if (data_type == "VIDEO") {
    reconstruction_options.data_type =
        AutomaticReconstructionController::DataType::VIDEO;
  } else if (data_type == "DSLR") {
    reconstruction_options.data_type =
        AutomaticReconstructionController::DataType::DSLR;
  } else if (data_type == "INTERNET") {
    reconstruction_options.data_type =
        AutomaticReconstructionController::DataType::INTERNET;
  } else {
    LOG(FATAL) << "Invalid data type";
  }

  AutomaticReconstructionController controller(reconstruction_options);
  controller.Start();
  controller.Wait();

  return EXIT_SUCCESS;
}
