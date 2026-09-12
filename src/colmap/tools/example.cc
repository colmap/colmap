// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/option_manager.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"

// Simple example that reads and writes a reconstruction.
int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::filesystem::path input_path;
  std::filesystem::path output_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);
  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}
