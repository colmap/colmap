
#include "base/reconstruction.h"
#include "base/serialization.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/timer.h"
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string reconstruction_path;
  std::string export_path;

  OptionManager options;
  options.AddRequiredOption("reconstruction_path", &reconstruction_path);
  options.AddRequiredOption("export_path", &export_path);
  options.Parse(argc, argv);

  if (!ExistsFile(reconstruction_path)) {
    std::cerr << "ERROR: `reconstruction_path` is not a file\n";
  }

  if (!ExistsDir(export_path)) {
    std::cerr << "ERROR: `export_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  PrintHeading2("Loading binary reconstruction.");
  Reconstruction reconstruction;
  {
    Timer binary_read;
    binary_read.Start();
    ReadFromBinaryFile(reconstruction_path, &reconstruction);
    binary_read.Pause();
    binary_read.PrintSeconds();
  }

  PrintHeading2("Writing reconstruction to " + export_path);
  {
    Timer text_write;
    text_write.Start();
    reconstruction.Write(export_path);
    text_write.Pause();
    text_write.PrintSeconds();
  }

  return EXIT_SUCCESS;
}
