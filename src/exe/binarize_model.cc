
#include "base/reconstruction.h"
#include "base/serialization.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/timer.h"
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string import_path;
  std::string export_path;

  OptionManager options;
  options.AddRequiredOption("import_path", &import_path);
  options.Parse(argc, argv);

  if (!ExistsDir(import_path)) {
    std::cerr << "ERROR: `import_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  PrintHeading2("Loading text-based reconstruction.");
  Reconstruction reconstruction;
  {
    Timer text_read;
    text_read.Start();
    reconstruction.Read(import_path);
    text_read.Pause();
    text_read.PrintSeconds();
  }

  PrintHeading2("Serializing reconstruction to " +
                import_path + "/reconstruction.bin");
  {
    Timer binary_write;
    binary_write.Start();
    WriteToBinaryFile(import_path + "/reconstruction.bin", reconstruction);
    binary_write.Pause();
    binary_write.PrintSeconds();
  }

  return EXIT_SUCCESS;
}
