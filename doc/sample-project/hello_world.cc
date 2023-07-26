#include <cstdlib>
#include <iostream>

#include <colmap/controllers/option_manager.h>
#include <colmap/util/string.h>

int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);

  std::string input_path;
  std::string output_path;

  colmap::OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.Parse(argc, argv);

  std::cout << colmap::StringPrintf("Hello %s!", "COLMAP") << std::endl;

  return EXIT_SUCCESS;
}
