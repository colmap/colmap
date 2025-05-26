#include <cstdlib>
#include <iostream>

#include <colmap/controllers/option_manager.h>
#include <colmap/util/string.h>

int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);

  std::string message;
  colmap::OptionManager options;
  options.AddRequiredOption("message", &message);
  options.Parse(argc, argv);

  std::cout << colmap::StringPrintf("Hello %s!\n", message.c_str());

  return EXIT_SUCCESS;
}
