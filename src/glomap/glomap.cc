#include "colmap/util/logging.h"

#include "glomap/exe/global_mapper.h"
#include "glomap/exe/rotation_averager.h"

#include <iostream>

namespace {

typedef std::function<int(int, char**)> command_func_t;
int ShowHelp(
    const std::vector<std::pair<std::string, command_func_t>>& commands) {
  std::cout << "GLOMAP -- Global Structure-from-Motion\n\n";

#ifdef GLOMAP_CUDA_ENABLED
  std::cout << "This version was compiled with CUDA!\n\n";
#else
  std::cout << "This version was NOT compiled CUDA!\n\n";
#endif

  std::cout << "Usage:\n";
  std::cout << "  glomap mapper --database_path DATABASE --output_path MODEL\n";
  std::cout << "  glomap mapper_resume --input_path MODEL_INPUT --output_path "
               "MODEL_OUTPUT\n";

  std::cout << "Available commands:\n";
  std::cout << "  help\n";
  for (const auto& command : commands) {
    std::cout << "  " << command.first << '\n';
  }
  std::cout << '\n';

  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);
  FLAGS_alsologtostderr = true;

  std::vector<std::pair<std::string, command_func_t>> commands;
  commands.emplace_back("mapper", &glomap::RunGlobalMapper);
  commands.emplace_back("mapper_resume", &glomap::RunGlobalMapperResume);
  commands.emplace_back("rotation_averager", &glomap::RunRotationAverager);

  if (argc == 1) {
    return ShowHelp(commands);
  }

  const std::string command = argv[1];
  if (command == "help" || command == "-h" || command == "--help") {
    return ShowHelp(commands);
  } else {
    command_func_t matched_command_func = nullptr;
    for (const auto& command_func : commands) {
      if (command == command_func.first) {
        matched_command_func = command_func.second;
        break;
      }
    }
    if (matched_command_func == nullptr) {
      std::cout << "Command " << command << " not recognized. "
                << "To list the available commands, run `glomap help`." << '\n';
      return EXIT_FAILURE;
    } else {
      int command_argc = argc - 1;
      char** command_argv = &argv[1];
      command_argv[0] = argv[0];
      return matched_command_func(command_argc, command_argv);
    }
  }

  return ShowHelp(commands);
}
