// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/misc.h"

#include "colmap/util/logging.h"

#include <cstdarg>

namespace colmap {

void RemoveCommandLineArgument(const std::string& arg, int* argc, char** argv) {
  for (int i = 0; i < *argc; ++i) {
    if (argv[i] == arg) {
      for (int j = i + 1; j < *argc; ++j) {
        argv[i] = argv[j];
      }
      *argc -= 1;
      break;
    }
  }
}

}  // namespace colmap
