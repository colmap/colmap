// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/misc.h"

#include "colmap/util/logging.h"

#include <cstdarg>
#include <fstream>

#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

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

std::optional<size_t> GetAvailableMemoryBytes() {
#if defined(__linux__)
  std::ifstream meminfo("/proc/meminfo");
  std::string key;
  size_t value_kb = 0;
  std::string unit;
  while (meminfo >> key >> value_kb) {
    std::getline(meminfo, unit);
    if (key == "MemAvailable:") {
      return value_kb * 1024ULL;
    }
  }
  const long pages = sysconf(_SC_AVPHYS_PAGES);
  const long page_size = sysconf(_SC_PAGESIZE);
  if (pages > 0 && page_size > 0) {
    return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
  }
  return std::nullopt;
#elif defined(__APPLE__)
  vm_size_t page_size = 0;
  if (host_page_size(mach_host_self(), &page_size) != KERN_SUCCESS) {
    return std::nullopt;
  }
  vm_statistics64_data_t vm_stats{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(),
                        HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&vm_stats),
                        &count) != KERN_SUCCESS) {
    return std::nullopt;
  }
  return static_cast<size_t>(vm_stats.free_count + vm_stats.inactive_count) *
         static_cast<size_t>(page_size);
#elif defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (!GlobalMemoryStatusEx(&status)) {
    return std::nullopt;
  }
  return static_cast<size_t>(status.ullAvailPhys);
#else
  return std::nullopt;
#endif
}

}  // namespace colmap
