// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/util/memory.h"

#if defined(_WIN32)
#include <windows.h>
// psapi.h must be included after windows.h.
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>

#include <cstdio>
#endif

namespace colmap {

size_t GetPeakRSSBytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS info;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
    return static_cast<size_t>(info.PeakWorkingSetSize);
  }
  return 0;
#elif defined(__APPLE__)
  // resident_size_max is the high-water mark of the resident set size.
  mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(),
                MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info),
                &count) == KERN_SUCCESS) {
    return static_cast<size_t>(info.resident_size_max);
  }
  return 0;
#elif defined(__linux__)
  // ru_maxrss is reported in kilobytes on Linux.
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    return static_cast<size_t>(usage.ru_maxrss) * 1024;
  }
  return 0;
#else
  return 0;
#endif
}

size_t GetCurrentRSSBytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS info;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
    return static_cast<size_t>(info.WorkingSetSize);
  }
  return 0;
#elif defined(__APPLE__)
  mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(),
                MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info),
                &count) == KERN_SUCCESS) {
    return static_cast<size_t>(info.resident_size);
  }
  return 0;
#elif defined(__linux__)
  // Parse the resident pages from /proc/self/statm (field 2) and scale by the
  // page size.
  FILE* file = std::fopen("/proc/self/statm", "r");
  if (file == nullptr) {
    return 0;
  }
  long rss_pages = 0;
  if (std::fscanf(file, "%*s%ld", &rss_pages) != 1) {
    std::fclose(file);
    return 0;
  }
  std::fclose(file);
  return static_cast<size_t>(rss_pages) *
         static_cast<size_t>(sysconf(_SC_PAGESIZE));
#else
  return 0;
#endif
}

}  // namespace colmap
