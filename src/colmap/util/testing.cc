// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/util/testing.h"

#include "colmap/util/logging.h"

#include <filesystem>
#include <mutex>
#include <set>

#include <gtest/gtest.h>

namespace colmap {

std::string CreateTestDir() {
  const testing::TestInfo* test_info = THROW_CHECK_NOTNULL(
      testing::UnitTest::GetInstance()->current_test_info());
  std::ostringstream test_name_stream;
  test_name_stream << test_info->test_suite_name() << "." << test_info->name();
  const std::string test_name = test_name_stream.str();

  const std::filesystem::path test_dir =
      std::filesystem::path("colmap_test_tmp_test_data") / test_name;

  // Create directory once. Cleanup artifacts from previous test runs.
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  static std::set<std::string> existing_test_names;
  if (existing_test_names.count(test_name) == 0) {
    if (std::filesystem::is_directory(test_dir)) {
      std::filesystem::remove_all(test_dir);
    }
    std::filesystem::create_directories(test_dir);
  }
  existing_test_names.insert(test_name);

  return test_dir.string();
}

}  // namespace colmap
