// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/testing.h"

#include "colmap/util/logging.h"

#include <mutex>
#include <set>

#include <gtest/gtest.h>

namespace colmap {

std::filesystem::path CreateTestDir() {
  const testing::TestInfo* test_info = THROW_CHECK_NOTNULL(
      testing::UnitTest::GetInstance()->current_test_info());
  std::ostringstream test_name_stream;
  test_name_stream << test_info->test_suite_name() << "." << test_info->name();
  const std::string test_name = test_name_stream.str();

  const std::filesystem::path test_dir =
      std::filesystem::temp_directory_path() / "colmap_test_data" / test_name;
  LOG(INFO) << "Creating test directory: " << test_dir;

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

  return test_dir;
}

}  // namespace colmap
