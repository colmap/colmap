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

#include "colmap/controllers/base_option_manager.h"

#include "colmap/util/enum_utils.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Test enum for AddAndRegisterDefaultEnumOption tests
MAKE_ENUM_CLASS(TestEnumType, 0, VALUE_A, VALUE_B, VALUE_C);

TEST(BaseOptionManager, Reset) {
  BaseOptionManager options;
  *options.database_path = "/test/path";
  *options.image_path = "/test/images";
  options.AddDatabaseOptions();
  options.AddImageOptions();
  EXPECT_EQ(*options.database_path, "/test/path");
  EXPECT_EQ(*options.image_path, "/test/images");

  options.Reset();

  EXPECT_EQ(*options.database_path, "");
  EXPECT_EQ(*options.image_path, "");
}

TEST(BaseOptionManager, ResetOptions) {
  BaseOptionManager options;
  *options.database_path = "/test/path";
  *options.image_path = "/test/images";

  options.ResetOptions(/*reset_paths=*/true);
  EXPECT_EQ(*options.database_path, "");
  EXPECT_EQ(*options.image_path, "");

  *options.database_path = "/test/path";
  *options.image_path = "/test/images";
  options.ResetOptions(/*reset_paths=*/false);
  EXPECT_EQ(*options.database_path, "/test/path");
  EXPECT_EQ(*options.image_path, "/test/images");
}

TEST(BaseOptionManager, AddOptionsIdempotent) {
  BaseOptionManager options;

  // Adding options multiple times should not cause issues
  options.AddLogOptions();
  options.AddLogOptions();

  options.AddRandomOptions();
  options.AddRandomOptions();

  options.AddDatabaseOptions();
  options.AddDatabaseOptions();

  options.AddImageOptions();
  options.AddImageOptions();

  // If idempotency is not maintained, the above would cause errors
  SUCCEED();
}

TEST(BaseOptionManager, WriteAndRead) {
  const std::string test_dir = CreateTestDir();
  const std::string config_path = test_dir + "/config.ini";

  // Create necessary directories
  CreateDirIfNotExists(test_dir + "/images");

  // Create and configure a BaseOptionManager
  BaseOptionManager options_write;
  options_write.AddDatabaseOptions();
  options_write.AddImageOptions();

  *options_write.database_path = test_dir + "/database.db";
  *options_write.image_path = test_dir + "/images";

  // Write to file
  options_write.Write(config_path);
  EXPECT_TRUE(ExistsFile(config_path));

  // Read from file
  BaseOptionManager options_read;
  options_read.AddDatabaseOptions();
  options_read.AddImageOptions();

  EXPECT_TRUE(options_read.Read(config_path));

  // Verify that values were read correctly
  EXPECT_EQ(*options_read.database_path, *options_write.database_path);
  EXPECT_EQ(*options_read.image_path, *options_write.image_path);
}

TEST(BaseOptionManager, ReRead) {
  const std::string test_dir = CreateTestDir();
  const std::string config_path = test_dir + "/config.ini";

  // Create necessary directories
  CreateDirIfNotExists(test_dir + "/images");

  // Create and write initial config
  BaseOptionManager options_write;
  options_write.AddDatabaseOptions();
  options_write.AddImageOptions();
  *options_write.database_path = test_dir + "/database.db";
  *options_write.image_path = test_dir + "/images";
  options_write.Write(config_path);

  // Read with ReRead
  BaseOptionManager options_read;
  EXPECT_TRUE(options_read.ReRead(config_path));

  // Verify values
  EXPECT_EQ(*options_read.database_path, *options_write.database_path);
  EXPECT_EQ(*options_read.image_path, *options_write.image_path);
}

TEST(BaseOptionManager, ReadNonExistentFile) {
  BaseOptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();

  EXPECT_FALSE(options.Read("/path/that/does/not/exist.ini"));
}

TEST(BaseOptionManager, Check) {
  const std::string test_dir = CreateTestDir();

  BaseOptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();

  // Should fail with non-existent paths
  *options.database_path = test_dir + "/database.db";
  *options.image_path = "/path/that/does/not/exist";
  EXPECT_FALSE(options.Check());

  // Should succeed with valid paths
  CreateDirIfNotExists(test_dir + "/images");
  *options.image_path = test_dir + "/images";
  EXPECT_TRUE(options.Check());
}

TEST(BaseOptionManager, CheckDatabaseParentDir) {
  const std::string test_dir = CreateTestDir();

  BaseOptionManager options;
  options.AddDatabaseOptions();

  // Should succeed when database parent dir exists
  *options.database_path = test_dir + "/database.db";
  EXPECT_TRUE(options.Check());

  // Should fail when database path is a directory
  CreateDirIfNotExists(test_dir + "/bad_database");
  *options.database_path = test_dir + "/bad_database";
  EXPECT_FALSE(options.Check());
}

TEST(BaseOptionManager, ParseWithOptions) {
  const std::string test_dir = CreateTestDir();
  CreateDirIfNotExists(test_dir + "/images");

  BaseOptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();

  const std::string database_path = test_dir + "/database.db";
  const std::string image_path = test_dir + "/images";

  // Create argv with additional options
  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path,
      "--image_path",
      image_path,
  };

  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  // Verify parsed values
  EXPECT_EQ(*options.database_path, database_path);
  EXPECT_EQ(*options.image_path, image_path);
}

TEST(BaseOptionManager, ParseWithProjectPath) {
  const std::string test_dir = CreateTestDir();
  const std::string config_path = test_dir + "/config.ini";
  CreateDirIfNotExists(test_dir + "/images");

  // Create and write a config file
  BaseOptionManager options_write;
  options_write.AddDatabaseOptions();
  options_write.AddImageOptions();

  *options_write.database_path = test_dir + "/database.db";
  *options_write.image_path = test_dir + "/images";
  options_write.Write(config_path);

  // Parse using project_path
  BaseOptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();

  const std::vector<std::string> args = {
      "colmap",
      "--project_path",
      config_path,
  };

  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  // Verify values were loaded from config file
  EXPECT_EQ(*options.database_path, *options_write.database_path);
  EXPECT_EQ(*options.image_path, *options_write.image_path);
}

TEST(BaseOptionManager, ParseEmptyArguments) {
  BaseOptionManager options;

  const std::vector<std::string> args = {"colmap"};
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  // Should succeed with no required options
  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));
}

TEST(BaseOptionManager, ParseUnknownArgumentsFails) {
  const std::string test_dir = CreateTestDir();

  BaseOptionManager options;
  options.AddDatabaseOptions();

  const std::string database_path = test_dir + "/database.db";

  // Create argv with an unknown option
  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path,
      "--unknown_option",
      "value",
  };

  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  // Should return false when encountering unknown option
  EXPECT_FALSE(options.Parse(argv.size(), argv.data()));
}

// Helper class to test enum options through BaseOptionManager
class TestEnumOptionManager : public BaseOptionManager {
 public:
  TestEnumOptionManager() : BaseOptionManager(/*add_project_options=*/false) {
    AddAndRegisterDefaultEnumOption("test_enum",
                                    &test_enum_value,
                                    TestEnumTypeToString,
                                    TestEnumTypeFromString);
  }

  TestEnumType test_enum_value = TestEnumType::VALUE_A;
};

TEST(BaseOptionManager, EnumOptionDefaultValue) {
  TestEnumOptionManager options;

  // Default value should be VALUE_A
  EXPECT_EQ(options.test_enum_value, TestEnumType::VALUE_A);

  // Parse with no enum option specified
  const std::vector<std::string> args = {"test"};
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  // Should still be default value
  EXPECT_EQ(options.test_enum_value, TestEnumType::VALUE_A);
}

TEST(BaseOptionManager, EnumOptionParseFromCommandLine) {
  TestEnumOptionManager options;

  // Parse with enum option set to VALUE_B
  const std::vector<std::string> args = {"test", "--test_enum", "VALUE_B"};
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  // Should be VALUE_B after parsing
  EXPECT_EQ(options.test_enum_value, TestEnumType::VALUE_B);
}

TEST(BaseOptionManager, EnumOptionParseFromCommandLineValueC) {
  TestEnumOptionManager options;

  // Parse with enum option set to VALUE_C
  const std::vector<std::string> args = {"test", "--test_enum", "VALUE_C"};
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  // Should be VALUE_C after parsing
  EXPECT_EQ(options.test_enum_value, TestEnumType::VALUE_C);
}

TEST(BaseOptionManager, EnumOptionInvalidValue) {
  TestEnumOptionManager options;

  // Parse with invalid enum value
  const std::vector<std::string> args = {"test", "--test_enum", "INVALID_VALUE"};
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  // Should fail due to invalid enum value
  EXPECT_FALSE(options.Parse(argv.size(), argv.data()));
}

// Helper class to test enum options with non-default initial value
class TestEnumOptionManagerWithValueB : public BaseOptionManager {
 public:
  TestEnumOptionManagerWithValueB()
      : BaseOptionManager(/*add_project_options=*/false) {
    AddAndRegisterDefaultEnumOption("test_enum",
                                    &test_enum_value,
                                    TestEnumTypeToString,
                                    TestEnumTypeFromString);
  }

  TestEnumType test_enum_value = TestEnumType::VALUE_B;
};

TEST(BaseOptionManager, EnumOptionNonDefaultInitialValue) {
  TestEnumOptionManagerWithValueB options;

  // Default value should be VALUE_B (non-default)
  EXPECT_EQ(options.test_enum_value, TestEnumType::VALUE_B);

  // Parse with no enum option specified
  const std::vector<std::string> args = {"test"};
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  // Should still be VALUE_B (the initial value)
  EXPECT_EQ(options.test_enum_value, TestEnumType::VALUE_B);
}

}  // namespace
}  // namespace colmap
