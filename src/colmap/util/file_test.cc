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

#include "colmap/util/file.h"

#include "colmap/util/testing.h"

#include <cstring>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(EnsureTrailingSlash, Nominal) {
  EXPECT_EQ(EnsureTrailingSlash(""), "/");
  EXPECT_EQ(EnsureTrailingSlash("/"), "/");
  EXPECT_EQ(EnsureTrailingSlash("////"), "////");
  EXPECT_EQ(EnsureTrailingSlash("test"), "test/");
  EXPECT_EQ(EnsureTrailingSlash("/test"), "/test/");
}

TEST(HasFileExtension, Nominal) {
  EXPECT_FALSE(HasFileExtension("", ".jpg"));
  EXPECT_FALSE(HasFileExtension("testjpg", ".jpg"));
  EXPECT_TRUE(HasFileExtension("test.jpg", ".jpg"));
  EXPECT_TRUE(HasFileExtension("test.jpg", ".Jpg"));
  EXPECT_TRUE(HasFileExtension("test.jpg", ".JPG"));
  EXPECT_TRUE(HasFileExtension("test.", "."));
}

TEST(SplitFileExtension, Nominal) {
  std::string root;
  std::string ext;
  SplitFileExtension("", &root, &ext);
  EXPECT_EQ(root, "");
  EXPECT_EQ(ext, "");
  SplitFileExtension(".", &root, &ext);
  EXPECT_EQ(root, "");
  EXPECT_EQ(ext, "");
  SplitFileExtension("file", &root, &ext);
  EXPECT_EQ(root, "file");
  EXPECT_EQ(ext, "");
  SplitFileExtension("file.", &root, &ext);
  EXPECT_EQ(root, "file");
  EXPECT_EQ(ext, "");
  SplitFileExtension("file.jpg", &root, &ext);
  EXPECT_EQ(root, "file");
  EXPECT_EQ(ext, ".jpg");
  SplitFileExtension("dir/file.jpg", &root, &ext);
  EXPECT_EQ(root, "dir/file");
  EXPECT_EQ(ext, ".jpg");
  SplitFileExtension("/dir/file.jpg", &root, &ext);
  EXPECT_EQ(root, "/dir/file");
  EXPECT_EQ(ext, ".jpg");
  SplitFileExtension("dir/file.suffix.jpg", &root, &ext);
  EXPECT_EQ(root, "dir/file.suffix");
  EXPECT_EQ(ext, ".jpg");
  SplitFileExtension("dir.suffix/file.suffix.jpg", &root, &ext);
  EXPECT_EQ(root, "dir.suffix/file.suffix");
  EXPECT_EQ(ext, ".jpg");
  SplitFileExtension("dir.suffix/file.", &root, &ext);
  EXPECT_EQ(root, "dir.suffix/file");
  EXPECT_EQ(ext, "");
  SplitFileExtension("./dir.suffix/file.", &root, &ext);
  EXPECT_EQ(root, "./dir.suffix/file");
  EXPECT_EQ(ext, "");
}

TEST(GetPathBaseName, Nominal) {
  EXPECT_EQ(GetPathBaseName(""), "");
  EXPECT_EQ(GetPathBaseName("test"), "test");
  EXPECT_EQ(GetPathBaseName("/test"), "test");
  EXPECT_EQ(GetPathBaseName("test/"), "test");
  EXPECT_EQ(GetPathBaseName("/test/"), "test");
  EXPECT_EQ(GetPathBaseName("test1/test2"), "test2");
  EXPECT_EQ(GetPathBaseName("/test1/test2"), "test2");
  EXPECT_EQ(GetPathBaseName("/test1/test2/"), "test2");
  EXPECT_EQ(GetPathBaseName("/test1/test2/"), "test2");
  EXPECT_EQ(GetPathBaseName("\\test1/test2/"), "test2");
  EXPECT_EQ(GetPathBaseName("\\test1\\test2\\"), "test2");
  EXPECT_EQ(GetPathBaseName("/test1/test2/test3.ext"), "test3.ext");
}

TEST(GetParentDir, Nominal) {
  EXPECT_EQ(GetParentDir(""), "");
  EXPECT_EQ(GetParentDir("test"), "");
  EXPECT_EQ(GetParentDir("/test"), "/");
  EXPECT_EQ(GetParentDir("/"), "/");
  EXPECT_EQ(GetParentDir("test/test"), "test");
}

TEST(JoinPaths, Nominal) {
  EXPECT_EQ(JoinPaths(""), "");
  EXPECT_EQ(JoinPaths("test"), "test");
  EXPECT_EQ(JoinPaths("/test"), "/test");
  EXPECT_EQ(JoinPaths("test/"), "test/");
  EXPECT_EQ(JoinPaths("/test/"), "/test/");
  EXPECT_EQ(JoinPaths("test1/test2"), "test1/test2");
  EXPECT_EQ(JoinPaths("/test1/test2"), "/test1/test2");
  EXPECT_EQ(JoinPaths("/test1/test2/"), "/test1/test2/");
  EXPECT_EQ(JoinPaths("/test1/test2/"), "/test1/test2/");
  EXPECT_EQ(JoinPaths("\\test1/test2/"), "\\test1/test2/");
  EXPECT_EQ(JoinPaths("\\test1\\test2\\"), "\\test1\\test2\\");
#ifdef _MSC_VER
  EXPECT_EQ(JoinPaths("test1", "test2"), "test1\\test2");
  EXPECT_EQ(JoinPaths("/test1", "test2"), "/test1\\test2");
#else
  EXPECT_EQ(JoinPaths("test1", "test2"), "test1/test2");
  EXPECT_EQ(JoinPaths("/test1", "test2"), "/test1/test2");
#endif
  EXPECT_EQ(JoinPaths("/test1", "/test2"), "/test2");
  EXPECT_EQ(JoinPaths("/test1", "/test2/"), "/test2/");
  EXPECT_EQ(JoinPaths("/test1", "/test2/", "test3.ext"), "/test2/test3.ext");
}

TEST(ReadWriteBinaryBlob, Nominal) {
  const std::string file_path = CreateTestDir() + "/test.bin";
  const int kNumBytes = 123;
  std::vector<char> data(kNumBytes);
  for (int i = 0; i < kNumBytes; ++i) {
    data[i] = (i * 100 + 4 + i) % 256;
  }

  WriteBinaryBlob(file_path, {data.data(), data.size()});

  std::vector<char> read_data;
  ReadBinaryBlob(file_path, &read_data);

  EXPECT_EQ(read_data, data);
}

#ifdef COLMAP_DOWNLOAD_ENABLED

TEST(DownloadFile, Nominal) {
  const std::string file_path = CreateTestDir() + "/test.bin";
  const int kNumBytes = 123;
  std::string data(kNumBytes, '0');
  for (int i = 0; i < kNumBytes; ++i) {
    data[i] = (i * 100 + 4 + i) % 256;
  }
  WriteBinaryBlob(file_path, {data.data(), data.size()});

  const std::optional<std::string> downloaded_data =
      DownloadFile("file://" + std::filesystem::absolute(file_path).string());
  ASSERT_TRUE(downloaded_data.has_value());
  EXPECT_EQ(*downloaded_data, data);

  ASSERT_FALSE(DownloadFile("file://" +
                            std::filesystem::absolute(file_path).string() +
                            "_not_found")
                   .has_value());
}

#endif

}  // namespace
}  // namespace colmap
