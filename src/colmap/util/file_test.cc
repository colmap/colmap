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

#include "colmap/util/file.h"

#include "colmap/util/testing.h"

#include <cstring>
#include <fstream>

#include <gmock/gmock.h>
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

TEST(AddFileExtension, Nominal) {
  EXPECT_EQ(AddFileExtension("test", ".txt"), "test.txt");
  EXPECT_EQ(AddFileExtension("test.jpg", ".txt"), "test.jpg.txt");
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
  if constexpr (std::filesystem::path::preferred_separator == '\\') {
    EXPECT_EQ(GetPathBaseName("\\test1\\test2\\"), "test2");
  } else {
    EXPECT_EQ(GetPathBaseName("\\test1\\test2\\"), "\\test1\\test2\\");
  }
  EXPECT_EQ(GetPathBaseName("/test1/test2/test3.ext"), "test3.ext");
}

TEST(GetParentDir, Nominal) {
  EXPECT_EQ(GetParentDir(""), "");
  EXPECT_EQ(GetParentDir("test"), "");
  EXPECT_EQ(GetParentDir("/test"), "/");
  EXPECT_EQ(GetParentDir("/"), "/");
  EXPECT_EQ(GetParentDir("test/test"), "test");
}

TEST(NormalizePath, Nominal) {
  EXPECT_EQ(NormalizePath("a/b//./c"), "a/b/c");
  EXPECT_EQ(NormalizePath("a/b//./c/"), "a/b/c/");
  EXPECT_EQ(NormalizePath("/a/b//./c"), "/a/b/c");
  if constexpr (std::filesystem::path::preferred_separator == '\\') {
    EXPECT_EQ(NormalizePath("/a\\b//./c"), "/a/b/c");
  } else {
    EXPECT_EQ(NormalizePath("/a\\b//./c"), "/a\\b/c");
  }
}

TEST(GetNormalizedRelativePath, Nominal) {
  EXPECT_EQ(GetNormalizedRelativePath("a/b/c", "a/b"), "c");
  EXPECT_EQ(GetNormalizedRelativePath("/a/b/c", "/a/b"), "c");
  EXPECT_EQ(GetNormalizedRelativePath("/a/b/c", "/a"), "b/c");
  EXPECT_EQ(GetNormalizedRelativePath("/a//b/c/", "/a/"), "b/c/");
  if constexpr (std::filesystem::path::preferred_separator == '\\') {
    EXPECT_EQ(GetNormalizedRelativePath("a/b\\c/", "a/"), "b/c/");
  } else {
    EXPECT_EQ(GetNormalizedRelativePath("a/b\\c/", "a/"), "b\\c/");
  }
}

TEST(FileCopy, Nominal) {
  const auto dir = CreateTestDir();
  const auto src_path = dir / "source.txt";
  const auto dst_path = dir / "destination.txt";

  {
    std::ofstream file(src_path);
    file << "test content";
  }

  FileCopy(src_path, dst_path, FileCopyType::COPY);
  EXPECT_TRUE(ExistsFile(dst_path));
  {
    std::ifstream file(dst_path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    EXPECT_EQ(buffer.str(), "test content");
  }

  const auto dst_hard_link_path = dir / "destination_hard_link.txt";
  FileCopy(src_path, dst_hard_link_path, FileCopyType::HARD_LINK);
  EXPECT_TRUE(ExistsFile(dst_hard_link_path));

  const auto dst_soft_link_path = dir / "destination_soft_link.txt";
  FileCopy(src_path, dst_soft_link_path, FileCopyType::SOFT_LINK);
  EXPECT_TRUE(ExistsFile(dst_soft_link_path));
}

TEST(GetRecursiveFileList, Nominal) {
  const auto dir = CreateTestDir();
  const auto file1 = dir / "file1.txt";
  const auto file2 = dir / "file2.txt";
  const auto subdir = dir / "subdir";
  const auto file3 = dir / "file3.txt";

  {
    std::ofstream f1(file1);
    std::ofstream f2(file2);
    std::ofstream f3(file3);
  }
  CreateDirIfNotExists(subdir);

  const auto file_list = GetRecursiveFileList(dir);
  EXPECT_THAT(file_list,
              testing::UnorderedElementsAre(
                  file1.string(), file2.string(), file3.string()));
}

TEST(GetDirList, Nominal) {
  const auto dir = CreateTestDir();
  const auto subdir1 = dir / "subdir1";
  const auto subdir2 = dir / "subdir2";
  const auto subdir1_nested = subdir1 / "nested";
  const auto file = dir / "file.txt";

  CreateDirIfNotExists(subdir1);
  CreateDirIfNotExists(subdir2);
  CreateDirIfNotExists(subdir1_nested);

  {
    std::ofstream f(file);
  }

  const auto dir_list = GetDirList(dir);
  EXPECT_THAT(
      dir_list,
      testing::UnorderedElementsAre(subdir1.string(), subdir2.string()));
}

TEST(HomeDir, Nominal) {
  // Just test that it doesn't crash, since there is no guarantee that it
  // resolves successfully on a particular machine.
  const auto home_dir = HomeDir();
  if (home_dir) {
    LOG(INFO) << *home_dir;
  }
}

TEST(ReadWriteBinaryBlob, Nominal) {
  const auto file_path = CreateTestDir() / "test.bin";
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

TEST(IsURI, Nominal) {
  EXPECT_FALSE(IsURI(""));
  EXPECT_TRUE(IsURI("http://"));
  EXPECT_TRUE(IsURI("https://"));
  EXPECT_TRUE(IsURI("file://"));
  EXPECT_TRUE(IsURI("http://foobar"));
  EXPECT_TRUE(IsURI("https://foobar"));
  EXPECT_TRUE(IsURI("file://foobar"));
  EXPECT_FALSE(IsURI("http"));
  EXPECT_FALSE(IsURI("https"));
  EXPECT_FALSE(IsURI("file"));
}

#ifdef COLMAP_DOWNLOAD_ENABLED

TEST(DownloadFile, Nominal) {
  const auto file_path = CreateTestDir() / "test.bin";
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

TEST(ComputeSHA256, Nominal) {
  EXPECT_EQ(ComputeSHA256(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(ComputeSHA256("hello world"),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
}

TEST(MaybeDownloadAndCacheFile, Nominal) {
  const auto test_dir = CreateTestDir();
  OverwriteDownloadCacheDir(test_dir);

  const std::string data = "123asd<>?";
  const std::string name = "cached.bin";
  const std::string sha256 =
      "2915068022d460a622fb078147aee8d590c0a1bb1907d35fd27cb2f7bdb991dd";
  const auto server_file_path = test_dir / "server.bin";
  const auto cached_file_path = test_dir / (sha256 + "-" + name);
  WriteBinaryBlob(server_file_path, {data.data(), data.size()});

  const std::string uri = "file://" +
                          std::filesystem::absolute(server_file_path).string() +
                          ";" + name + ";" + sha256;

  EXPECT_EQ(MaybeDownloadAndCacheFile(uri), cached_file_path);
  EXPECT_EQ(MaybeDownloadAndCacheFile(uri), cached_file_path);
  EXPECT_EQ(MaybeDownloadAndCacheFile(cached_file_path.string()),
            cached_file_path);
}

#endif

}  // namespace
}  // namespace colmap
