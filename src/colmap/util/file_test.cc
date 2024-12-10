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

#include "thirdparty/httplib/httplib.h"

#include <cstring>
#include <thread>

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

#ifdef COLMAP_HTTP_ENABLED

TEST(DownloadFile, Nominal) {
  const std::string kHost = "localhost";
  const std::string kPath = "/hello";
  const std::string kExpectedResponse = "Hello World";

  httplib::Server server;
  server.Get(kPath,
             [&kExpectedResponse](const httplib::Request& request,
                                  httplib::Response& response) {
               response.set_content(kExpectedResponse, "text/plain");
             });

  int port = -1;
  for (int i = 0; i < 3; ++i) {
    port = server.bind_to_any_port(kHost);
  }
  LOG(INFO) << "Binding server to port " << port;

  ASSERT_NE(port, -1);
  std::thread thread([&server, &kHost, &port] { server.listen(kHost, port); });

  std::ostringstream host;
  host << "http://" << kHost << ":" << port;

  const std::optional<std::string> data = DownloadFile(host.str(), kPath);
  ASSERT_TRUE(data.has_value());
  EXPECT_EQ(*data, kExpectedResponse);

  server.stop();
  thread.join();
}

#endif

}  // namespace
}  // namespace colmap
