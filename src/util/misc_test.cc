// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "util/misc"
#include "util/testing.h"

#include "util/misc.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEnsureTrailingSlash) {
  BOOST_CHECK_EQUAL(EnsureTrailingSlash(""), "/");
  BOOST_CHECK_EQUAL(EnsureTrailingSlash("/"), "/");
  BOOST_CHECK_EQUAL(EnsureTrailingSlash("////"), "////");
  BOOST_CHECK_EQUAL(EnsureTrailingSlash("test"), "test/");
  BOOST_CHECK_EQUAL(EnsureTrailingSlash("/test"), "/test/");
}

BOOST_AUTO_TEST_CASE(TestHasFileExtension) {
  BOOST_CHECK_EQUAL(HasFileExtension("", ".jpg"), false);
  BOOST_CHECK_EQUAL(HasFileExtension("testjpg", ".jpg"), false);
  BOOST_CHECK_EQUAL(HasFileExtension("test.jpg", ".jpg"), true);
  BOOST_CHECK_EQUAL(HasFileExtension("test.jpg", ".Jpg"), true);
  BOOST_CHECK_EQUAL(HasFileExtension("test.jpg", ".JPG"), true);
  BOOST_CHECK_EQUAL(HasFileExtension("test.", "."), true);
}

BOOST_AUTO_TEST_CASE(TestSplitFileExtension) {
  std::string root;
  std::string ext;
  SplitFileExtension("", &root, &ext);
  BOOST_CHECK_EQUAL(root, "");
  BOOST_CHECK_EQUAL(ext, "");
  SplitFileExtension(".", &root, &ext);
  BOOST_CHECK_EQUAL(root, "");
  BOOST_CHECK_EQUAL(ext, "");
  SplitFileExtension("file", &root, &ext);
  BOOST_CHECK_EQUAL(root, "file");
  BOOST_CHECK_EQUAL(ext, "");
  SplitFileExtension("file.", &root, &ext);
  BOOST_CHECK_EQUAL(root, "file");
  BOOST_CHECK_EQUAL(ext, "");
  SplitFileExtension("file.jpg", &root, &ext);
  BOOST_CHECK_EQUAL(root, "file");
  BOOST_CHECK_EQUAL(ext, ".jpg");
  SplitFileExtension("dir/file.jpg", &root, &ext);
  BOOST_CHECK_EQUAL(root, "dir/file");
  BOOST_CHECK_EQUAL(ext, ".jpg");
  SplitFileExtension("/dir/file.jpg", &root, &ext);
  BOOST_CHECK_EQUAL(root, "/dir/file");
  BOOST_CHECK_EQUAL(ext, ".jpg");
  SplitFileExtension("dir/file.suffix.jpg", &root, &ext);
  BOOST_CHECK_EQUAL(root, "dir/file.suffix");
  BOOST_CHECK_EQUAL(ext, ".jpg");
  SplitFileExtension("dir.suffix/file.suffix.jpg", &root, &ext);
  BOOST_CHECK_EQUAL(root, "dir.suffix/file.suffix");
  BOOST_CHECK_EQUAL(ext, ".jpg");
  SplitFileExtension("dir.suffix/file.", &root, &ext);
  BOOST_CHECK_EQUAL(root, "dir.suffix/file");
  BOOST_CHECK_EQUAL(ext, "");
  SplitFileExtension("./dir.suffix/file.", &root, &ext);
  BOOST_CHECK_EQUAL(root, "./dir.suffix/file");
  BOOST_CHECK_EQUAL(ext, "");
}

BOOST_AUTO_TEST_CASE(TestGetPathBaseName) {
  BOOST_CHECK_EQUAL(GetPathBaseName(""), "");
  BOOST_CHECK_EQUAL(GetPathBaseName("test"), "test");
  BOOST_CHECK_EQUAL(GetPathBaseName("/test"), "test");
  BOOST_CHECK_EQUAL(GetPathBaseName("test/"), "test");
  BOOST_CHECK_EQUAL(GetPathBaseName("/test/"), "test");
  BOOST_CHECK_EQUAL(GetPathBaseName("test1/test2"), "test2");
  BOOST_CHECK_EQUAL(GetPathBaseName("/test1/test2"), "test2");
  BOOST_CHECK_EQUAL(GetPathBaseName("/test1/test2/"), "test2");
  BOOST_CHECK_EQUAL(GetPathBaseName("/test1/test2/"), "test2");
  BOOST_CHECK_EQUAL(GetPathBaseName("\\test1/test2/"), "test2");
  BOOST_CHECK_EQUAL(GetPathBaseName("\\test1\\test2\\"), "test2");
  BOOST_CHECK_EQUAL(GetPathBaseName("/test1/test2/test3.ext"), "test3.ext");
}

BOOST_AUTO_TEST_CASE(TestGetParentDir) {
  BOOST_CHECK_EQUAL(GetParentDir(""), "");
  BOOST_CHECK_EQUAL(GetParentDir("test"), "");
  BOOST_CHECK_EQUAL(GetParentDir("/test"), "/");
  BOOST_CHECK_EQUAL(GetParentDir("/"), "");
  BOOST_CHECK_EQUAL(GetParentDir("test/test"), "test");
}

BOOST_AUTO_TEST_CASE(TestJoinPaths) {
  BOOST_CHECK_EQUAL(JoinPaths(""), "");
  BOOST_CHECK_EQUAL(JoinPaths("test"), "test");
  BOOST_CHECK_EQUAL(JoinPaths("/test"), "/test");
  BOOST_CHECK_EQUAL(JoinPaths("test/"), "test/");
  BOOST_CHECK_EQUAL(JoinPaths("/test/"), "/test/");
  BOOST_CHECK_EQUAL(JoinPaths("test1/test2"), "test1/test2");
  BOOST_CHECK_EQUAL(JoinPaths("/test1/test2"), "/test1/test2");
  BOOST_CHECK_EQUAL(JoinPaths("/test1/test2/"), "/test1/test2/");
  BOOST_CHECK_EQUAL(JoinPaths("/test1/test2/"), "/test1/test2/");
  BOOST_CHECK_EQUAL(JoinPaths("\\test1/test2/"), "\\test1/test2/");
  BOOST_CHECK_EQUAL(JoinPaths("\\test1\\test2\\"), "\\test1\\test2\\");
#ifdef _MSC_VER
  BOOST_CHECK_EQUAL(JoinPaths("test1", "test2"), "test1\\test2");
  BOOST_CHECK_EQUAL(JoinPaths("/test1", "test2"), "/test1\\test2");
#else
  BOOST_CHECK_EQUAL(JoinPaths("test1", "test2"), "test1/test2");
  BOOST_CHECK_EQUAL(JoinPaths("/test1", "test2"), "/test1/test2");
#endif
  BOOST_CHECK_EQUAL(JoinPaths("/test1", "/test2"), "/test1/test2");
  BOOST_CHECK_EQUAL(JoinPaths("/test1", "/test2/"), "/test1/test2/");
  BOOST_CHECK_EQUAL(JoinPaths("/test1", "/test2/", "test3.ext"),
                    "/test1/test2/test3.ext");
}

BOOST_AUTO_TEST_CASE(TestVectorContainsValue) {
  BOOST_CHECK(VectorContainsValue<int>({1, 2, 3}, 1));
  BOOST_CHECK(!VectorContainsValue<int>({2, 3}, 1));
}

BOOST_AUTO_TEST_CASE(TestVectorContainsDuplicateValues) {
  BOOST_CHECK(!VectorContainsDuplicateValues<int>({}));
  BOOST_CHECK(!VectorContainsDuplicateValues<int>({1}));
  BOOST_CHECK(!VectorContainsDuplicateValues<int>({1, 2}));
  BOOST_CHECK(!VectorContainsDuplicateValues<int>({1, 2, 3}));
  BOOST_CHECK(VectorContainsDuplicateValues<int>({1, 1, 2, 3}));
  BOOST_CHECK(VectorContainsDuplicateValues<int>({1, 1, 2, 2, 3}));
  BOOST_CHECK(VectorContainsDuplicateValues<int>({1, 2, 3, 3}));
  BOOST_CHECK(!VectorContainsDuplicateValues<std::string>({"a"}));
  BOOST_CHECK(!VectorContainsDuplicateValues<std::string>({"a", "b"}));
  BOOST_CHECK(VectorContainsDuplicateValues<std::string>({"a", "a"}));
}

BOOST_AUTO_TEST_CASE(TestCSVToVector) {
  const std::vector<int> list1 = CSVToVector<int>("1, 2, 3 , 4,5,6 ");
  BOOST_CHECK_EQUAL(list1.size(), 6);
  BOOST_CHECK_EQUAL(list1[0], 1);
  BOOST_CHECK_EQUAL(list1[1], 2);
  BOOST_CHECK_EQUAL(list1[2], 3);
  BOOST_CHECK_EQUAL(list1[3], 4);
  BOOST_CHECK_EQUAL(list1[4], 5);
  BOOST_CHECK_EQUAL(list1[5], 6);
  const std::vector<int> list2 = CSVToVector<int>("1; 2; 3 ; 4;5;6 ");
  BOOST_CHECK_EQUAL(list2.size(), 6);
  BOOST_CHECK_EQUAL(list2[0], 1);
  BOOST_CHECK_EQUAL(list2[1], 2);
  BOOST_CHECK_EQUAL(list2[2], 3);
  BOOST_CHECK_EQUAL(list2[3], 4);
  BOOST_CHECK_EQUAL(list2[4], 5);
  BOOST_CHECK_EQUAL(list2[5], 6);
  const std::vector<int> list3 = CSVToVector<int>("1;, 2;; 3 ; 4;5;6 ");
  BOOST_CHECK_EQUAL(list3.size(), 6);
  BOOST_CHECK_EQUAL(list3[0], 1);
  BOOST_CHECK_EQUAL(list3[1], 2);
  BOOST_CHECK_EQUAL(list3[2], 3);
  BOOST_CHECK_EQUAL(list3[3], 4);
  BOOST_CHECK_EQUAL(list3[4], 5);
  BOOST_CHECK_EQUAL(list3[5], 6);
}

BOOST_AUTO_TEST_CASE(TestVectorToCSV) {
  BOOST_CHECK_EQUAL(VectorToCSV<int>({}), "");
  BOOST_CHECK_EQUAL(VectorToCSV<int>({1}), "1");
  BOOST_CHECK_EQUAL(VectorToCSV<int>({1, 2, 3}), "1, 2, 3");
}

BOOST_AUTO_TEST_CASE(TestRemoveCommandLineArgument) {
  int argc = 3;
  std::unique_ptr<char[]> arg1(new char[4]);
  memcpy(arg1.get(), "abc", 4 * sizeof(char));
  std::unique_ptr<char[]> arg2(new char[4]);
  memcpy(arg2.get(), "def", 4 * sizeof(char));
  std::unique_ptr<char[]> arg3(new char[4]);
  memcpy(arg3.get(), "ghi", 4 * sizeof(char));
  std::vector<char*> argv = {arg1.get(), arg2.get(), arg3.get()};

  RemoveCommandLineArgument("abcd", &argc, argv.data());
  BOOST_CHECK_EQUAL(argc, 3);
  BOOST_CHECK_EQUAL(argv[0], "abc");
  BOOST_CHECK_EQUAL(argv[1], "def");
  BOOST_CHECK_EQUAL(argv[2], "ghi");

  RemoveCommandLineArgument("def", &argc, argv.data());
  BOOST_CHECK_EQUAL(argc, 2);
  BOOST_CHECK_EQUAL(argv[0], "abc");
  BOOST_CHECK_EQUAL(argv[1], "ghi");

  RemoveCommandLineArgument("ghi", &argc, argv.data());
  BOOST_CHECK_EQUAL(argc, 1);
  BOOST_CHECK_EQUAL(argv[0], "abc");

  RemoveCommandLineArgument("abc", &argc, argv.data());
  BOOST_CHECK_EQUAL(argc, 0);

  RemoveCommandLineArgument("abc", &argc, argv.data());
  BOOST_CHECK_EQUAL(argc, 0);
}
