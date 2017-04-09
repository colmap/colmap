// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
  BOOST_CHECK_EQUAL(JoinPaths("test1", "test2"), "test1/test2");
  BOOST_CHECK_EQUAL(JoinPaths("/test1", "test2"), "/test1/test2");
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
