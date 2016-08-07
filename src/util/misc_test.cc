// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "util/misc"
#include <boost/test/unit_test.hpp>

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
