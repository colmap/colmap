// COLMAP - Structure-from-Motion and Multi-View Stereo.
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
#define BOOST_TEST_MODULE "util/string"
#include <boost/test/unit_test.hpp>

#include "util/string.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestStringPrintf) {
  BOOST_CHECK_EQUAL(StringPrintf("%s", "test"), "test");
  BOOST_CHECK_EQUAL(StringPrintf("%d", 1), "1");
  BOOST_CHECK_EQUAL(StringPrintf("%.3f", 1.234), "1.234");
  BOOST_CHECK_EQUAL(StringPrintf("test%s", "test"), "testtest");
  BOOST_CHECK_EQUAL(StringPrintf("test%d", 1), "test1");
  BOOST_CHECK_EQUAL(StringPrintf("test%.3f", 1.234), "test1.234");
  BOOST_CHECK_EQUAL(StringPrintf("%s%s", "test", "test"), "testtest");
  BOOST_CHECK_EQUAL(StringPrintf("%d%s", 1, "test"), "1test");
  BOOST_CHECK_EQUAL(StringPrintf("%.3f%s", 1.234, "test"), "1.234test");
}

BOOST_AUTO_TEST_CASE(TestStringReplace) {
  BOOST_CHECK_EQUAL(StringReplace("test", "-", ""), "test");
  BOOST_CHECK_EQUAL(StringReplace("test", "t", "a"), "aesa");
  BOOST_CHECK_EQUAL(StringReplace("test", "t", "---"), "---es---");
  BOOST_CHECK_EQUAL(StringReplace("test", "", "a"), "test");
  BOOST_CHECK_EQUAL(StringReplace("test", "", ""), "test");
  BOOST_CHECK_EQUAL(StringReplace("ttt", "ttt", "+++"), "+++");
}

BOOST_AUTO_TEST_CASE(TestStringSplit) {
  const std::vector<std::string> list1 = StringSplit("1,2,3,4,5 , 6", ",");
  BOOST_CHECK_EQUAL(list1.size(), 6);
  BOOST_CHECK_EQUAL(list1[0], "1");
  BOOST_CHECK_EQUAL(list1[1], "2");
  BOOST_CHECK_EQUAL(list1[2], "3");
  BOOST_CHECK_EQUAL(list1[3], "4");
  BOOST_CHECK_EQUAL(list1[4], "5 ");
  BOOST_CHECK_EQUAL(list1[5], " 6");
  const std::vector<std::string> list2 = StringSplit("1,2,3,4,5 , 6", "");
  BOOST_CHECK_EQUAL(list2.size(), 1);
  BOOST_CHECK_EQUAL(list2[0], "1,2,3,4,5 , 6");
  const std::vector<std::string> list3 = StringSplit("1,,2,,3,4,5 , 6", ",");
  BOOST_CHECK_EQUAL(list3.size(), 6);
  BOOST_CHECK_EQUAL(list3[0], "1");
  BOOST_CHECK_EQUAL(list3[1], "2");
  BOOST_CHECK_EQUAL(list3[2], "3");
  BOOST_CHECK_EQUAL(list3[3], "4");
  BOOST_CHECK_EQUAL(list3[4], "5 ");
  BOOST_CHECK_EQUAL(list3[5], " 6");
  const std::vector<std::string> list4 = StringSplit("1,,2,,3,4,5 , 6", ",,");
  BOOST_CHECK_EQUAL(list4.size(), 6);
  BOOST_CHECK_EQUAL(list4[0], "1");
  BOOST_CHECK_EQUAL(list4[1], "2");
  BOOST_CHECK_EQUAL(list4[2], "3");
  BOOST_CHECK_EQUAL(list4[3], "4");
  BOOST_CHECK_EQUAL(list4[4], "5 ");
  BOOST_CHECK_EQUAL(list4[5], " 6");
  const std::vector<std::string> list5 = StringSplit("1,,2,,3,4,5 , 6", ", ");
  BOOST_CHECK_EQUAL(list5.size(), 6);
  BOOST_CHECK_EQUAL(list5[0], "1");
  BOOST_CHECK_EQUAL(list5[1], "2");
  BOOST_CHECK_EQUAL(list5[2], "3");
  BOOST_CHECK_EQUAL(list5[3], "4");
  BOOST_CHECK_EQUAL(list5[4], "5");
  BOOST_CHECK_EQUAL(list5[5], "6");
}

BOOST_AUTO_TEST_CASE(TestStringStartsWith) {
  BOOST_CHECK(!StringStartsWith("", ""));
  BOOST_CHECK(!StringStartsWith("a", ""));
  BOOST_CHECK(!StringStartsWith("", "a"));
  BOOST_CHECK(StringStartsWith("a", "a"));
  BOOST_CHECK(StringStartsWith("aa", "a"));
  BOOST_CHECK(StringStartsWith("aa", "aa"));
  BOOST_CHECK(StringStartsWith("aaaaaaaaa", "aa"));
}

#define TEST_STRING_TRIM(Func, str, ref_str_trimmed) \
  {                                                  \
    std::string str_trimmed = str;                   \
    Func(&str_trimmed);                              \
    BOOST_CHECK_EQUAL(str_trimmed, ref_str_trimmed); \
  }

BOOST_AUTO_TEST_CASE(TestStringStrim) {
  TEST_STRING_TRIM(StringTrim, "", "");
  TEST_STRING_TRIM(StringTrim, " ", "");
  TEST_STRING_TRIM(StringTrim, "a", "a");
  TEST_STRING_TRIM(StringTrim, " a", "a");
  TEST_STRING_TRIM(StringTrim, "a ", "a");
  TEST_STRING_TRIM(StringTrim, " a ", "a");
  TEST_STRING_TRIM(StringTrim, "  a  ", "a");
  TEST_STRING_TRIM(StringTrim, "aa  ", "aa");
  TEST_STRING_TRIM(StringTrim, "a  a  ", "a  a");
  TEST_STRING_TRIM(StringTrim, "a  a  a ", "a  a  a");
  TEST_STRING_TRIM(StringTrim, "\n\r\t", "");
}

BOOST_AUTO_TEST_CASE(TestStringLeftString) {
  TEST_STRING_TRIM(StringLeftTrim, "", "");
  TEST_STRING_TRIM(StringLeftTrim, " ", "");
  TEST_STRING_TRIM(StringLeftTrim, "a", "a");
  TEST_STRING_TRIM(StringLeftTrim, " a", "a");
  TEST_STRING_TRIM(StringLeftTrim, "a ", "a ");
  TEST_STRING_TRIM(StringLeftTrim, " a ", "a ");
  TEST_STRING_TRIM(StringLeftTrim, "  a  ", "a  ");
  TEST_STRING_TRIM(StringLeftTrim, "aa  ", "aa  ");
  TEST_STRING_TRIM(StringLeftTrim, "a  a  ", "a  a  ");
  TEST_STRING_TRIM(StringLeftTrim, "  a  a", "a  a");
  TEST_STRING_TRIM(StringLeftTrim, "a  a  a ", "a  a  a ");
  TEST_STRING_TRIM(StringTrim, "\n\r\ta", "a");
}

BOOST_AUTO_TEST_CASE(TestStringStrimRight) {
  TEST_STRING_TRIM(StringRightTrim, "", "");
  TEST_STRING_TRIM(StringRightTrim, " ", "");
  TEST_STRING_TRIM(StringRightTrim, "a", "a");
  TEST_STRING_TRIM(StringRightTrim, " a", " a");
  TEST_STRING_TRIM(StringRightTrim, "a ", "a");
  TEST_STRING_TRIM(StringRightTrim, " a ", " a");
  TEST_STRING_TRIM(StringRightTrim, "  a  ", "  a");
  TEST_STRING_TRIM(StringRightTrim, "aa  ", "aa");
  TEST_STRING_TRIM(StringRightTrim, "a  a  ", "a  a");
  TEST_STRING_TRIM(StringRightTrim, "a  a  a ", "a  a  a");
  TEST_STRING_TRIM(StringTrim, "a\n\r\t", "a");
}
