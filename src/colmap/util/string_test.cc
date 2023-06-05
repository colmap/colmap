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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "util/string"
#include "colmap/util/string.h"

#include "colmap/util/testing.h"

using namespace colmap;

#define TEST_STRING_INPLACE(Func, str, ref_str) \
  {                                             \
    std::string str_inplace = str;              \
    Func(&str_inplace);                         \
    BOOST_CHECK_EQUAL(str_inplace, ref_str);    \
  }

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

BOOST_AUTO_TEST_CASE(TestStringGetAfter) {
  BOOST_CHECK_EQUAL(StringGetAfter("test", ""), "test");
  BOOST_CHECK_EQUAL(StringGetAfter("test", "notinit"), "");
  BOOST_CHECK_EQUAL(StringGetAfter("test", "e"), "st");
  BOOST_CHECK_EQUAL(StringGetAfter("test, multiple tests", "test"), "s");
  BOOST_CHECK_EQUAL(StringGetAfter("", ""), "");
  BOOST_CHECK_EQUAL(StringGetAfter("path/to/dataset/sub1/image.png", "sub1/"),
                    "image.png");
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
  const std::vector<std::string> list6 = StringSplit(",1,,2,,3,4,5 , 6 ", ", ");
  BOOST_CHECK_EQUAL(list6.size(), 8);
  BOOST_CHECK_EQUAL(list6[0], "");
  BOOST_CHECK_EQUAL(list6[1], "1");
  BOOST_CHECK_EQUAL(list6[2], "2");
  BOOST_CHECK_EQUAL(list6[3], "3");
  BOOST_CHECK_EQUAL(list6[4], "4");
  BOOST_CHECK_EQUAL(list6[5], "5");
  BOOST_CHECK_EQUAL(list6[6], "6");
  BOOST_CHECK_EQUAL(list6[7], "");
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

BOOST_AUTO_TEST_CASE(TestStringStrim) {
  TEST_STRING_INPLACE(StringTrim, "", "");
  TEST_STRING_INPLACE(StringTrim, " ", "");
  TEST_STRING_INPLACE(StringTrim, "a", "a");
  TEST_STRING_INPLACE(StringTrim, " a", "a");
  TEST_STRING_INPLACE(StringTrim, "a ", "a");
  TEST_STRING_INPLACE(StringTrim, " a ", "a");
  TEST_STRING_INPLACE(StringTrim, "  a  ", "a");
  TEST_STRING_INPLACE(StringTrim, "aa  ", "aa");
  TEST_STRING_INPLACE(StringTrim, "a  a  ", "a  a");
  TEST_STRING_INPLACE(StringTrim, "a  a  a ", "a  a  a");
  TEST_STRING_INPLACE(StringTrim, "\n\r\t", "");
}

BOOST_AUTO_TEST_CASE(TestStringLeftString) {
  TEST_STRING_INPLACE(StringLeftTrim, "", "");
  TEST_STRING_INPLACE(StringLeftTrim, " ", "");
  TEST_STRING_INPLACE(StringLeftTrim, "a", "a");
  TEST_STRING_INPLACE(StringLeftTrim, " a", "a");
  TEST_STRING_INPLACE(StringLeftTrim, "a ", "a ");
  TEST_STRING_INPLACE(StringLeftTrim, " a ", "a ");
  TEST_STRING_INPLACE(StringLeftTrim, "  a  ", "a  ");
  TEST_STRING_INPLACE(StringLeftTrim, "aa  ", "aa  ");
  TEST_STRING_INPLACE(StringLeftTrim, "a  a  ", "a  a  ");
  TEST_STRING_INPLACE(StringLeftTrim, "  a  a", "a  a");
  TEST_STRING_INPLACE(StringLeftTrim, "a  a  a ", "a  a  a ");
  TEST_STRING_INPLACE(StringTrim, "\n\r\ta", "a");
}

BOOST_AUTO_TEST_CASE(TestStringStrimRight) {
  TEST_STRING_INPLACE(StringRightTrim, "", "");
  TEST_STRING_INPLACE(StringRightTrim, " ", "");
  TEST_STRING_INPLACE(StringRightTrim, "a", "a");
  TEST_STRING_INPLACE(StringRightTrim, " a", " a");
  TEST_STRING_INPLACE(StringRightTrim, "a ", "a");
  TEST_STRING_INPLACE(StringRightTrim, " a ", " a");
  TEST_STRING_INPLACE(StringRightTrim, "  a  ", "  a");
  TEST_STRING_INPLACE(StringRightTrim, "aa  ", "aa");
  TEST_STRING_INPLACE(StringRightTrim, "a  a  ", "a  a");
  TEST_STRING_INPLACE(StringRightTrim, "a  a  a ", "a  a  a");
  TEST_STRING_INPLACE(StringTrim, "a\n\r\t", "a");
}

BOOST_AUTO_TEST_CASE(TestStringToLower) {
  TEST_STRING_INPLACE(StringToLower, "", "");
  TEST_STRING_INPLACE(StringToLower, " ", " ");
  TEST_STRING_INPLACE(StringToLower, "a", "a");
  TEST_STRING_INPLACE(StringToLower, " a", " a");
  TEST_STRING_INPLACE(StringToLower, "a ", "a ");
  TEST_STRING_INPLACE(StringToLower, " a ", " a ");
  TEST_STRING_INPLACE(StringToLower, "aa  ", "aa  ");
  TEST_STRING_INPLACE(StringToLower, "A", "a");
  TEST_STRING_INPLACE(StringToLower, " A", " a");
  TEST_STRING_INPLACE(StringToLower, "A ", "a ");
  TEST_STRING_INPLACE(StringToLower, " A ", " a ");
  TEST_STRING_INPLACE(StringToLower, "AA  ", "aa  ");
  TEST_STRING_INPLACE(StringToLower,
                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                      "abcdefghijklmnopqrstuvwxyz");
  TEST_STRING_INPLACE(StringToLower,
                      "0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                      "0123456789 abcdefghijklmnopqrstuvwxyz");
}

BOOST_AUTO_TEST_CASE(TestStringToUpper) {
  TEST_STRING_INPLACE(StringToUpper, "", "");
  TEST_STRING_INPLACE(StringToUpper, " ", " ");
  TEST_STRING_INPLACE(StringToUpper, "A", "A");
  TEST_STRING_INPLACE(StringToUpper, " A", " A");
  TEST_STRING_INPLACE(StringToUpper, "A ", "A ");
  TEST_STRING_INPLACE(StringToUpper, " A ", " A ");
  TEST_STRING_INPLACE(StringToUpper, "AA  ", "AA  ");
  TEST_STRING_INPLACE(StringToUpper, "a", "A");
  TEST_STRING_INPLACE(StringToUpper, " a", " A");
  TEST_STRING_INPLACE(StringToUpper, "a ", "A ");
  TEST_STRING_INPLACE(StringToUpper, " a ", " A ");
  TEST_STRING_INPLACE(StringToUpper, "aa  ", "AA  ");
  TEST_STRING_INPLACE(StringToUpper,
                      "abcdefghijklmnopqrstuvwxyz",
                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
  TEST_STRING_INPLACE(StringToUpper,
                      "0123456789 abcdefghijklmnopqrstuvwxyz",
                      "0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ");
}

BOOST_AUTO_TEST_CASE(TestStringContains) {
  BOOST_CHECK(StringContains("", ""));
  BOOST_CHECK(StringContains("a", ""));
  BOOST_CHECK(StringContains("a", "a"));
  BOOST_CHECK(StringContains("ab", "a"));
  BOOST_CHECK(StringContains("ab", "ab"));
  BOOST_CHECK(!StringContains("", "a"));
  BOOST_CHECK(!StringContains("ab", "c"));
}
