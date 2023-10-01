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

#include "colmap/util/string.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

#define TEST_STRING_INPLACE(Func, str, ref_str) \
  {                                             \
    std::string str_inplace = str;              \
    Func(&str_inplace);                         \
    EXPECT_EQ(str_inplace, ref_str);            \
  }

TEST(StringPrintf, Nominal) {
  EXPECT_EQ(StringPrintf("%s", "test"), "test");
  EXPECT_EQ(StringPrintf("%d", 1), "1");
  EXPECT_EQ(StringPrintf("%.3f", 1.234), "1.234");
  EXPECT_EQ(StringPrintf("test%s", "test"), "testtest");
  EXPECT_EQ(StringPrintf("test%d", 1), "test1");
  EXPECT_EQ(StringPrintf("test%.3f", 1.234), "test1.234");
  EXPECT_EQ(StringPrintf("%s%s", "test", "test"), "testtest");
  EXPECT_EQ(StringPrintf("%d%s", 1, "test"), "1test");
  EXPECT_EQ(StringPrintf("%.3f%s", 1.234, "test"), "1.234test");
}

TEST(StringReplace, Nominal) {
  EXPECT_EQ(StringReplace("test", "-", ""), "test");
  EXPECT_EQ(StringReplace("test", "t", "a"), "aesa");
  EXPECT_EQ(StringReplace("test", "t", "---"), "---es---");
  EXPECT_EQ(StringReplace("test", "", "a"), "test");
  EXPECT_EQ(StringReplace("test", "", ""), "test");
  EXPECT_EQ(StringReplace("ttt", "ttt", "+++"), "+++");
}

TEST(StringGetAfter, Nominal) {
  EXPECT_EQ(StringGetAfter("test", ""), "test");
  EXPECT_EQ(StringGetAfter("test", "notinit"), "");
  EXPECT_EQ(StringGetAfter("test", "e"), "st");
  EXPECT_EQ(StringGetAfter("test, multiple tests", "test"), "s");
  EXPECT_EQ(StringGetAfter("", ""), "");
  EXPECT_EQ(StringGetAfter("path/to/dataset/sub1/image.png", "sub1/"),
            "image.png");
}

TEST(StringSplit, Nominal) {
  const std::vector<std::string> list1 = StringSplit("1,2,3,4,5 , 6", ",");
  EXPECT_EQ(list1.size(), 6);
  EXPECT_EQ(list1[0], "1");
  EXPECT_EQ(list1[1], "2");
  EXPECT_EQ(list1[2], "3");
  EXPECT_EQ(list1[3], "4");
  EXPECT_EQ(list1[4], "5 ");
  EXPECT_EQ(list1[5], " 6");
  const std::vector<std::string> list2 = StringSplit("1,2,3,4,5 , 6", "");
  EXPECT_EQ(list2.size(), 1);
  EXPECT_EQ(list2[0], "1,2,3,4,5 , 6");
  const std::vector<std::string> list3 = StringSplit("1,,2,,3,4,5 , 6", ",");
  EXPECT_EQ(list3.size(), 6);
  EXPECT_EQ(list3[0], "1");
  EXPECT_EQ(list3[1], "2");
  EXPECT_EQ(list3[2], "3");
  EXPECT_EQ(list3[3], "4");
  EXPECT_EQ(list3[4], "5 ");
  EXPECT_EQ(list3[5], " 6");
  const std::vector<std::string> list4 = StringSplit("1,,2,,3,4,5 , 6", ",,");
  EXPECT_EQ(list4.size(), 6);
  EXPECT_EQ(list4[0], "1");
  EXPECT_EQ(list4[1], "2");
  EXPECT_EQ(list4[2], "3");
  EXPECT_EQ(list4[3], "4");
  EXPECT_EQ(list4[4], "5 ");
  EXPECT_EQ(list4[5], " 6");
  const std::vector<std::string> list5 = StringSplit("1,,2,,3,4,5 , 6", ", ");
  EXPECT_EQ(list5.size(), 6);
  EXPECT_EQ(list5[0], "1");
  EXPECT_EQ(list5[1], "2");
  EXPECT_EQ(list5[2], "3");
  EXPECT_EQ(list5[3], "4");
  EXPECT_EQ(list5[4], "5");
  EXPECT_EQ(list5[5], "6");
  const std::vector<std::string> list6 = StringSplit(",1,,2,,3,4,5 , 6 ", ", ");
  EXPECT_EQ(list6.size(), 8);
  EXPECT_EQ(list6[0], "");
  EXPECT_EQ(list6[1], "1");
  EXPECT_EQ(list6[2], "2");
  EXPECT_EQ(list6[3], "3");
  EXPECT_EQ(list6[4], "4");
  EXPECT_EQ(list6[5], "5");
  EXPECT_EQ(list6[6], "6");
  EXPECT_EQ(list6[7], "");
}

TEST(StringStartsWith, Nominal) {
  EXPECT_FALSE(StringStartsWith("", ""));
  EXPECT_FALSE(StringStartsWith("a", ""));
  EXPECT_FALSE(StringStartsWith("", "a"));
  EXPECT_TRUE(StringStartsWith("a", "a"));
  EXPECT_TRUE(StringStartsWith("aa", "a"));
  EXPECT_TRUE(StringStartsWith("aa", "aa"));
  EXPECT_TRUE(StringStartsWith("aaaaaaaaa", "aa"));
}

TEST(StringStrim, Nominal) {
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

TEST(StringLeftString, Nominal) {
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

TEST(StringStrimRight, Nominal) {
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

TEST(StringToLower, Nominal) {
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

TEST(StringToUpper, Nominal) {
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

TEST(StringContains, Nominal) {
  EXPECT_TRUE(StringContains("", ""));
  EXPECT_TRUE(StringContains("a", ""));
  EXPECT_TRUE(StringContains("a", "a"));
  EXPECT_TRUE(StringContains("ab", "a"));
  EXPECT_TRUE(StringContains("ab", "ab"));
  EXPECT_FALSE(StringContains("", "a"));
  EXPECT_FALSE(StringContains("ab", "c"));
}

}  // namespace
}  // namespace colmap
