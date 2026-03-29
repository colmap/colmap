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

#include "colmap/util/misc.h"

#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(VectorContainsValue, Nominal) {
  EXPECT_TRUE(VectorContainsValue<int>({1, 2, 3}, 1));
  EXPECT_FALSE(VectorContainsValue<int>({2, 3}, 1));
}

TEST(VectorContainsDuplicateValues, Nominal) {
  EXPECT_FALSE(VectorContainsDuplicateValues<int>({}));
  EXPECT_FALSE(VectorContainsDuplicateValues<int>({1}));
  EXPECT_FALSE(VectorContainsDuplicateValues<int>({1, 2}));
  EXPECT_FALSE(VectorContainsDuplicateValues<int>({1, 2, 3}));
  EXPECT_TRUE(VectorContainsDuplicateValues<int>({1, 1, 2, 3}));
  EXPECT_TRUE(VectorContainsDuplicateValues<int>({1, 1, 2, 2, 3}));
  EXPECT_TRUE(VectorContainsDuplicateValues<int>({1, 2, 3, 3}));
  EXPECT_FALSE(VectorContainsDuplicateValues<std::string>({"a"}));
  EXPECT_FALSE(VectorContainsDuplicateValues<std::string>({"a", "b"}));
  EXPECT_TRUE(VectorContainsDuplicateValues<std::string>({"a", "a"}));
}

TEST(CSVToVector, Int) {
  const std::vector<int> list1 = CSVToVector<int>("1, 2, 3 , 4,5,6 ");
  EXPECT_EQ(list1.size(), 6);
  EXPECT_EQ(list1[0], 1);
  EXPECT_EQ(list1[1], 2);
  EXPECT_EQ(list1[2], 3);
  EXPECT_EQ(list1[3], 4);
  EXPECT_EQ(list1[4], 5);
  EXPECT_EQ(list1[5], 6);
  const std::vector<int> list2 = CSVToVector<int>("1; 2; 3 ; 4;5;6 ");
  EXPECT_EQ(list2.size(), 6);
  EXPECT_EQ(list2[0], 1);
  EXPECT_EQ(list2[1], 2);
  EXPECT_EQ(list2[2], 3);
  EXPECT_EQ(list2[3], 4);
  EXPECT_EQ(list2[4], 5);
  EXPECT_EQ(list2[5], 6);
  const std::vector<int> list3 = CSVToVector<int>("1;, 2;; 3 ; 4;5;6 ");
  EXPECT_EQ(list3.size(), 6);
  EXPECT_EQ(list3[0], 1);
  EXPECT_EQ(list3[1], 2);
  EXPECT_EQ(list3[2], 3);
  EXPECT_EQ(list3[3], 4);
  EXPECT_EQ(list3[4], 5);
  EXPECT_EQ(list3[5], 6);
}

TEST(CSVToVector, IntEmpty) {
  EXPECT_THAT(CSVToVector<int>(""), testing::IsEmpty());
  EXPECT_THAT(CSVToVector<int>(" "), testing::IsEmpty());
}

TEST(CSVToVector, IntInvalid) {
  const std::vector<int> list = CSVToVector<int>("1, 2, invalid, 3");
  EXPECT_EQ(list.size(), 0);
}

TEST(CSVToVector, Float) {
  const std::vector<float> list1 =
      CSVToVector<float>("1.5, 2.7, 3.14 , 4.0,5.9,6.2 ");
  EXPECT_EQ(list1.size(), 6);
  EXPECT_FLOAT_EQ(list1[0], 1.5f);
  EXPECT_FLOAT_EQ(list1[1], 2.7f);
  EXPECT_FLOAT_EQ(list1[2], 3.14f);
  EXPECT_FLOAT_EQ(list1[3], 4.0f);
  EXPECT_FLOAT_EQ(list1[4], 5.9f);
  EXPECT_FLOAT_EQ(list1[5], 6.2f);
  const std::vector<float> list2 = CSVToVector<float>("1.5; 2.7; 3.14");
  EXPECT_EQ(list2.size(), 3);
  EXPECT_FLOAT_EQ(list2[0], 1.5f);
  EXPECT_FLOAT_EQ(list2[1], 2.7f);
  EXPECT_FLOAT_EQ(list2[2], 3.14f);
}

TEST(CSVToVector, FloatEmpty) {
  EXPECT_THAT(CSVToVector<float>(""), testing::IsEmpty());
  EXPECT_THAT(CSVToVector<float>(" "), testing::IsEmpty());
}

TEST(CSVToVector, FloatInvalid) {
  const std::vector<float> list = CSVToVector<float>("1.5, 2.7, invalid, 3.14");
  EXPECT_EQ(list.size(), 0);
}

TEST(CSVToVector, Double) {
  const std::vector<double> list1 =
      CSVToVector<double>("1.5, 2.7, 3.14159 , 4.0,5.9,6.28318 ");
  EXPECT_EQ(list1.size(), 6);
  EXPECT_DOUBLE_EQ(list1[0], 1.5);
  EXPECT_DOUBLE_EQ(list1[1], 2.7);
  EXPECT_DOUBLE_EQ(list1[2], 3.14159);
  EXPECT_DOUBLE_EQ(list1[3], 4.0);
  EXPECT_DOUBLE_EQ(list1[4], 5.9);
  EXPECT_DOUBLE_EQ(list1[5], 6.28318);
  const std::vector<double> list2 = CSVToVector<double>("1.5; 2.7; 3.14159");
  EXPECT_EQ(list2.size(), 3);
  EXPECT_DOUBLE_EQ(list2[0], 1.5);
  EXPECT_DOUBLE_EQ(list2[1], 2.7);
  EXPECT_DOUBLE_EQ(list2[2], 3.14159);
}

TEST(CSVToVector, DoubleEmpty) {
  EXPECT_THAT(CSVToVector<double>(""), testing::IsEmpty());
  EXPECT_THAT(CSVToVector<double>(" "), testing::IsEmpty());
}

TEST(CSVToVector, DoubleInvalid) {
  const std::vector<double> list =
      CSVToVector<double>("1.5, 2.7, invalid, 3.14");
  EXPECT_EQ(list.size(), 0);
}

TEST(CSVToVector, String) {
  const std::vector<std::string> list1 =
      CSVToVector<std::string>("foo, bar, baz , qux,hello,world ");
  EXPECT_EQ(list1.size(), 6);
  EXPECT_EQ(list1[0], "foo");
  EXPECT_EQ(list1[1], "bar");
  EXPECT_EQ(list1[2], "baz");
  EXPECT_EQ(list1[3], "qux");
  EXPECT_EQ(list1[4], "hello");
  EXPECT_EQ(list1[5], "world");
  const std::vector<std::string> list2 =
      CSVToVector<std::string>("foo; bar; baz");
  EXPECT_EQ(list2.size(), 3);
  EXPECT_EQ(list2[0], "foo");
  EXPECT_EQ(list2[1], "bar");
  EXPECT_EQ(list2[2], "baz");
  const std::vector<std::string> list3 =
      CSVToVector<std::string>("foo;, bar;; baz");
  EXPECT_EQ(list3.size(), 3);
  EXPECT_EQ(list3[0], "foo");
  EXPECT_EQ(list3[1], "bar");
  EXPECT_EQ(list3[2], "baz");
}

TEST(CSVToVector, StringEmpty) {
  EXPECT_THAT(CSVToVector<std::string>(""), testing::IsEmpty());
  EXPECT_THAT(CSVToVector<std::string>(" "), testing::IsEmpty());
}

TEST(CSVToVector, StringWithSpaces) {
  const std::vector<std::string> list =
      CSVToVector<std::string>(" hello , world , test ");
  EXPECT_EQ(list.size(), 3);
  EXPECT_EQ(list[0], "hello");
  EXPECT_EQ(list[1], "world");
  EXPECT_EQ(list[2], "test");
}

TEST(VectorToCSV, Nominal) {
  EXPECT_EQ(VectorToCSV<int>({}), "");
  EXPECT_EQ(VectorToCSV<int>({1}), "1");
  EXPECT_EQ(VectorToCSV<int>({1, 2, 3}), "1, 2, 3");
}

TEST(RemoveCommandLineArgument, Nominal) {
  int argc = 3;
  std::unique_ptr<char[]> arg1(new char[4]);
  std::memcpy(arg1.get(), "abc", 4 * sizeof(char));
  std::unique_ptr<char[]> arg2(new char[4]);
  std::memcpy(arg2.get(), "def", 4 * sizeof(char));
  std::unique_ptr<char[]> arg3(new char[4]);
  std::memcpy(arg3.get(), "ghi", 4 * sizeof(char));
  std::vector<char*> argv = {arg1.get(), arg2.get(), arg3.get()};

  RemoveCommandLineArgument("abcd", &argc, argv.data());
  EXPECT_EQ(argc, 3);
  EXPECT_EQ(std::string(argv[0]), "abc");
  EXPECT_EQ(std::string(argv[1]), "def");
  EXPECT_EQ(std::string(argv[2]), "ghi");

  RemoveCommandLineArgument("def", &argc, argv.data());
  EXPECT_EQ(argc, 2);
  EXPECT_EQ(std::string(argv[0]), "abc");
  EXPECT_EQ(std::string(argv[1]), "ghi");

  RemoveCommandLineArgument("ghi", &argc, argv.data());
  EXPECT_EQ(argc, 1);
  EXPECT_EQ(std::string(argv[0]), "abc");

  RemoveCommandLineArgument("abc", &argc, argv.data());
  EXPECT_EQ(argc, 0);

  RemoveCommandLineArgument("abc", &argc, argv.data());
  EXPECT_EQ(argc, 0);
}

}  // namespace
}  // namespace colmap
