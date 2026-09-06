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

#include "colmap/util/enum_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

MAKE_ENUM(MyEnum, 0, VAL_A, VAL_B);
MAKE_ENUM_CLASS_OVERLOAD_STREAM(MyEnumClass, -1, UNDEFINED, VAL_A, VAL_B);

struct NestedEnumHolder {
  MAKE_ENUM(NestedEnum, 0, VAL_A, VAL_B);
};

struct NestedEnumClassHolder {
  MAKE_ENUM_CLASS(NestedEnum, -1, UNDEFINED, VAL_A, VAL_B);
};

TEST(MakeEnum, Nominal) {
  EXPECT_EQ(VAL_A, 0);
  EXPECT_EQ(VAL_B, 1);
  EXPECT_EQ(MyEnumToString(VAL_A), "VAL_A");
  EXPECT_EQ(MyEnumToString(VAL_B), "VAL_B");
  EXPECT_EQ(MyEnumFromString("VAL_A"), VAL_A);
  EXPECT_EQ(MyEnumFromString("VAL_B"), VAL_B);
  std::ostringstream stream;
  stream << VAL_A;
  EXPECT_EQ(stream.str(), "0");
  EXPECT_THAT(MyEnumValues(), testing::ElementsAre(VAL_A, VAL_B));
  EXPECT_THAT(MyEnumStrings(), testing::ElementsAre("VAL_A", "VAL_B"));
  EXPECT_THAT(MyEnumValues([](MyEnum value) { return value != VAL_A; }),
              testing::ElementsAre(VAL_B));
  EXPECT_THAT(MyEnumStrings([](MyEnum value) { return value != VAL_A; }),
              testing::ElementsAre("VAL_B"));
}

TEST(MakeEnumClass, Nominal) {
  EXPECT_EQ(static_cast<int>(MyEnumClass::UNDEFINED), -1);
  EXPECT_EQ(static_cast<int>(MyEnumClass::VAL_A), 0);
  EXPECT_EQ(static_cast<int>(MyEnumClass::VAL_B), 1);
  EXPECT_EQ(MyEnumClassToString(-1), "UNDEFINED");
  EXPECT_EQ(MyEnumClassToString(0), "VAL_A");
  EXPECT_EQ(MyEnumClassToString(1), "VAL_B");
  EXPECT_EQ(MyEnumClassToString(MyEnumClass::UNDEFINED), "UNDEFINED");
  EXPECT_EQ(MyEnumClassToString(MyEnumClass::VAL_A), "VAL_A");
  EXPECT_EQ(MyEnumClassToString(MyEnumClass::VAL_B), "VAL_B");
  EXPECT_EQ(MyEnumClassFromString("UNDEFINED"), MyEnumClass::UNDEFINED);
  EXPECT_EQ(MyEnumClassFromString("VAL_A"), MyEnumClass::VAL_A);
  EXPECT_EQ(MyEnumClassFromString("VAL_B"), MyEnumClass::VAL_B);
  std::ostringstream stream;
  stream << MyEnumClass::VAL_A;
  EXPECT_EQ(stream.str(), "VAL_A");
  EXPECT_THAT(
      MyEnumClassValues(),
      testing::ElementsAre(
          MyEnumClass::UNDEFINED, MyEnumClass::VAL_A, MyEnumClass::VAL_B));
  EXPECT_THAT(MyEnumClassStrings(),
              testing::ElementsAre("UNDEFINED", "VAL_A", "VAL_B"));
  EXPECT_THAT(MyEnumClassValues([](MyEnumClass value) {
                return value != MyEnumClass::UNDEFINED;
              }),
              testing::ElementsAre(MyEnumClass::VAL_A, MyEnumClass::VAL_B));
  EXPECT_THAT(MyEnumClassStrings([](MyEnumClass value) {
                return value != MyEnumClass::UNDEFINED;
              }),
              testing::ElementsAre("VAL_A", "VAL_B"));
}

TEST(MakeEnum, Nested) {
  EXPECT_EQ(
      NestedEnumHolder::NestedEnumToString(NestedEnumHolder::NestedEnum::VAL_A),
      "VAL_A");
  EXPECT_EQ(
      NestedEnumHolder::NestedEnumToString(NestedEnumHolder::NestedEnum::VAL_B),
      "VAL_B");
  EXPECT_EQ(NestedEnumHolder::NestedEnumFromString("VAL_A"),
            NestedEnumHolder::NestedEnum::VAL_A);
  EXPECT_EQ(NestedEnumHolder::NestedEnumFromString("VAL_B"),
            NestedEnumHolder::NestedEnum::VAL_B);
  std::ostringstream stream;
  stream << NestedEnumHolder::NestedEnum::VAL_A;
  EXPECT_EQ(stream.str(), "0");
  EXPECT_THAT(NestedEnumHolder::NestedEnumValues(),
              testing::ElementsAre(NestedEnumHolder::NestedEnum::VAL_A,
                                   NestedEnumHolder::NestedEnum::VAL_B));
  EXPECT_THAT(NestedEnumHolder::NestedEnumStrings(),
              testing::ElementsAre("VAL_A", "VAL_B"));
  EXPECT_THAT(NestedEnumHolder::NestedEnumValues(
                  [](NestedEnumHolder::NestedEnum value) {
                    return value != NestedEnumHolder::NestedEnum::VAL_A;
                  }),
              testing::ElementsAre(NestedEnumHolder::NestedEnum::VAL_B));
  EXPECT_THAT(NestedEnumHolder::NestedEnumStrings(
                  [](NestedEnumHolder::NestedEnum value) {
                    return value != NestedEnumHolder::NestedEnum::VAL_A;
                  }),
              testing::ElementsAre("VAL_B"));
}

TEST(MakeEnumClass, Nested) {
  // Note: no stream test here, because the operator<< generated by
  // MAKE_ENUM_CLASS_OVERLOAD_STREAM does not support nested enums.
  EXPECT_EQ(NestedEnumClassHolder::NestedEnumToString(-1), "UNDEFINED");
  EXPECT_EQ(NestedEnumClassHolder::NestedEnumToString(0), "VAL_A");
  EXPECT_EQ(NestedEnumClassHolder::NestedEnumToString(1), "VAL_B");
  EXPECT_EQ(NestedEnumClassHolder::NestedEnumToString(
                NestedEnumClassHolder::NestedEnum::UNDEFINED),
            "UNDEFINED");
  EXPECT_EQ(NestedEnumClassHolder::NestedEnumToString(
                NestedEnumClassHolder::NestedEnum::VAL_A),
            "VAL_A");
  EXPECT_EQ(NestedEnumClassHolder::NestedEnumFromString("UNDEFINED"),
            NestedEnumClassHolder::NestedEnum::UNDEFINED);
  EXPECT_EQ(NestedEnumClassHolder::NestedEnumFromString("VAL_A"),
            NestedEnumClassHolder::NestedEnum::VAL_A);
  EXPECT_THAT(NestedEnumClassHolder::NestedEnumValues(),
              testing::ElementsAre(NestedEnumClassHolder::NestedEnum::UNDEFINED,
                                   NestedEnumClassHolder::NestedEnum::VAL_A,
                                   NestedEnumClassHolder::NestedEnum::VAL_B));
  EXPECT_THAT(NestedEnumClassHolder::NestedEnumStrings(),
              testing::ElementsAre("UNDEFINED", "VAL_A", "VAL_B"));
  EXPECT_THAT(NestedEnumClassHolder::NestedEnumValues(
                  [](NestedEnumClassHolder::NestedEnum value) {
                    return value !=
                           NestedEnumClassHolder::NestedEnum::UNDEFINED;
                  }),
              testing::ElementsAre(NestedEnumClassHolder::NestedEnum::VAL_A,
                                   NestedEnumClassHolder::NestedEnum::VAL_B));
  EXPECT_THAT(NestedEnumClassHolder::NestedEnumStrings(
                  [](NestedEnumClassHolder::NestedEnum value) {
                    return value !=
                           NestedEnumClassHolder::NestedEnum::UNDEFINED;
                  }),
              testing::ElementsAre("VAL_A", "VAL_B"));
}

}  // namespace
}  // namespace colmap
