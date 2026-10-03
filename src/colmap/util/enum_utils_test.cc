// SPDX-License-Identifier: BSD-3-Clause

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
  using NestedEnum = NestedEnumHolder::NestedEnum;
  EXPECT_THAT(NestedEnumHolder::NestedEnumValues(),
              testing::ElementsAre(NestedEnum::VAL_A, NestedEnum::VAL_B));
  EXPECT_THAT(NestedEnumHolder::NestedEnumStrings(),
              testing::ElementsAre("VAL_A", "VAL_B"));
  EXPECT_THAT(NestedEnumHolder::NestedEnumValues(
                  [](NestedEnum value) { return value != NestedEnum::VAL_A; }),
              testing::ElementsAre(NestedEnum::VAL_B));
  EXPECT_THAT(NestedEnumHolder::NestedEnumStrings(
                  [](NestedEnum value) { return value != NestedEnum::VAL_A; }),
              testing::ElementsAre("VAL_B"));
}

TEST(MakeEnumClass, Nested) {
  using NestedEnum = NestedEnumClassHolder::NestedEnum;
  EXPECT_THAT(NestedEnumClassHolder::NestedEnumValues(),
              testing::ElementsAre(
                  NestedEnum::UNDEFINED, NestedEnum::VAL_A, NestedEnum::VAL_B));
  EXPECT_THAT(NestedEnumClassHolder::NestedEnumStrings(),
              testing::ElementsAre("UNDEFINED", "VAL_A", "VAL_B"));
  EXPECT_THAT(NestedEnumClassHolder::NestedEnumValues([](NestedEnum value) {
                return value != NestedEnum::UNDEFINED;
              }),
              testing::ElementsAre(NestedEnum::VAL_A, NestedEnum::VAL_B));
  EXPECT_THAT(NestedEnumClassHolder::NestedEnumStrings([](NestedEnum value) {
                return value != NestedEnum::UNDEFINED;
              }),
              testing::ElementsAre("VAL_A", "VAL_B"));
}

}  // namespace
}  // namespace colmap
