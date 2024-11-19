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

#include "colmap/util/enum_to_string.h"

#include <string>

#include <gtest/gtest.h>

namespace colmap {
namespace {

MAKE_ENUM(MyEnum, 0, ClassA, ClassB);
MAKE_ENUM_CLASS_OVERLOAD_STREAM(MyEnumClass, -1, UNDEFINED, ClassA, ClassB);

TEST(MakeEnum, Nominal) {
  EXPECT_EQ(ClassA, 0);
  EXPECT_EQ(ClassB, 1);
  EXPECT_EQ(MyEnumToString(ClassA), "ClassA");
  EXPECT_EQ(MyEnumToString(ClassB), "ClassB");
}

TEST(MakeEnumClass, Nominal) {
  EXPECT_EQ(static_cast<int>(MyEnumClass::UNDEFINED), -1);
  EXPECT_EQ(static_cast<int>(MyEnumClass::ClassA), 0);
  EXPECT_EQ(static_cast<int>(MyEnumClass::ClassB), 1);
  EXPECT_EQ(MyEnumClassToString(-1), "UNDEFINED");
  EXPECT_EQ(MyEnumClassToString(0), "ClassA");
  EXPECT_EQ(MyEnumClassToString(1), "ClassB");
  std::ostringstream stream;
  stream << MyEnumClass::ClassA;
  EXPECT_EQ(stream.str(), "ClassA");
}

}  // namespace
}  // namespace colmap
