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

#include "colmap/util/logging.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

std::string PrintingFn(const std::string& message) {
  if (message.empty()) {
    LOG(FATAL_THROW) << "Error in PrintingFn";
  }
  return message;
}

void ThrowCheck(const bool cond) { THROW_CHECK(cond) << "Error!"; }

void ThrowCheckEqual(const int val) { THROW_CHECK_EQ(val, 1) << "Error!"; }

TEST(ExceptionLogging, Nominal) {
  EXPECT_NO_THROW(ThrowCheck(true));
  EXPECT_THROW(ThrowCheck(false), std::invalid_argument);
  EXPECT_NO_THROW(ThrowCheckEqual(1));
  EXPECT_THROW(ThrowCheckEqual(0), std::invalid_argument);
  EXPECT_THROW(THROW_CHECK_NOTNULL(nullptr), std::invalid_argument);
  EXPECT_THROW({ LOG(FATAL_THROW) << "Error!"; }, std::invalid_argument);
  EXPECT_THROW(
      { LOG_FATAL_THROW(std::logic_error) << "Error!"; }, std::logic_error);
}

TEST(ExceptionLogging, Nested) {
  EXPECT_NO_THROW(PrintingFn("message"));
  EXPECT_THROW(PrintingFn(""), std::invalid_argument);
  EXPECT_THROW(
      { LOG(FATAL_THROW) << "Error: " << PrintingFn("message"); },
      std::invalid_argument);
  EXPECT_THROW(
      { LOG(FATAL_THROW) << "Error: " << PrintingFn(""); },
      std::invalid_argument);
}

}  // namespace
}  // namespace colmap
