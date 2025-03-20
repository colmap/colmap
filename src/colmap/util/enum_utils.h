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

#pragma once

#include "colmap/util/logging.h"

#include <iostream>
#include <memory>

#include <boost/preprocessor.hpp>

namespace colmap {

// Custom macro for enum to/from string support. Only enum structs / classes
// with consecutive indexes are supported.
//
// Example:
// [Reference]: enum class MyEnum {C1, C2, C3};
// [New code]: MAKE_ENUM_CLASS(MyEnum, 0, C1, C2, C3);
//             MyEnumToString(MyEnum::C1);  -> "C1"
//             MyEnumFromString("C2");      -> MyEnum::C2

#define ENUM_TO_STRING_PROCESS_ELEMENT(r, start_idx, idx, elem) \
  case ((idx) + (start_idx)):                                   \
    return BOOST_PP_STRINGIZE(elem);
#define ENUM_FROM_STRING_PROCESS_ELEMENT(r, name, idx, elem) \
  if (str == BOOST_PP_STRINGIZE(elem)) {                     \
    return name::elem;                                       \
  }

#define DEFINE_ENUM_TO_FROM_STRING(name, start_idx, ...)                     \
  [[maybe_unused]] static std::string_view name##ToString(int value) {       \
    switch (value) {                                                         \
      BOOST_PP_SEQ_FOR_EACH_I(ENUM_TO_STRING_PROCESS_ELEMENT,                \
                              start_idx,                                     \
                              BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__));        \
      default:                                                               \
        throw std::runtime_error("Unknown value: " + std::to_string(value) + \
                                 " for enum: " + BOOST_PP_STRINGIZE(name));  \
    }                                                                        \
  }                                                                          \
  [[maybe_unused]] static std::string_view name##ToString(name value) {      \
    return name##ToString(static_cast<int>(value));                          \
  }                                                                          \
  [[maybe_unused]] static name name##FromString(std::string_view str) {      \
    BOOST_PP_SEQ_FOR_EACH_I(ENUM_FROM_STRING_PROCESS_ELEMENT,                \
                            name,                                            \
                            BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__));          \
    throw std::runtime_error("Unknown string value: " + std::string(str) +   \
                             " for enum: " + BOOST_PP_STRINGIZE(name));      \
  }

#define ENUM_PROCESS_ELEMENT(r, start_idx, idx, elem) \
  elem = (idx) + (start_idx),

#define ENUM_VALUES(start_idx, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(          \
      ENUM_PROCESS_ELEMENT, start_idx, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define MAKE_ENUM(name, start_idx, ...)              \
  enum name { ENUM_VALUES(start_idx, __VA_ARGS__) }; \
  DEFINE_ENUM_TO_FROM_STRING(name, start_idx, __VA_ARGS__)

#define MAKE_ENUM_CLASS(name, start_idx, ...)              \
  enum class name { ENUM_VALUES(start_idx, __VA_ARGS__) }; \
  DEFINE_ENUM_TO_FROM_STRING(name, start_idx, __VA_ARGS__)

// This only works for non-nested enum classes.
#define MAKE_ENUM_CLASS_OVERLOAD_STREAM(name, start_idx, ...)     \
  MAKE_ENUM_CLASS(name, start_idx, __VA_ARGS__);                  \
  inline std::ostream& operator<<(std::ostream& os, name value) { \
    return os << name##ToString(value);                           \
  }

}  // namespace colmap
