// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <boost/preprocessor.hpp>

// Custom macro for enum to/from string support. Only enum structs / classes
// with consecutive indexes are supported.
//
// Example:
// [Reference]: enum class MyEnum {C1, C2, C3};
// [New code]: MAKE_ENUM_CLASS(MyEnum, 0, C1, C2, C3);
//             MyEnumToString(MyEnum::C1);  -> "C1"
//             MyEnumFromString("C2");      -> MyEnum::C2
//             MyEnumValues();              -> {MyEnum::C1, MyEnum::C2, ...}
//             MyEnumStrings();             -> {"C1", "C2", "C3"}

#define ENUM_TO_STRING_PROCESS_ELEMENT(r, start_idx, idx, elem) \
  case ((idx) + (start_idx)):                                   \
    return BOOST_PP_STRINGIZE(elem);
#define ENUM_FROM_STRING_PROCESS_ELEMENT(r, name, idx, elem) \
  if (str == BOOST_PP_STRINGIZE(elem)) {                     \
    return name::elem;                                       \
  }
#define ENUM_VALUE_ELEMENT(r, name, elem) name::elem,

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
  }                                                                          \
  [[maybe_unused]] static std::vector<name> name##Values(                    \
      bool (*filter)(name) = nullptr) {                                      \
    static const std::vector<name> values = {BOOST_PP_SEQ_FOR_EACH(          \
        ENUM_VALUE_ELEMENT, name, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))};   \
    if (filter == nullptr) {                                                 \
      return values;                                                         \
    }                                                                        \
    std::vector<name> filtered;                                              \
    filtered.reserve(values.size());                                         \
    for (const name value : values) {                                        \
      if (filter(value)) {                                                   \
        filtered.push_back(value);                                           \
      }                                                                      \
    }                                                                        \
    return filtered;                                                         \
  }                                                                          \
  [[maybe_unused]] static std::vector<std::string_view> name##Strings(       \
      bool (*filter)(name) = nullptr) {                                      \
    const std::vector<name> values = name##Values(filter);                   \
    std::vector<std::string_view> strings;                                   \
    strings.reserve(values.size());                                          \
    for (const name value : values) {                                        \
      strings.push_back(name##ToString(value));                              \
    }                                                                        \
    return strings;                                                          \
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
