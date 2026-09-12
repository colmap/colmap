// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
#include <vector>

namespace colmap {

// Locale-independent string-to-double conversion.
// Always uses '.' as decimal separator regardless of LC_NUMERIC.
double StringToDouble(const std::string& str);

// Format string by replacing embedded format specifiers with their respective
// values, see `printf` for more details. This is a modified implementation
// of Google's BSD-licensed StringPrintf function.
std::string StringPrintf(const char* format, ...);

// Replace all occurrences of `old_str` with `new_str` in the given string.
std::string StringReplace(const std::string& str,
                          const std::string& old_str,
                          const std::string& new_str);

// Get substring of string after the last occurrence of the search key.
std::string StringGetAfter(const std::string& str, const std::string& key);

// Split string into list of words using the given delimiters.
std::vector<std::string> StringSplit(const std::string& str,
                                     const std::string& delim);

// Check whether a string starts with a certain prefix.
bool StringStartsWith(const std::string& str, const std::string& prefix);

// Remove whitespace from string on both, left, or right sides.
void StringTrim(std::string* str);
void StringLeftTrim(std::string* str);
void StringRightTrim(std::string* str);

// Convert string to lower/upper case.
void StringToLower(std::string* str);
void StringToUpper(std::string* str);

// Check whether the sub-string is contained in the given string.
bool StringContains(const std::string& str, const std::string& sub_str);

// Convert a string from the platform's default encoding to UTF-8.
// On Windows: converts from ANSI code page (ACP) to UTF-8.
// On POSIX: assumes the input is already UTF-8 and returns it unchanged.
std::string PlatformToUTF8(const std::string& str);

// Convert a UTF-8 encoded string to the platform's default encoding.
// On Windows: converts to ANSI code page (ACP).
// On POSIX: returns the input unchanged.
std::string UTF8ToPlatform(const std::string& str);

namespace internal {
#ifdef _WIN32
// Convert a string from the specified Windows code page to UTF-8.
//
// This function is used internally for unit testing to verify roundtrip
// correctness of encoding conversions independent of the system code page.
std::string CodePageToUTF8Win(const std::string& str, unsigned int code_page);

// Convert a UTF-8 string to the specified Windows code page.
//
// This function is used internally for unit testing to verify roundtrip
// correctness of encoding conversions independent of the system code page.
std::string UTF8ToCodePageWin(const std::string& str, unsigned int code_page);
#endif
}  // namespace internal

}  // namespace colmap
