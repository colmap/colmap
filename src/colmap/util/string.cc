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

#include "colmap/util/string.h"

#include <algorithm>
#include <cstdarg>
#include <fstream>
#include <sstream>

#include <boost/algorithm/string.hpp>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace colmap {
namespace {

// The StringAppendV function is borrowed from Google under the BSD license:
//
// Copyright 2012 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following disclaimer
//       in the documentation and/or other materials provided with the
//       distribution.
//     * Neither the name of Google Inc. nor the names of its
//       contributors may be used to endorse or promote products derived from
//       this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

void StringAppendV(std::string* dst, const char* format, va_list ap) {
  // First try with a small fixed size buffer.
  static const int kFixedBufferSize = 1024;
  char fixed_buffer[kFixedBufferSize];

  // It is possible for methods that use a va_list to invalidate
  // the data in it upon use.  The fix is to make a copy
  // of the structure before using it and use that copy instead.
  va_list backup_ap;
  va_copy(backup_ap, ap);
  int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
  va_end(backup_ap);

  if (result < kFixedBufferSize) {
    if (result >= 0) {
      // Normal case - everything fits.
      dst->append(fixed_buffer, result);
      return;
    }

#ifdef _MSC_VER
    // Error or MSVC running out of space.  MSVC 8.0 and higher
    // can be asked about space needed with the special idiom below:
    va_copy(backup_ap, ap);
    result = vsnprintf(nullptr, 0, format, backup_ap);
    va_end(backup_ap);
#endif

    if (result < 0) {
      // Just an error.
      return;
    }
  }

  // Increase the buffer size to the size requested by vsnprintf,
  // plus one for the closing \0.
  const int variable_buffer_size = result + 1;
  std::unique_ptr<char[]> variable_buffer(new char[variable_buffer_size]);

  // Restore the va_list before we use it again.
  va_copy(backup_ap, ap);
  result =
      vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
  va_end(backup_ap);

  if (result >= 0 && result < variable_buffer_size) {
    dst->append(variable_buffer.get(), result);
  }
}

bool IsNotWhiteSpace(const int character) {
  return character != ' ' && character != '\n' && character != '\r' &&
         character != '\t';
}

}  // namespace

namespace internal {
#ifdef _WIN32
std::string CodePageToUTF8Win(const std::string& str, unsigned int code_page) {
  int wide_len = MultiByteToWideChar(code_page, 0, str.c_str(), -1, nullptr, 0);
  if (wide_len <= 0) return "";

  std::wstring wide_str(wide_len, L'\0');
  MultiByteToWideChar(code_page, 0, str.c_str(), -1, &wide_str[0], wide_len);

  int utf8_len = WideCharToMultiByte(
      CP_UTF8, 0, wide_str.c_str(), -1, nullptr, 0, nullptr, nullptr);
  if (utf8_len <= 0) return "";

  std::string utf8_str(utf8_len, '\0');
  WideCharToMultiByte(CP_UTF8,
                      0,
                      wide_str.c_str(),
                      -1,
                      &utf8_str[0],
                      utf8_len,
                      nullptr,
                      nullptr);

  if (!utf8_str.empty() && utf8_str.back() == '\0') {
    utf8_str.pop_back();
  }

  return utf8_str;
};

std::string UTF8ToCodePageWin(const std::string& str, unsigned int code_page) {
  int wide_len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
  if (wide_len <= 0) return "";

  std::wstring wide_str(wide_len, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wide_str[0], wide_len);

  int local_len = WideCharToMultiByte(
      code_page, 0, wide_str.c_str(), -1, nullptr, 0, nullptr, nullptr);
  if (local_len <= 0) return "";

  std::string local_str(local_len, '\0');
  WideCharToMultiByte(code_page,
                      0,
                      wide_str.c_str(),
                      -1,
                      &local_str[0],
                      local_len,
                      nullptr,
                      nullptr);

  if (!local_str.empty() && local_str.back() == '\0') {
    local_str.pop_back();
  }

  return local_str;
}

#endif
}  // namespace internal

std::string StringPrintf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  std::string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  return result;
}

std::string StringReplace(const std::string& str,
                          const std::string& old_str,
                          const std::string& new_str) {
  if (old_str.empty()) {
    return str;
  }
  size_t position = 0;
  std::string mod_str = str;
  while ((position = mod_str.find(old_str, position)) != std::string::npos) {
    mod_str.replace(position, old_str.size(), new_str);
    position += new_str.size();
  }
  return mod_str;
}

std::string StringGetAfter(const std::string& str, const std::string& key) {
  if (key.empty()) {
    return str;
  }
  std::size_t found = str.rfind(key);
  if (found != std::string::npos) {
    return str.substr(found + key.length(),
                      str.length() - (found + key.length()));
  }
  return "";
}

std::vector<std::string> StringSplit(const std::string& str,
                                     const std::string& delim) {
  std::vector<std::string> elems;
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  boost::split(elems, str, boost::is_any_of(delim), boost::token_compress_on);
  return elems;
}

bool StringStartsWith(const std::string& str, const std::string& prefix) {
  return !prefix.empty() && prefix.size() <= str.size() &&
         str.substr(0, prefix.size()) == prefix;
}

void StringLeftTrim(std::string* str) {
  str->erase(str->begin(),
             std::find_if(str->begin(), str->end(), IsNotWhiteSpace));
}

void StringRightTrim(std::string* str) {
  str->erase(std::find_if(str->rbegin(), str->rend(), IsNotWhiteSpace).base(),
             str->end());
}

void StringTrim(std::string* str) {
  StringLeftTrim(str);
  StringRightTrim(str);
}

void StringToLower(std::string* str) {
  std::transform(str->begin(), str->end(), str->begin(), ::tolower);
}

void StringToUpper(std::string* str) {
  std::transform(str->begin(), str->end(), str->begin(), ::toupper);
}

bool StringContains(const std::string& str, const std::string& sub_str) {
  return str.find(sub_str) != std::string::npos;
}

std::string PlatformToUTF8(const std::string& str) {
#ifdef _WIN32
  return internal::CodePageToUTF8Win(str, GetACP());
#else
  // Assume UTF-8 on POSIX systems
  return input;
#endif
}

std::string UTF8ToPlatform(const std::string& str) {
#ifdef _WIN32
  return internal::UTF8ToCodePageWin(str, GetACP());
#else
  // On POSIX, assume UTF-8 is the system encoding
  return utf8_str;
#endif
}

}  // namespace colmap
