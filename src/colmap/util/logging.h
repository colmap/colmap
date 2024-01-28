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

#pragma once

#include "colmap/util/string.h"

#include <exception>
#include <iostream>
#include <string>

#include <glog/logging.h>

// Option checker macros. In contrast to glog, this function does not abort the
// program, but simply returns false on failure.
#define CHECK_OPTION_IMPL(expr) \
  __CheckOptionImpl(__FILE__, __LINE__, (expr), #expr)
#define CHECK_OPTION(expr)                                     \
  if (!__CheckOptionImpl(__FILE__, __LINE__, (expr), #expr)) { \
    return false;                                              \
  }
#define CHECK_OPTION_OP(name, op, val1, val2) \
  if (!__CheckOptionOpImpl(__FILE__,          \
                           __LINE__,          \
                           (val1 op val2),    \
                           val1,              \
                           val2,              \
                           #val1,             \
                           #val2,             \
                           #op)) {            \
    return false;                             \
  }
#define CHECK_OPTION_EQ(val1, val2) CHECK_OPTION_OP(_EQ, ==, val1, val2)
#define CHECK_OPTION_NE(val1, val2) CHECK_OPTION_OP(_NE, !=, val1, val2)
#define CHECK_OPTION_LE(val1, val2) CHECK_OPTION_OP(_LE, <=, val1, val2)
#define CHECK_OPTION_LT(val1, val2) CHECK_OPTION_OP(_LT, <, val1, val2)
#define CHECK_OPTION_GE(val1, val2) CHECK_OPTION_OP(_GE, >=, val1, val2)
#define CHECK_OPTION_GT(val1, val2) CHECK_OPTION_OP(_GT, >, val1, val2)

// Option checker macros. In contrast to glog, this function does not abort the
// program, but simply throws an exception on failure.
#define THROW_EXCEPTION(exception, msg) \
  throw TemplateException<exception>(__FILE__, __LINE__, ToString(msg));

#define THROW_CUSTOM_CHECK_MSG(condition, exception, msg) \
  if (!(condition))                                       \
    throw TemplateException<exception>(                   \
        __FILE__,                                         \
        __LINE__,                                         \
        __GetCheckString(#condition) + std::string(" ") + ToString(msg));

#define THROW_CUSTOM_CHECK(condition, exception) \
  if (!(condition))                              \
    throw TemplateException<exception>(          \
        __FILE__, __LINE__, __GetCheckString(#condition));

#define THROW_CHECK(expr) __ThrowCheckImpl(__FILE__, __LINE__, !!(expr), #expr);

#define THROW_CHECK_MSG(expr, msg) \
  __ThrowCheckImplMsg(__FILE__, __LINE__, !!(expr), #expr, ToString(msg))

#define THROW_CHECK_NOTNULL(val) \
  __ThrowCheckNotNull(__FILE__, __LINE__, (val), #val)

#define THROW_CHECK_OP(name, op, val1, val2) \
  __ThrowCheckOpImpl(                        \
      __FILE__, __LINE__, (val1 op val2), val1, val2, #val1, #val2, #op);

#define THROW_CHECK_EQ(val1, val2) THROW_CHECK_OP(_EQ, ==, val1, val2)
#define THROW_CHECK_NE(val1, val2) THROW_CHECK_OP(_NE, !=, val1, val2)
#define THROW_CHECK_LE(val1, val2) THROW_CHECK_OP(_LE, <=, val1, val2)
#define THROW_CHECK_LT(val1, val2) THROW_CHECK_OP(_LT, <, val1, val2)
#define THROW_CHECK_GE(val1, val2) THROW_CHECK_OP(_GE, >=, val1, val2)
#define THROW_CHECK_GT(val1, val2) THROW_CHECK_OP(_GT, >, val1, val2)

namespace colmap {

// Initialize glog at the beginning of the program.
void InitializeGlog(char** argv);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

const char* __GetConstFileBaseName(const char* file);

bool __CheckOptionImpl(const char* file,
                       int line,
                       bool result,
                       const char* expr_str);

template <typename T1, typename T2>
bool __CheckOptionOpImpl(const char* file,
                         const int line,
                         const bool result,
                         const T1& val1,
                         const T2& val2,
                         const char* val1_str,
                         const char* val2_str,
                         const char* op_str) {
  if (result) {
    return true;
  } else {
    LOG(ERROR) << StringPrintf("[%s:%d] Check failed: %s %s %s (%s vs. %s)",
                               __GetConstFileBaseName(file),
                               line,
                               val1_str,
                               op_str,
                               val2_str,
                               std::to_string(val1).c_str(),
                               std::to_string(val2).c_str());
    return false;
  }
}

template <typename T>
inline std::string ToString(T msg) {
  return std::to_string(msg);
}

inline std::string ToString(std::string msg) { return msg; }

inline std::string ToString(const char* msg) { return std::string(msg); }

template <typename T>
inline T TemplateException(const char* file,
                           const int line,
                           const std::string& txt) {
  return T(StringPrintf(
      "[%s:%d] %s", __GetConstFileBaseName(file), line, txt.c_str()));
}

inline std::string __GetCheckString(const char* cond_str) {
  return "Check Failed: " + std::string(cond_str);
}

inline void __ThrowCheckImpl(const char* file,
                             const int line,
                             const bool result,
                             const char* expr_str) {
  if (!result) {
    throw TemplateException<std::invalid_argument>(
        file, line, __GetCheckString(expr_str).c_str());
  }
}

inline void __ThrowCheckImplMsg(const char* file,
                                const int line,
                                const bool result,
                                const char* expr_str,
                                const std::string& msg) {
  if (!result) {
    std::string m = std::string(expr_str) + " : " + msg;
    throw TemplateException<std::invalid_argument>(
        file, line, __GetCheckString(m.c_str()));
  }
}

template <typename T>
T __ThrowCheckNotNull(const char* file,
                      const int line,
                      T&& t,
                      const char* name) {
  if (t == nullptr) {
    std::string msg = "\"" + std::string(name) + "\" Must be non nullptr";
    throw TemplateException<std::invalid_argument>(
        file, line, __GetCheckString(msg.c_str()));
  } else {
    return std::forward<T>(t);
  }
}

template <typename T1, typename T2>
void __ThrowCheckOpImpl(const char* file,
                        const int line,
                        const bool result,
                        const T1& val1,
                        const T2& val2,
                        const char* val1_str,
                        const char* val2_str,
                        const char* op_str) {
  if (!result) {
    std::stringstream ss;
    ss << val1_str << " " << op_str << " " << val2_str << " (" << val1
       << " vs. " << val2 << ")";
    std::string msg = ss.str();
    throw TemplateException<std::invalid_argument>(
        file, line, __GetCheckString(msg.c_str()));
  }
}

}  // namespace colmap
