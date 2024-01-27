#pragma once

#include "colmap/util/misc.h"

#include <exception>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename T>
inline std::string ToString(T msg) {
  return std::to_string(msg);
}

inline std::string ToString(std::string msg) { return msg; }

inline std::string ToString(const char* msg) { return std::string(msg); }

inline const char* __ColmapGetConstFileBaseName(const char* file) {
  const char* base = strrchr(file, '/');
  if (!base) {
    base = strrchr(file, '\\');
  }
  return base ? (base + 1) : file;
}

template <typename T>
inline T TemplateException(const char* file,
                           const int line,
                           const std::string& txt) {
  std::stringstream ss;
  ss << "[" << __ColmapGetConstFileBaseName(file) << ":" << line << "] " << txt;
  return T(ss.str());
}

inline std::string __GetConditionString(const char* cond_str) {
  std::stringstream ss;
  ss << "Condition Failed: " << cond_str;
  return ss.str();
}

inline std::string __GetCheckString(const char* cond_str) {
  std::stringstream ss;
  ss << "Check Failed: " << cond_str;
  return ss.str();
}

inline std::string __MergeTwoConstChar(const char* expr1, const char* expr2) {
  return (std::string(expr1) + std::string(" ") + expr2);
}

inline void __ThrowCheckImpl(const char* file,
                             const int line,
                             const bool result,
                             const char* expr_str) {
  if (!result) {
    throw TemplateException<py::value_error>(
        file, line, __GetCheckString(expr_str).c_str());
  }
}

inline void __ThrowCheckImplMsg(const char* file,
                                const int line,
                                const bool result,
                                const char* expr_str,
                                const std::string& msg) {
  if (!result) {
    std::stringstream ss;
    ss << expr_str << " : " << msg;
    std::string m = ss.str();
    throw TemplateException<py::value_error>(
        file, line, __GetCheckString(m.c_str()));
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
    throw TemplateException<py::value_error>(
        file, line, __GetCheckString(msg.c_str()));
  }
}

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

#define THROW_CHECK(expr) __ThrowCheckImpl(__FILE__, __LINE__, (expr), #expr);

#define THROW_CHECK_MSG(expr, msg) \
  __ThrowCheckImplMsg(__FILE__, __LINE__, (expr), #expr, ToString(msg))

#define THROW_CHECK_OP(name, op, val1, val2) \
  __ThrowCheckOpImpl(                        \
      __FILE__, __LINE__, (val1 op val2), val1, val2, #val1, #val2, #op);

#define THROW_CHECK_EQ(val1, val2) THROW_CHECK_OP(_EQ, ==, val1, val2)
#define THROW_CHECK_NE(val1, val2) THROW_CHECK_OP(_NE, !=, val1, val2)
#define THROW_CHECK_LE(val1, val2) THROW_CHECK_OP(_LE, <=, val1, val2)
#define THROW_CHECK_LT(val1, val2) THROW_CHECK_OP(_LT, <, val1, val2)
#define THROW_CHECK_GE(val1, val2) THROW_CHECK_OP(_GE, >=, val1, val2)
#define THROW_CHECK_GT(val1, val2) THROW_CHECK_OP(_GT, >, val1, val2)

#define THROW_CHECK_FILE_EXISTS(path) \
  THROW_CHECK_MSG(ExistsFile(path),   \
                  std::string("File ") + (path) + " does not exist.");

#define THROW_CHECK_DIR_EXISTS(path) \
  THROW_CHECK_MSG(ExistsDir(path),   \
                  std::string("Directory ") + (path) + " does not exist.");

#define THROW_CHECK_FILE_OPEN(path)                   \
  THROW_CHECK_MSG(                                    \
      std::ofstream(path, std::ios::trunc).is_open(), \
      std::string(": Could not open ") + (path) +     \
          ". Is the path a directory or does the parent dir not exist?");

#define THROW_CHECK_HAS_FILE_EXTENSION(path, ext) \
  THROW_CHECK_MSG(HasFileExtension(path, ext),    \
                  std::string("Path ") + (path) + \
                      " does not match file extension " + (ext) + ".");
