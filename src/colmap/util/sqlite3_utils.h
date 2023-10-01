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

#include <cstdio>
#include <cstdlib>
#include <string>

#include <sqlite3.h>

namespace colmap {

inline int SQLite3CallHelper(const int result_code,
                             const std::string& filename,
                             const int line_number) {
  switch (result_code) {
    case SQLITE_OK:
    case SQLITE_ROW:
    case SQLITE_DONE:
      return result_code;
    default:
      fprintf(stderr,
              "SQLite error [%s, line %i]: %s\n",
              filename.c_str(),
              line_number,
              sqlite3_errstr(result_code));
      throw std::runtime_error("SQLite error");
  }
}

#define SQLITE3_CALL(func) SQLite3CallHelper(func, __FILE__, __LINE__)

#define SQLITE3_EXEC(database, sql, callback)                     \
  {                                                               \
    char* err_msg = nullptr;                                      \
    const int result_code =                                       \
        sqlite3_exec(database, sql, callback, nullptr, &err_msg); \
    if (result_code != SQLITE_OK) {                               \
      fprintf(stderr,                                             \
              "SQLite error [%s, line %i]: %s\n",                 \
              __FILE__,                                           \
              __LINE__,                                           \
              err_msg);                                           \
      sqlite3_free(err_msg);                                      \
    }                                                             \
  }

}  // namespace colmap
