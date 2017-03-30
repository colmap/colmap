// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_UTIL_SQLITE3_UTILS_
#define COLMAP_SRC_UTIL_SQLITE3_UTILS_

#include <cstdio>
#include <cstdlib>
#include <string>

#include "ext/SQLite/sqlite3.h"

namespace colmap {

inline int SQLite3CallHelper(const int result_code, const std::string& filename,
                             const int line_number) {
  switch (result_code) {
    case SQLITE_OK:
    case SQLITE_ROW:
    case SQLITE_DONE:
      return result_code;
    default:
      fprintf(stderr, "SQLite error [%s, line %i]: %s\n", filename.c_str(),
              line_number, sqlite3_errstr(result_code));
      exit(EXIT_FAILURE);
  }
}

#define SQLITE3_CALL(func) SQLite3CallHelper(func, __FILE__, __LINE__)

#define SQLITE3_EXEC(database, sql, callback)                                 \
  {                                                                           \
    char* err_msg = nullptr;                                                  \
    int rc = sqlite3_exec(database, sql, callback, nullptr, &err_msg);        \
    if (rc != SQLITE_OK) {                                                    \
      fprintf(stderr, "SQLite error [%s, line %i]: %s\n", __FILE__, __LINE__, \
              err_msg);                                                       \
      sqlite3_free(err_msg);                                                  \
    }                                                                         \
  }

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_SQLITE3_UTILS_
